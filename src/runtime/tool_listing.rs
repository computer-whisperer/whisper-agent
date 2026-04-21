//! Tool-catalog enumeration and system-prompt listing rendering.
//!
//! Splits the single enumeration primitive — "every tool this thread
//! could reach, with admission status" — out of `Scheduler` so it can
//! power four views from one walk:
//!
//! 1. **Wire `tools:` array** (filtered by `ToolSurface.core_tools`)
//! 2. **System-prompt listing** appended at thread seed
//! 3. **`find_tool`** regex + category filter
//! 4. **`describe_tool`** exact name lookup
//!
//! The pure [`render_listing`] / [`classify_admission`] functions live
//! here without a `Scheduler` dependency so they're unit-testable
//! against fabricated inputs.

use whisper_agent_protocol::permission::AllowMap;
use whisper_agent_protocol::tool_surface::InitialListing;

use crate::tools::mcp::ToolAnnotations;

/// Which bucket a tool came from. Used for grouping in the system-
/// prompt listing and for `find_tool`'s `category` filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCategory {
    /// Server builtin (`pod_*`, `dispatch_thread`, `request_escalation`,
    /// `describe_tool`, `find_tool`).
    Builtin,
    /// From a host-env MCP. `env_name` is the `[[allow.host_env]]`
    /// binding name — which is also the tool's wire-level prefix.
    HostEnv { env_name: String },
    /// From a shared MCP host (named entry in `[allow.mcp_hosts]`,
    /// unprefixed on the wire).
    SharedMcp { host_name: String },
}

impl ToolCategory {
    /// Stable short label used in the system-prompt listing section
    /// header and as the `category` value returned by `find_tool`.
    pub fn label(&self) -> String {
        match self {
            ToolCategory::Builtin => "builtin".to_string(),
            ToolCategory::HostEnv { env_name } => format!("host_env:{env_name}"),
            ToolCategory::SharedMcp { host_name } => format!("shared_mcp:{host_name}"),
        }
    }

    /// High-level bucket (`"builtin"`, `"host_env"`, `"shared_mcp"`),
    /// used by `find_tool`'s `category` filter so the user doesn't
    /// have to know the binding name to filter by origin.
    pub fn coarse(&self) -> &'static str {
        match self {
            ToolCategory::Builtin => "builtin",
            ToolCategory::HostEnv { .. } => "host_env",
            ToolCategory::SharedMcp { .. } => "shared_mcp",
        }
    }
}

/// One tool the thread could reach — either it's currently admitted
/// by scope, or the pod ceiling admits it and it's askable via
/// `request_escalation` AddTool.
///
/// Carries the full schema so `describe_tool` and the core-tools
/// filter can operate off a single walk. `find_tool` and the listing
/// ignore the schema — they use `name`, `description`, and
/// `requires_escalation`.
#[derive(Debug, Clone)]
pub struct AdmissibleTool {
    /// Wire-level tool name (prefixed for host-env tools).
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub annotations: ToolAnnotations,
    pub category: ToolCategory,
    /// `true` when scope denies but the pod ceiling admits — the
    /// model can request admission via `request_escalation`'s
    /// AddTool variant. `false` when the tool is already in scope.
    pub requires_escalation: bool,
}

/// Admission outcome for a specific tool under a (scope, ceiling) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolAdmission {
    /// Scope admits — model can call directly.
    Admitted,
    /// Scope denies but ceiling admits — model can widen via escalation.
    Askable,
    /// Ceiling denies — truly out of reach. Should not appear in any
    /// listing or find result.
    OutOfReach,
}

/// Classify a tool's admission under the thread's effective scope vs.
/// the pod ceiling. Pure — no scheduler state.
pub fn classify_admission(
    thread_scope_tools: &AllowMap<String>,
    pod_ceiling_tools: &AllowMap<String>,
    name: &str,
) -> ToolAdmission {
    let scoped = thread_scope_tools.disposition(&name.to_string()).admits();
    if scoped {
        return ToolAdmission::Admitted;
    }
    let ceiling = pod_ceiling_tools.disposition(&name.to_string()).admits();
    if ceiling {
        ToolAdmission::Askable
    } else {
        ToolAdmission::OutOfReach
    }
}

/// Render the system-prompt listing appended at thread seed.
///
/// - `AllNames` → every admitted-or-askable tool, grouped by source,
///   with an "available via escalation" trailer when any tool is
///   askable AND the thread has an interactive channel.
/// - `CoreOnly` → counts per group + a short summary of core tools.
/// - `None` → empty string.
///
/// Output is plain text; callers append it to the base system prompt
/// (usually with a leading blank line).
pub fn render_listing(
    tools: &[AdmissibleTool],
    mode: InitialListing,
    escalation_available: bool,
    core_tool_names: &[String],
) -> String {
    match mode {
        InitialListing::None => String::new(),
        InitialListing::AllNames => render_all_names(tools, escalation_available),
        InitialListing::CoreOnly => render_core_only(tools, core_tool_names),
    }
}

fn group_by_category(tools: &[AdmissibleTool]) -> Vec<(String, Vec<&AdmissibleTool>)> {
    use std::collections::BTreeMap;
    // Group by `ToolCategory::label()` so the ordering within groups
    // is stable (alphabetical binding names) and distinct categories
    // land in predictable slots. `Builtin` sorts first by leading
    // "b"; host_env / shared_mcp follow in their natural prefix order.
    let mut map: BTreeMap<String, Vec<&AdmissibleTool>> = BTreeMap::new();
    for tool in tools {
        map.entry(tool.category.label()).or_default().push(tool);
    }
    // Within each group, sort by name for determinism.
    let mut out: Vec<(String, Vec<&AdmissibleTool>)> = map.into_iter().collect();
    for (_, v) in out.iter_mut() {
        v.sort_by(|a, b| a.name.cmp(&b.name));
    }
    out
}

fn group_header(label: &str) -> String {
    match label {
        "builtin" => "Builtin tools".to_string(),
        other if other.starts_with("host_env:") => {
            let env = &other["host_env:".len()..];
            format!("Host env `{env}`")
        }
        other if other.starts_with("shared_mcp:") => {
            let host = &other["shared_mcp:".len()..];
            format!("Shared MCP `{host}`")
        }
        other => other.to_string(),
    }
}

pub(crate) fn one_line_description(desc: &str) -> String {
    // Tool descriptions are often multi-sentence; for the listing we
    // want one line so the prompt stays tight. Truncate at the first
    // period/newline and cap at ~120 chars so a GitHub-style "this
    // tool does X (see also: Y)" still fits.
    let first = desc.split(['\n', '.']).next().unwrap_or(desc).trim();
    if first.len() > 120 {
        let mut t = first.chars().take(117).collect::<String>();
        t.push_str("...");
        t
    } else {
        first.to_string()
    }
}

fn render_all_names(tools: &[AdmissibleTool], escalation_available: bool) -> String {
    let admitted: Vec<AdmissibleTool> = tools
        .iter()
        .filter(|t| !t.requires_escalation)
        .cloned()
        .collect();
    let askable: Vec<AdmissibleTool> = tools
        .iter()
        .filter(|t| t.requires_escalation)
        .cloned()
        .collect();

    let mut out = String::new();
    out.push_str("## Available tools\n\n");
    out.push_str(
        "The tools listed below are accessible to you. Full schemas are not inlined — call \
         `describe_tool` with a name to fetch one before first use. You can also search by \
         pattern with `find_tool`.\n\n",
    );

    for (label, entries) in group_by_category(&admitted) {
        out.push_str(&format!("### {}\n", group_header(&label)));
        for t in entries {
            out.push_str(&format!(
                "- `{}` — {}\n",
                t.name,
                one_line_description(&t.description)
            ));
        }
        out.push('\n');
    }

    if escalation_available && !askable.is_empty() {
        out.push_str("### Available via escalation\n");
        out.push_str(
            "These tools are within the pod's allow-ceiling but not currently in your scope. \
             Request them with `request_escalation` (variant `add_tool`) when you need one.\n",
        );
        for (label, entries) in group_by_category(&askable) {
            out.push_str(&format!("\n#### {}\n", group_header(&label)));
            for t in entries {
                out.push_str(&format!(
                    "- `{}` — {}\n",
                    t.name,
                    one_line_description(&t.description)
                ));
            }
        }
        out.push('\n');
    }

    out
}

/// Apply `find_tool`'s filter + pagination to a set of admissible
/// tools. Returns the page of matches and the unpaged total, so the
/// caller can render a "N more, use offset=..." tail.
///
/// Pure — takes the full tool list (from `admissible_and_askable_tools`)
/// and the parsed filter args, no scheduler state. The regex is pre-
/// compiled by the caller (it's a user input that can fail to parse,
/// which is the caller's concern to surface).
pub fn filter_tools_for_find(
    tools: &[AdmissibleTool],
    regex: &regex::Regex,
    category_coarse: Option<&str>,
    include_escalation: bool,
    limit: usize,
    offset: usize,
) -> (Vec<AdmissibleTool>, usize) {
    let matched: Vec<AdmissibleTool> = tools
        .iter()
        .filter(|t| {
            if !include_escalation && t.requires_escalation {
                return false;
            }
            if let Some(cat) = category_coarse
                && cat != t.category.coarse()
            {
                return false;
            }
            regex.is_match(&t.name) || regex.is_match(&t.description)
        })
        .cloned()
        .collect();
    let total = matched.len();
    let page: Vec<AdmissibleTool> = matched.into_iter().skip(offset).take(limit).collect();
    (page, total)
}

/// Render an append-only activation message announcing newly-available
/// tools (by name, with one-line descriptions — schemas are fetched
/// via `describe_tool` on demand). Used when `activation_surface =
/// Announce`.
///
/// The message is a self-contained `## Tools newly available` section
/// so the model can tell it apart from the initial listing at seed
/// time. Grouped by source for readability.
pub fn render_activation_announce(tools: &[AdmissibleTool]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("## Tools newly available\n\n");
    out.push_str(
        "The tools listed below have just become available in this thread's scope. \
         Full schemas aren't inlined — call `describe_tool` with a name to fetch one.\n\n",
    );
    // Reuse the seed-time grouping so the appearance is consistent.
    for (label, entries) in group_by_category(tools) {
        out.push_str(&format!("### {}\n", group_header(&label)));
        for t in entries {
            out.push_str(&format!(
                "- `{}` — {}\n",
                t.name,
                one_line_description(&t.description)
            ));
        }
        out.push('\n');
    }
    out
}

/// Render an append-only activation message that inlines the full
/// schemas of newly-available tools. Used when `activation_surface =
/// InjectSchema`. Trades upfront cost (schema tokens) for zero
/// round-trips to `describe_tool` before first call.
pub fn render_activation_inject(tools: &[AdmissibleTool]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("## Tools newly available\n\n");
    out.push_str(
        "The tools listed below have just become available in this thread's scope. \
         Full schemas are inlined for immediate use.\n\n",
    );
    for t in tools {
        let schema = serde_json::to_string_pretty(&t.input_schema)
            .unwrap_or_else(|e| format!("<schema serialization failed: {e}>"));
        out.push_str(&format!(
            "### `{name}` [{cat}]\n{desc}\n\nInput schema:\n```json\n{schema}\n```\n\n",
            name = t.name,
            cat = t.category.label(),
            desc = t.description
        ));
    }
    out
}

fn render_core_only(tools: &[AdmissibleTool], core_tool_names: &[String]) -> String {
    let mut out = String::new();
    out.push_str("## Available tools\n\n");
    out.push_str(
        "Your wire-level `tools` array carries only a small core set — the rest are discoverable \
         by name via `find_tool` and fetchable by schema via `describe_tool`.\n\n",
    );
    // Counts per coarse bucket.
    let admitted: Vec<&AdmissibleTool> = tools.iter().filter(|t| !t.requires_escalation).collect();
    let builtins = admitted
        .iter()
        .filter(|t| matches!(t.category, ToolCategory::Builtin))
        .count();
    let host_env = admitted
        .iter()
        .filter(|t| matches!(t.category, ToolCategory::HostEnv { .. }))
        .count();
    let shared_mcp = admitted
        .iter()
        .filter(|t| matches!(t.category, ToolCategory::SharedMcp { .. }))
        .count();
    out.push_str(&format!(
        "Admitted: {} builtins, {} host-env MCP tools, {} shared MCP tools.\n\n",
        builtins, host_env, shared_mcp
    ));
    if !core_tool_names.is_empty() {
        out.push_str("Core tools (full schemas loaded): ");
        out.push_str(
            &core_tool_names
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push_str(".\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use whisper_agent_protocol::permission::Disposition;

    fn allowmap_with_overrides(
        default: Disposition,
        entries: &[(&str, Disposition)],
    ) -> AllowMap<String> {
        let overrides = entries.iter().map(|(k, d)| (k.to_string(), *d)).collect();
        AllowMap { default, overrides }
    }

    fn tool(
        name: &str,
        desc: &str,
        cat: ToolCategory,
        requires_escalation: bool,
    ) -> AdmissibleTool {
        AdmissibleTool {
            name: name.into(),
            description: desc.into(),
            input_schema: json!({"type": "object"}),
            annotations: ToolAnnotations::default(),
            category: cat,
            requires_escalation,
        }
    }

    #[test]
    fn classify_admitted_when_scope_allows() {
        let scope = AllowMap::allow_all();
        let ceiling = AllowMap::allow_all();
        assert_eq!(
            classify_admission(&scope, &ceiling, "bash"),
            ToolAdmission::Admitted
        );
    }

    #[test]
    fn classify_askable_when_scope_denies_but_ceiling_admits() {
        let scope = allowmap_with_overrides(Disposition::Allow, &[("bash", Disposition::Deny)]);
        let ceiling = AllowMap::allow_all();
        assert_eq!(
            classify_admission(&scope, &ceiling, "bash"),
            ToolAdmission::Askable
        );
    }

    #[test]
    fn classify_out_of_reach_when_ceiling_denies() {
        let scope = AllowMap::deny_all();
        let ceiling = allowmap_with_overrides(Disposition::Allow, &[("bash", Disposition::Deny)]);
        assert_eq!(
            classify_admission(&scope, &ceiling, "bash"),
            ToolAdmission::OutOfReach
        );
    }

    #[test]
    fn listing_none_is_empty() {
        let out = render_listing(&[], InitialListing::None, false, &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn listing_all_names_groups_by_source() {
        let tools = vec![
            tool(
                "pod_read_file",
                "read a pod file",
                ToolCategory::Builtin,
                false,
            ),
            tool(
                "rustdev_bash",
                "run a shell command",
                ToolCategory::HostEnv {
                    env_name: "rustdev".into(),
                },
                false,
            ),
            tool(
                "create_issue",
                "open a GitHub issue",
                ToolCategory::SharedMcp {
                    host_name: "github".into(),
                },
                false,
            ),
        ];
        let out = render_listing(&tools, InitialListing::AllNames, false, &[]);
        assert!(out.contains("## Available tools"));
        assert!(out.contains("### Builtin tools"));
        assert!(out.contains("`pod_read_file`"));
        assert!(out.contains("### Host env `rustdev`"));
        assert!(out.contains("`rustdev_bash`"));
        assert!(out.contains("### Shared MCP `github`"));
        assert!(out.contains("`create_issue`"));
    }

    #[test]
    fn listing_all_names_includes_escalation_section_when_available() {
        let tools = vec![
            tool("pod_read_file", "read", ToolCategory::Builtin, false),
            tool(
                "rustdev_bash",
                "shell",
                ToolCategory::HostEnv {
                    env_name: "rustdev".into(),
                },
                true,
            ),
        ];
        let out = render_listing(&tools, InitialListing::AllNames, true, &[]);
        assert!(out.contains("Available via escalation"));
        assert!(out.contains("`rustdev_bash`"));
    }

    #[test]
    fn listing_all_names_omits_escalation_section_when_channel_closed() {
        let tools = vec![tool(
            "rustdev_bash",
            "shell",
            ToolCategory::HostEnv {
                env_name: "rustdev".into(),
            },
            true,
        )];
        let out = render_listing(&tools, InitialListing::AllNames, false, &[]);
        assert!(!out.contains("Available via escalation"));
        // Askable tools aren't listed at all on autonomous threads —
        // there's nowhere to ask.
        assert!(!out.contains("rustdev_bash"));
    }

    #[test]
    fn listing_core_only_shows_counts_and_core_set() {
        let tools = vec![
            tool("pod_read_file", "read", ToolCategory::Builtin, false),
            tool("pod_write_file", "write", ToolCategory::Builtin, false),
            tool(
                "rustdev_bash",
                "shell",
                ToolCategory::HostEnv {
                    env_name: "rustdev".into(),
                },
                false,
            ),
        ];
        let core = vec!["describe_tool".to_string(), "find_tool".to_string()];
        let out = render_listing(&tools, InitialListing::CoreOnly, false, &core);
        assert!(out.contains("2 builtins"));
        assert!(out.contains("1 host-env"));
        assert!(out.contains("`describe_tool`"));
        assert!(out.contains("`find_tool`"));
    }

    #[test]
    fn one_line_description_truncates_long_descriptions() {
        let long = "This is a very long description that goes on and on, well past the one-hundred-twenty-character boundary we enforce for listings, and keeps going indefinitely";
        let got = one_line_description(long);
        assert!(got.len() <= 120);
        assert!(got.ends_with("..."));
    }

    #[test]
    fn one_line_description_stops_at_first_sentence() {
        let multi = "Short. Then more stuff.";
        assert_eq!(one_line_description(multi), "Short");
    }

    #[test]
    fn activation_announce_groups_by_source() {
        let tools = vec![
            tool(
                "rustdev_bash",
                "shell",
                ToolCategory::HostEnv {
                    env_name: "rustdev".into(),
                },
                false,
            ),
            tool(
                "create_issue",
                "github",
                ToolCategory::SharedMcp {
                    host_name: "github".into(),
                },
                false,
            ),
        ];
        let out = render_activation_announce(&tools);
        assert!(out.contains("Tools newly available"));
        assert!(out.contains("`rustdev_bash`"));
        assert!(out.contains("`create_issue`"));
        assert!(out.contains("Host env `rustdev`"));
        assert!(out.contains("Shared MCP `github`"));
    }

    #[test]
    fn activation_announce_empty_is_empty() {
        assert!(render_activation_announce(&[]).is_empty());
    }

    fn sample_catalog() -> Vec<AdmissibleTool> {
        vec![
            tool(
                "pod_read_file",
                "read a pod file",
                ToolCategory::Builtin,
                false,
            ),
            tool(
                "pod_write_file",
                "write a pod file",
                ToolCategory::Builtin,
                false,
            ),
            tool(
                "describe_tool",
                "return a tool's schema",
                ToolCategory::Builtin,
                false,
            ),
            tool(
                "rustdev_bash",
                "run a shell command",
                ToolCategory::HostEnv {
                    env_name: "rustdev".into(),
                },
                false,
            ),
            tool(
                "rustdev_edit",
                "edit files in the sandbox",
                ToolCategory::HostEnv {
                    env_name: "rustdev".into(),
                },
                true,
            ),
            tool(
                "create_issue",
                "open a GitHub issue",
                ToolCategory::SharedMcp {
                    host_name: "github".into(),
                },
                false,
            ),
        ]
    }

    #[test]
    fn filter_matches_name_or_description() {
        let cat = sample_catalog();
        let re = regex::Regex::new("write|edit").unwrap();
        let (page, total) = filter_tools_for_find(&cat, &re, None, true, 50, 0);
        assert_eq!(total, 2);
        let names: Vec<&str> = page.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"pod_write_file"));
        assert!(names.contains(&"rustdev_edit"));
    }

    #[test]
    fn filter_excludes_escalation_when_flag_off() {
        let cat = sample_catalog();
        let re = regex::Regex::new("rustdev_").unwrap();
        let (page, total) = filter_tools_for_find(&cat, &re, None, false, 50, 0);
        assert_eq!(total, 1);
        assert_eq!(page[0].name, "rustdev_bash");
    }

    #[test]
    fn filter_honors_coarse_category() {
        let cat = sample_catalog();
        let re = regex::Regex::new(".*").unwrap();
        let (page, _) = filter_tools_for_find(&cat, &re, Some("shared_mcp"), true, 50, 0);
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].name, "create_issue");
    }

    #[test]
    fn filter_paginates() {
        let cat = sample_catalog();
        let re = regex::Regex::new(".*").unwrap();
        let (page1, total) = filter_tools_for_find(&cat, &re, None, true, 2, 0);
        assert_eq!(total, 6);
        assert_eq!(page1.len(), 2);
        let (page2, _) = filter_tools_for_find(&cat, &re, None, true, 2, 4);
        assert_eq!(page2.len(), 2);
    }

    #[test]
    fn filter_unknown_category_returns_empty() {
        let cat = sample_catalog();
        let re = regex::Regex::new(".*").unwrap();
        let (page, total) = filter_tools_for_find(&cat, &re, Some("unknown"), true, 50, 0);
        assert_eq!(total, 0);
        assert!(page.is_empty());
    }

    #[test]
    fn activation_inject_includes_schemas() {
        let tools = vec![tool(
            "rustdev_bash",
            "shell",
            ToolCategory::HostEnv {
                env_name: "rustdev".into(),
            },
            false,
        )];
        let out = render_activation_inject(&tools);
        assert!(out.contains("Tools newly available"));
        assert!(out.contains("`rustdev_bash`"));
        // Schema body is JSON-fenced.
        assert!(out.contains("```json"));
        assert!(out.contains("\"type\""));
    }
}
