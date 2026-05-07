//! Wikitext → plaintext extraction for the MediaWiki source adapter.
//!
//! Raw wikitext is what `pages-articles-multistream.xml.bz2` ships,
//! and what the chunker used to slice verbatim — including
//! `{{cite ...}}` templates, `<ref>` tags, navbox boilerplate, and
//! end-of-article reference / external-links / see-also sections.
//! Sampling the prior build's chunks.bin showed ~30 % of chunks were
//! reference-laden noise; another non-trivial fraction was infobox /
//! navbox boilerplate. With the embedder as the bottleneck, every
//! chunk dropped is a proportional speedup, so we now run the page
//! body through [`parse_wiki_text_2`] before constructing the
//! [`SourceRecord`] — the chunker downstream sees plaintext only.
//!
//! What the AST walk emits:
//! - **Text / CharacterEntity** verbatim
//! - **Heading** as `\n\n<text>\n\n` — but if the heading text matches
//!   one of [`SKIP_SECTIONS`] (References, External links, See also,
//!   Notes, Bibliography, Further reading, …), the whole section is
//!   skipped: subsequent nodes are dropped until the next heading at
//!   the same or shallower level.
//! - **Link** as the anchor text (or the target if no anchor)
//! - **UnorderedList / OrderedList / DefinitionList** as `\n- item`
//!   lines
//! - **ParagraphBreak / HorizontalDivider** as `\n\n`
//! - **Bold / Italic / BoldItalic** are formatting toggles — the
//!   surrounding `Text` carries the actual content
//! - **Preformatted** is recursed into
//!
//! What the AST walk drops:
//! - **Template** entirely (covers `{{cite}}`, `{{infobox}}`,
//!   `{{navbox}}`, `{{authority control}}`, …)
//! - **Tag** entirely (extension tags: `<ref>`, `<gallery>`, `<math>`,
//!   `<syntaxhighlight>`, …)
//! - **Category / Image / File / Redirect / MagicWord / Parameter**
//! - **ExternalLink** (anchor text rarely useful out of context;
//!   rendered URLs make embeddings dense with junk)
//! - **Table** entirely — extracting cell text in a way that survives
//!   embeddings is a follow-up; today's content is mostly infobox /
//!   data tables that don't read as prose
//! - **Comment / StartTag / EndTag** (HTML stragglers)
//!
//! After the walk, [`normalize_whitespace`] collapses runs of spaces
//! and ≥3 consecutive newlines down to two. The result is what the
//! chunker sees.
//!
//! On parse failure (the 5-second per-page timeout fires, or the
//! parser hits a malformed construct it can't recover from), we fall
//! back to the raw wikitext — better to chunk a noisy page than to
//! drop it entirely.

use parse_wiki_text_2::{Configuration, ConfigurationSource, Node};

/// Enwiki-shaped parser configuration. The `parse_wiki_text_2` crate
/// keeps its own `default` module private (0.2.0), so we inline the
/// same source here. Tracking upstream: refresh from
/// `parse-wiki-text-2/src/default.rs` when bumping the dep major.
fn enwiki_configuration() -> Configuration {
    Configuration::new(&ConfigurationSource {
        category_namespaces: &["category"],
        extension_tags: &[
            "categorytree",
            "ce",
            "charinsert",
            "chem",
            "gallery",
            "graph",
            "hiero",
            "imagemap",
            "indicator",
            "inputbox",
            "mapframe",
            "maplink",
            "math",
            "nowiki",
            "poem",
            "pre",
            "ref",
            "references",
            "score",
            "section",
            "source",
            "syntaxhighlight",
            "templatedata",
            "timeline",
        ],
        file_namespaces: &["file", "image"],
        link_trail: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        magic_words: &[
            "DISAMBIG",
            "FORCETOC",
            "HIDDENCAT",
            "INDEX",
            "NEWSECTIONLINK",
            "NOCC",
            "NOCOLLABORATIONHUBTOC",
            "NOCONTENTCONVERT",
            "NOEDITSECTION",
            "NOGALLERY",
            "NOGLOBAL",
            "NOINDEX",
            "NONEWSECTIONLINK",
            "NOTC",
            "NOTITLECONVERT",
            "NOTOC",
            "STATICREDIRECT",
            "TOC",
        ],
        protocols: &[
            "//",
            "bitcoin:",
            "ftp://",
            "ftps://",
            "geo:",
            "git://",
            "gopher://",
            "http://",
            "https://",
            "irc://",
            "ircs://",
            "magnet:",
            "mailto:",
            "mms://",
            "news:",
            "nntp://",
            "redis://",
            "sftp://",
            "sip:",
            "sips:",
            "sms:",
            "ssh://",
            "svn://",
            "tel:",
            "telnet://",
            "urn:",
            "worldwind://",
            "xmpp:",
        ],
        redirect_magic_words: &["REDIRECT"],
    })
}

/// Heading texts (compared case-insensitively, post-trim) that mark
/// the start of an end-of-article section we skip wholesale. The skip
/// extends until the next heading at the same or shallower level.
const SKIP_SECTIONS: &[&str] = &[
    "references",
    "reference",
    "external links",
    "external link",
    "see also",
    "notes",
    "note",
    "bibliography",
    "further reading",
    "sources",
    "citations",
    "footnotes",
    "literature",
    "works cited",
    "selected works",
    "selected publications",
    "publications",
    "discography",
    "filmography",
];

/// Convert raw wikitext to plaintext suitable for chunking. On any
/// parser error (timeout, malformed input the parser bailed on) returns
/// the raw text unchanged — the caller's chunker can still digest it.
pub fn to_plaintext(wikitext: &str) -> String {
    let cfg = enwiki_configuration();
    let parsed = match cfg.parse(wikitext) {
        Ok(o) => o,
        Err(_) => return wikitext.to_string(),
    };
    let mut buf = String::with_capacity(wikitext.len() / 2);
    walk(&parsed.nodes, &mut buf);
    normalize_whitespace(&buf)
}

fn walk(nodes: &[Node<'_>], buf: &mut String) {
    let mut iter = nodes.iter().peekable();
    while let Some(node) = iter.next() {
        match node {
            Node::Text { value, .. } => buf.push_str(value),

            Node::CharacterEntity { character, .. } => buf.push(*character),

            Node::Heading { level, nodes, .. } => {
                let heading_text = collect_text(nodes);
                let normalized = heading_text.trim().to_lowercase();
                if SKIP_SECTIONS.iter().any(|s| normalized == *s) {
                    skip_until_heading_at_or_above(&mut iter, *level);
                    continue;
                }
                buf.push_str("\n\n");
                buf.push_str(heading_text.trim());
                buf.push_str("\n\n");
            }

            Node::ParagraphBreak { .. } | Node::HorizontalDivider { .. } => {
                buf.push_str("\n\n");
            }

            Node::Link { text, target, .. } => {
                if text.is_empty() {
                    // Self-link / piped-target-only — emit the target
                    // (drop the namespace prefix if any).
                    let anchor = target.rsplit_once(':').map_or(*target, |(_, t)| t);
                    buf.push_str(anchor);
                } else {
                    walk(text, buf);
                }
            }

            Node::UnorderedList { items, .. } | Node::OrderedList { items, .. } => {
                for item in items {
                    buf.push('\n');
                    walk(&item.nodes, buf);
                }
                buf.push('\n');
            }

            Node::DefinitionList { items, .. } => {
                for item in items {
                    buf.push('\n');
                    walk(&item.nodes, buf);
                }
                buf.push('\n');
            }

            Node::Preformatted { nodes, .. } => walk(nodes, buf),

            // Toggle markers carry no content — the surrounding Text
            // is what reads as prose.
            Node::Bold { .. } | Node::Italic { .. } | Node::BoldItalic { .. } => {}

            // Everything else: drop. See module docs for rationale.
            Node::Template { .. }
            | Node::Tag { .. }
            | Node::Category { .. }
            | Node::Image { .. }
            | Node::Redirect { .. }
            | Node::MagicWord { .. }
            | Node::Parameter { .. }
            | Node::ExternalLink { .. }
            | Node::Table { .. }
            | Node::Comment { .. }
            | Node::StartTag { .. }
            | Node::EndTag { .. } => {}
        }
    }
}

/// Advance `iter` past every node up to (but not including) the next
/// `Heading` at level ≤ `cutoff`. Used to drop the body of a
/// SKIP_SECTIONS heading.
fn skip_until_heading_at_or_above<'a, I>(iter: &mut std::iter::Peekable<I>, cutoff: u8)
where
    I: Iterator<Item = &'a Node<'a>>,
{
    while let Some(next) = iter.peek() {
        if let Node::Heading { level, .. } = next
            && *level <= cutoff
        {
            return;
        }
        iter.next();
    }
}

fn collect_text(nodes: &[Node<'_>]) -> String {
    let mut s = String::new();
    walk(nodes, &mut s);
    s
}

/// Collapse runs of intra-line whitespace to single spaces and runs
/// of ≥3 newlines to exactly two. Trims leading/trailing whitespace.
fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut newline_run: u32 = 0;
    let mut last_was_inline_space = false;
    for c in s.chars() {
        if c == '\n' {
            newline_run += 1;
            if newline_run <= 2 {
                out.push('\n');
            }
            last_was_inline_space = false;
        } else if c == ' ' || c == '\t' || c == '\r' {
            // Collapse repeated inline whitespace to a single space.
            if !last_was_inline_space && !out.ends_with('\n') {
                out.push(' ');
                last_was_inline_space = true;
            }
            newline_run = 0;
        } else {
            out.push(c);
            newline_run = 0;
            last_was_inline_space = false;
        }
    }
    out.trim().to_string()
}

/// Report a stripped fraction so callers (and tests) can sanity-check
/// the conversion is doing real work. `original_chars / extracted_chars`
/// — values >= 1.5 mean we're shrinking content meaningfully.
#[cfg(test)]
fn shrink_ratio(original: &str, extracted: &str) -> f64 {
    if extracted.is_empty() {
        return f64::INFINITY;
    }
    original.len() as f64 / extracted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_cite_templates() {
        let wt = "Apollo 11 was the first mission.{{cite book |author=Smith |title=Lunar}} \
                  The astronauts walked on the Moon.{{cite journal|...}}";
        let out = to_plaintext(wt);
        assert!(!out.contains("cite"));
        assert!(!out.contains("Smith"));
        assert!(out.contains("Apollo 11 was the first mission."));
        assert!(out.contains("walked on the Moon"));
    }

    #[test]
    fn strips_ref_tags() {
        let wt = "The mass is 5.97×10²⁴ kg.<ref name=\"foo\">Smith 2020 p. 14</ref> \
                  It orbits at 1 AU.<ref>{{cite book |title=Astronomy}}</ref>";
        let out = to_plaintext(wt);
        assert!(!out.contains("Smith 2020"));
        assert!(!out.contains("ref name"));
        assert!(out.contains("5.97"));
        assert!(out.contains("1 AU"));
    }

    #[test]
    fn strips_categories_and_images() {
        let wt = "Body text here.\n\n[[File:Apollo11.jpg|thumb|right|caption]]\n\n\
                  More body text.\n\n[[Category:NASA]]\n[[Category:1969 events]]";
        let out = to_plaintext(wt);
        assert!(!out.contains("Apollo11.jpg"));
        assert!(!out.contains("Category:"));
        assert!(out.contains("Body text here"));
        assert!(out.contains("More body text"));
    }

    #[test]
    fn strips_navbox_and_authority_control_templates() {
        let wt = "Real prose paragraph.\n\n{{Navboxes |list = {{Foo navbox}} {{Bar navbox}}}}\n\
                  {{Authority control}}\n[[Category:Mammals]]";
        let out = to_plaintext(wt);
        assert!(!out.contains("Navbox"));
        assert!(!out.contains("Authority control"));
        assert!(out.contains("Real prose paragraph"));
    }

    #[test]
    fn skips_references_section_and_everything_after_it_at_same_level() {
        let wt = "The history of X is long.\n\n\
                  ==Origins==\nFounded in 1900.\n\n\
                  ==References==\n<ref name=\"a\"/>\n* Author A. ''Title''.\n\n\
                  ==External links==\n* [http://example.com Example]";
        let out = to_plaintext(wt);
        assert!(out.contains("Origins"));
        assert!(out.contains("Founded in 1900"));
        assert!(!out.contains("References"));
        assert!(!out.contains("External links"));
        assert!(!out.contains("Example"));
    }

    #[test]
    fn skip_section_resumes_at_higher_level_heading() {
        // A `==See also==` (level 2) under `=Top=` (level 1) should be
        // skipped, but the next `=Body=` (level 1) must resume normal
        // emission.
        let wt = "=Top=\nIntro.\n\n==See also==\n* [[Foo]]\n* [[Bar]]\n\n\
                  =Body=\nThe real text.";
        let out = to_plaintext(wt);
        assert!(out.contains("Intro"));
        assert!(out.contains("The real text"));
        assert!(!out.contains("Foo"));
        assert!(!out.contains("Bar"));
    }

    #[test]
    fn keeps_link_anchor_text() {
        let wt = "He met [[Albert Einstein|Einstein]] in 1933 and [[Niels Bohr]].";
        let out = to_plaintext(wt);
        assert!(out.contains("Einstein in 1933"));
        assert!(out.contains("Niels Bohr"));
    }

    #[test]
    fn keeps_list_items_as_lines() {
        let wt = "Important things:\n* First\n* Second\n* Third";
        let out = to_plaintext(wt);
        assert!(out.contains("First"));
        assert!(out.contains("Second"));
        assert!(out.contains("Third"));
    }

    #[test]
    fn normalizes_whitespace() {
        let s = "Foo   bar\n\n\n\n\nbaz\n   \n";
        let n = normalize_whitespace(s);
        assert_eq!(n, "Foo bar\n\nbaz");
    }

    #[test]
    fn empty_wikitext_yields_empty_plaintext() {
        assert_eq!(to_plaintext(""), "");
    }

    #[test]
    fn tables_dropped_entirely() {
        // Wiki table syntax: an infobox-style block. We skip these
        // entirely — extracting cell prose is a follow-up.
        let wt = "Lead paragraph.\n\n\
                  {| class=\"infobox\"\n\
                  | Population || 600\n\
                  | Capital || Foo City\n\
                  |}\n\
                  Body text.";
        let out = to_plaintext(wt);
        assert!(out.contains("Lead paragraph"));
        assert!(out.contains("Body text"));
        assert!(!out.contains("Population"));
        assert!(!out.contains("Foo City"));
    }

    #[test]
    fn shrink_ratio_is_substantial_on_typical_article_shape() {
        // Synthetic article that approximates the noise mix we see in
        // real wikipedia samples. The shrink ratio should be ≥ 1.5
        // (i.e. plaintext is < 67 % of wikitext); otherwise the
        // conversion isn't paying its way.
        let wt = "{{Infobox country |name=Atlantis |population=10000}}\n\n\
                  '''Atlantis''' was a [[mythical|legendary]] island.<ref>{{cite book\n\
                  |title=Timaeus |author=Plato}}</ref> The story has [[Plato]] as its\n\
                  source.<ref name=\"crit\">{{cite journal |title=Critias}}</ref>\n\n\
                  ==History==\nMentioned by [[Plato]] in two dialogues.\n\n\
                  ==See also==\n* [[Lemuria]]\n* [[Mu (lost continent)|Mu]]\n\n\
                  ==References==\n{{reflist}}\n\n\
                  ==External links==\n* [http://example.com Atlantis project]\n\n\
                  {{Authority control}}\n[[Category:Mythical places]]\n[[Category:Plato]]";
        let pt = to_plaintext(wt);
        let ratio = shrink_ratio(wt, &pt);
        assert!(
            ratio >= 1.5,
            "shrink ratio too small: original={} plaintext={} ratio={:.2}\nplaintext:\n{}",
            wt.len(),
            pt.len(),
            ratio,
            pt
        );
        // Sanity: the keep-this content survives.
        assert!(pt.contains("Atlantis"));
        assert!(pt.contains("legendary"));
        assert!(pt.contains("History"));
        assert!(pt.contains("Mentioned by"));
        // The drop-this content is gone.
        assert!(!pt.contains("Infobox"));
        assert!(!pt.contains("cite"));
        assert!(!pt.contains("Authority control"));
        assert!(!pt.contains("Category:"));
        assert!(!pt.contains("Lemuria"));
        assert!(!pt.contains("References"));
        assert!(!pt.contains("External links"));
    }

    #[test]
    #[ignore] // perf-only; not run on default `cargo test`
    fn perf_smoke_on_3kb_article() {
        // Synthetic ~3 KB article — about the median wikipedia size.
        // Ensures parse + walk completes in a sane time on the
        // chunker stage (~6.8 M articles × this ⇒ wall-clock budget).
        let para =
            "The '''[[Apollo program]]''' was a series of missions {{cite book|title=NASA}} \
             carried out by [[NASA]] between 1961 and 1972.<ref>NHistory.</ref> The program \
             aimed to land humans on the [[Moon]].<ref name=\"k\"/> Five further successful \
             landings occurred through 1972, ending with [[Apollo 17]]. ";
        let mut wt = String::new();
        for _ in 0..30 {
            wt.push_str(para);
        }
        wt.push_str(
            "\n\n==References==\n{{reflist}}\n\n==External links==\n* [http://nasa.gov NASA]\n",
        );
        let start = std::time::Instant::now();
        let n = 100;
        for _ in 0..n {
            let _ = to_plaintext(&wt);
        }
        let elapsed = start.elapsed();
        let per_article_ms = elapsed.as_micros() as f64 / 1000.0 / n as f64;
        println!(
            "to_plaintext on {}-byte article: {:.2} ms/article over {} runs",
            wt.len(),
            per_article_ms,
            n,
        );
        // We expect well under 100 ms per article on commodity HW.
        // Anything materially over that means parser is the chunker
        // bottleneck — investigate before shipping.
        assert!(
            per_article_ms < 100.0,
            "parser too slow at {per_article_ms} ms/article",
        );
    }

    #[test]
    fn does_not_panic_on_real_world_messy_input() {
        // Adversarial-ish input mixing all the hairy constructs. The
        // parser has a 5-second timeout; on timeout we fall back to
        // raw text. Either way: must not panic.
        let wt = "{{tl|Foo {{nested}} bar}}\n\n\
                  '''X''' [[Y|Z]] <ref><ref><ref></ref></ref></ref>\n\
                  {| {| {|\n| broken | table\n|}\n";
        let _ = to_plaintext(wt);
    }
}
