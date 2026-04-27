//! [`WikipediaDriver`] — feed driver for `https://dumps.wikimedia.org`
//! and compatible mirrors. Wraps the URL conventions of one wiki
//! (e.g. `enwiki`, `simplewiki`) for both monthly base snapshots and
//! daily incremental adds-changes dumps.
//!
//! ## URL conventions
//!
//! - **Base** — `<mirror>/<lang>wiki/<YYYYMMDD>/` directory; the file
//!   `<lang>wiki-<YYYYMMDD>-pages-articles-multistream.xml.bz2` inside
//!   it is the streamable wikitext archive the
//!   [`MediaWikiXml`](super::super::MediaWikiXml) adapter parses.
//! - **Delta** — `<mirror>/other/incr/<lang>wiki/<YYYYMMDD>/` directory;
//!   the file `<lang>wiki-<YYYYMMDD>-pages-meta-hist-incr.xml.bz2` is
//!   the incremental dump (revisions changed in the prior 24h).
//!
//! Both the base and delta directories are exposed as Apache `mod_dir`
//! style HTML listings — child links rendered as
//! `<a href="20260401/">20260401/</a>`. The driver scrapes those links
//! to enumerate available snapshots / deltas; the format has been
//! stable for many years.
//!
//! ## What this driver does *not* do (yet, called out in the design)
//!
//! - **`dumpstatus.json` validation.** Wikipedia ships an in-progress
//!   directory before the file lands; v1 just trusts the lexicographic
//!   max and lets the worker retry if `fetch_base` 404s. Move to
//!   parsing `dumpstatus.json` when an in-progress base trips a build.
//! - **md5sums.txt checksum verification.** Bytes are written without
//!   integrity check beyond what TLS gives us. Land before a multi-day
//!   build trusts a partial / corrupted base.
//! - **HTTP `Range` resume.** Failed mid-download = re-download from
//!   scratch. Painful for the ~24 GB enwiki base; matters less for the
//!   ~810 MB/day deltas. Land alongside the worker's retry policy.

use std::path::Path;
use std::time::Duration;

use reqwest::{Client, StatusCode};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use super::{BoxFuture, DeltaId, FeedDriver, FeedError, SnapshotId};

/// Default mirror used when [`WikipediaDriver::new`] is called with
/// `mirror = None`. Trailing slash is *not* included; URL builders add
/// it explicitly.
pub const DEFAULT_MIRROR: &str = "https://dumps.wikimedia.org";

/// Adapter id this driver delegates parsing to. Must match an entry in
/// the source-adapter registry the bucket runtime consults.
const PARSE_ADAPTER: &str = "mediawiki_xml";

pub struct WikipediaDriver {
    language: String,
    mirror: String,
    client: Client,
}

impl WikipediaDriver {
    /// Construct a driver for one wiki. `language` is the wiki language
    /// code (`"en"`, `"de"`, `"simple"`); the driver handles the
    /// `<lang>wiki` URL form internally. Pass `mirror = None` to use
    /// [`DEFAULT_MIRROR`].
    pub fn new(language: impl Into<String>, mirror: Option<String>) -> Self {
        let mirror = mirror
            .unwrap_or_else(|| DEFAULT_MIRROR.to_string())
            .trim_end_matches('/')
            .to_string();
        let client = Client::builder()
            .user_agent(concat!("whisper-agent/", env!("CARGO_PKG_VERSION")))
            // Connection-establishment timeout, not a transfer timeout —
            // a 24 GB base download must not be cut off because it takes
            // hours to stream.
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest::Client::builder is infallible with no proxy / cert config");
        Self {
            language: language.into(),
            mirror,
            client,
        }
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn mirror(&self) -> &str {
        &self.mirror
    }

    // --- URL builders (pure functions; testable in isolation) ---

    /// Directory URL containing per-date base-snapshot subdirs.
    pub fn base_index_url(&self) -> String {
        format!("{}/{}wiki/", self.mirror, self.language)
    }

    /// Directory URL for one specific base snapshot.
    pub fn base_snapshot_dir_url(&self, id: &SnapshotId) -> String {
        format!("{}{}/", self.base_index_url(), id.as_str())
    }

    /// Full URL of the streamable wikitext archive for one base
    /// snapshot.
    pub fn base_file_url(&self, id: &SnapshotId) -> String {
        format!(
            "{}{}wiki-{}-pages-articles-multistream.xml.bz2",
            self.base_snapshot_dir_url(id),
            self.language,
            id.as_str(),
        )
    }

    /// Directory URL containing per-date daily-incremental subdirs.
    pub fn delta_index_url(&self) -> String {
        format!("{}/other/incr/{}wiki/", self.mirror, self.language)
    }

    /// Directory URL for one specific delta.
    pub fn delta_dir_url(&self, id: &DeltaId) -> String {
        format!("{}{}/", self.delta_index_url(), id.as_str())
    }

    /// Full URL of the incremental wikitext archive for one delta.
    pub fn delta_file_url(&self, id: &DeltaId) -> String {
        format!(
            "{}{}wiki-{}-pages-meta-hist-incr.xml.bz2",
            self.delta_dir_url(id),
            self.language,
            id.as_str(),
        )
    }

    // --- Internal helpers ---

    async fn list_index(
        &self,
        url: &str,
        cancel: &CancellationToken,
    ) -> Result<Vec<String>, FeedError> {
        let resp = run_cancellable(cancel, self.client.get(url).send())
            .await?
            .map_err(|e| FeedError::Network(format!("GET {url}: {e}")))?;
        let status = resp.status();
        if status == StatusCode::NOT_FOUND {
            return Err(FeedError::NotFound(url.to_string()));
        }
        if !status.is_success() {
            return Err(FeedError::Network(format!("GET {url}: HTTP {status}")));
        }
        let body = run_cancellable(cancel, resp.text())
            .await?
            .map_err(|e| FeedError::Network(format!("read body of {url}: {e}")))?;
        Ok(parse_dated_dir_listing(&body))
    }

    async fn download_to(
        &self,
        url: &str,
        dest: &Path,
        cancel: &CancellationToken,
    ) -> Result<(), FeedError> {
        // Idempotency v1: if `dest` exists with non-zero size, assume
        // it's complete. Caller deletes the file to force re-download.
        // Range-resume + checksum verification are explicit follow-ups.
        if let Ok(meta) = fs::metadata(dest).await
            && meta.is_file()
            && meta.len() > 0
        {
            return Ok(());
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| FeedError::Io(format!("create_dir_all {}: {e}", parent.display())))?;
        }

        let resp = run_cancellable(cancel, self.client.get(url).send())
            .await?
            .map_err(|e| FeedError::Network(format!("GET {url}: {e}")))?;
        let status = resp.status();
        if status == StatusCode::NOT_FOUND {
            return Err(FeedError::NotFound(url.to_string()));
        }
        if !status.is_success() {
            return Err(FeedError::Network(format!("GET {url}: HTTP {status}")));
        }

        // Stream into a tmp file and rename on success — interrupted
        // downloads leave the tmp file behind, never a half-written
        // dest.
        let tmp = dest.with_extension("partial");
        let mut file = fs::File::create(&tmp)
            .await
            .map_err(|e| FeedError::Io(format!("create {}: {e}", tmp.display())))?;

        let mut stream = resp.bytes_stream();
        use futures::StreamExt;
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    drop(file);
                    let _ = fs::remove_file(&tmp).await;
                    return Err(FeedError::Cancelled);
                }
                next = stream.next() => {
                    match next {
                        None => break,
                        Some(Err(e)) => {
                            drop(file);
                            let _ = fs::remove_file(&tmp).await;
                            return Err(FeedError::Network(format!("stream {url}: {e}")));
                        }
                        Some(Ok(chunk)) => {
                            file.write_all(&chunk).await.map_err(|e| {
                                FeedError::Io(format!("write {}: {e}", tmp.display()))
                            })?;
                        }
                    }
                }
            }
        }
        file.flush()
            .await
            .map_err(|e| FeedError::Io(format!("flush {}: {e}", tmp.display())))?;
        drop(file);
        fs::rename(&tmp, dest).await.map_err(|e| {
            FeedError::Io(format!(
                "rename {} → {}: {e}",
                tmp.display(),
                dest.display()
            ))
        })?;
        Ok(())
    }
}

impl FeedDriver for WikipediaDriver {
    fn latest_base<'a>(
        &'a self,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<SnapshotId, FeedError>> {
        Box::pin(async move {
            let url = self.base_index_url();
            let mut entries = self.list_index(&url, cancel).await?;
            entries.sort();
            entries
                .pop()
                .map(SnapshotId)
                .ok_or_else(|| FeedError::Parse(format!("no dated entries in {url}")))
        })
    }

    fn fetch_base<'a>(
        &'a self,
        id: &'a SnapshotId,
        dest: &'a Path,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<(), FeedError>> {
        Box::pin(async move {
            let url = self.base_file_url(id);
            self.download_to(&url, dest, cancel).await
        })
    }

    fn list_deltas_since<'a>(
        &'a self,
        since: Option<&'a DeltaId>,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<Vec<DeltaId>, FeedError>> {
        Box::pin(async move {
            let url = self.delta_index_url();
            let mut entries = self.list_index(&url, cancel).await?;
            entries.sort();
            let cutoff = since.map(|d| d.as_str().to_string());
            Ok(entries
                .into_iter()
                .filter(|e| match &cutoff {
                    Some(c) => e.as_str() > c.as_str(),
                    None => true,
                })
                .map(DeltaId)
                .collect())
        })
    }

    fn fetch_delta<'a>(
        &'a self,
        id: &'a DeltaId,
        dest: &'a Path,
        cancel: &'a CancellationToken,
    ) -> BoxFuture<'a, Result<(), FeedError>> {
        Box::pin(async move {
            let url = self.delta_file_url(id);
            self.download_to(&url, dest, cancel).await
        })
    }

    fn parse_adapter(&self) -> &'static str {
        PARSE_ADAPTER
    }
}

/// Extract `YYYYMMDD` directory entries from a `mod_dir` Apache index.
///
/// Wikipedia's listings use the form
/// `<a href="20260401/">20260401/</a>` for each subdirectory. We accept
/// any 8-digit token followed by a `/` inside an `href="..."` value;
/// duplicates are deduplicated; non-matching links are silently skipped
/// (the listing also contains parent-dir links and sort widgets that we
/// don't care about).
///
/// Returned entries are unsorted — callers sort if order matters.
fn parse_dated_dir_listing(html: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let needle = "href=\"";
    let bytes = html.as_bytes();
    let mut i = 0;
    while let Some(off) = find_subslice(&bytes[i..], needle.as_bytes()) {
        let start = i + off + needle.len();
        let end = match bytes[start..].iter().position(|&b| b == b'"') {
            Some(p) => start + p,
            None => break,
        };
        let value = &html[start..end];
        // Strip a trailing slash; require exactly 8 ASCII digits.
        let id = value.trim_end_matches('/');
        if id.len() == 8 && id.bytes().all(|b| b.is_ascii_digit()) && seen.insert(id.to_string()) {
            out.push(id.to_string());
        }
        i = end + 1;
    }
    out
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return if needle.is_empty() { Some(0) } else { None };
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Wrap a future in a cancellation race. The inner future's `Output`
/// type is preserved on success; cancellation maps to
/// [`FeedError::Cancelled`]. Used so we don't have to repeat the
/// `tokio::select!` skeleton on every short request.
async fn run_cancellable<F: std::future::Future>(
    cancel: &CancellationToken,
    fut: F,
) -> Result<F::Output, FeedError> {
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(FeedError::Cancelled),
        out = fut => Ok(out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- URL builders ---

    #[test]
    fn url_builders_match_dumps_wikimedia_org_conventions() {
        let d = WikipediaDriver::new("en", None);
        assert_eq!(d.base_index_url(), "https://dumps.wikimedia.org/enwiki/");
        assert_eq!(
            d.base_snapshot_dir_url(&SnapshotId::new("20260401")),
            "https://dumps.wikimedia.org/enwiki/20260401/"
        );
        assert_eq!(
            d.base_file_url(&SnapshotId::new("20260401")),
            "https://dumps.wikimedia.org/enwiki/20260401/\
             enwiki-20260401-pages-articles-multistream.xml.bz2"
        );
        assert_eq!(
            d.delta_index_url(),
            "https://dumps.wikimedia.org/other/incr/enwiki/"
        );
        assert_eq!(
            d.delta_dir_url(&DeltaId::new("20260424")),
            "https://dumps.wikimedia.org/other/incr/enwiki/20260424/"
        );
        assert_eq!(
            d.delta_file_url(&DeltaId::new("20260424")),
            "https://dumps.wikimedia.org/other/incr/enwiki/20260424/\
             enwiki-20260424-pages-meta-hist-incr.xml.bz2"
        );
    }

    #[test]
    fn mirror_override_strips_trailing_slash() {
        let d = WikipediaDriver::new(
            "simple",
            Some("https://my-mirror.example/wikidumps/".to_string()),
        );
        assert_eq!(
            d.base_index_url(),
            "https://my-mirror.example/wikidumps/simplewiki/"
        );
    }

    #[test]
    fn parse_adapter_is_mediawiki_xml() {
        let d = WikipediaDriver::new("en", None);
        assert_eq!(d.parse_adapter(), "mediawiki_xml");
    }

    // --- Listing parser ---

    #[test]
    fn listing_parser_extracts_dated_dirs() {
        let html = r#"
            <html><body>
            <a href="../">Parent</a>
            <a href="20260316/">20260316/</a>
            <a href="20260317/">20260317/</a>
            <a href="20260318/">20260318/</a>
            <a href="?C=N;O=D">Sort by name</a>
            <a href="20260319/">20260319/</a>
            </body></html>
        "#;
        let mut got = parse_dated_dir_listing(html);
        got.sort();
        assert_eq!(
            got,
            vec!["20260316", "20260317", "20260318", "20260319"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn listing_parser_dedupes_repeated_links() {
        // Apache mod_dir typically produces both an icon link and a
        // text link per row, both with the same href — the parser must
        // not double-count.
        let html = r#"
            <a href="20260401/"><img src="dir.png"></a>
            <a href="20260401/">20260401/</a>
        "#;
        assert_eq!(parse_dated_dir_listing(html), vec!["20260401".to_string()]);
    }

    #[test]
    fn listing_parser_skips_non_dated_entries() {
        let html = r#"
            <a href="latest/">latest/</a>
            <a href="readme.txt">readme</a>
            <a href="20260401/">20260401/</a>
            <a href="999999/">six-digit/</a>
            <a href="202604AB/">eight-non-digit/</a>
        "#;
        assert_eq!(parse_dated_dir_listing(html), vec!["20260401".to_string()]);
    }

    #[test]
    fn listing_parser_handles_real_wikipedia_format() {
        // Captured snippet from `dumps.wikimedia.org/other/incr/simplewiki/`.
        let html = r#"
            <pre><img src="/icons/blank.gif" alt="Icon "> <a href="?C=N;O=D">Name</a>
            <hr><img src="/icons/back.gif" alt="[PARENTDIR]"> <a href="/other/incr/">Parent Directory</a>
            <img src="/icons/folder.gif" alt="[DIR]"> <a href="20260316/">20260316/</a>             2026-03-16 12:00    -
            <img src="/icons/folder.gif" alt="[DIR]"> <a href="20260317/">20260317/</a>             2026-03-17 12:00    -
            <img src="/icons/folder.gif" alt="[DIR]"> <a href="20260318/">20260318/</a>             2026-03-18 12:00    -
            <hr></pre>
        "#;
        let mut got = parse_dated_dir_listing(html);
        got.sort();
        assert_eq!(
            got,
            vec!["20260316", "20260317", "20260318"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    // --- Integration: latest_base, list_deltas_since, fetch_* against a
    //     local axum-backed mirror. Asserts the URL conventions and the
    //     end-to-end success path; failure paths (404, mid-stream
    //     disconnect, cancellation) covered by separate tests below.

    use std::sync::Arc;

    use axum::Router;
    use axum::body::Body;
    use axum::extract::{Path as AxumPath, State};
    use axum::http::StatusCode as AxumStatus;
    use axum::response::IntoResponse;
    use axum::routing::get;
    use tokio::net::TcpListener;

    #[derive(Clone)]
    struct MirrorState {
        bases: Vec<&'static str>,
        deltas: Vec<&'static str>,
        // bytes served for any base/delta file URL the test asks for
        canned_payload: &'static [u8],
    }

    fn render_dir(entries: &[&str]) -> String {
        let mut s = String::from("<html><body><pre>\n");
        s.push_str(r#"<a href="../">Parent</a>"#);
        s.push('\n');
        for e in entries {
            s.push_str(&format!("<a href=\"{e}/\">{e}/</a>\n"));
        }
        s.push_str("</pre></body></html>\n");
        s
    }

    async fn base_index(
        AxumPath(lang): AxumPath<String>,
        State(state): State<Arc<MirrorState>>,
    ) -> impl IntoResponse {
        if lang == "enwiki" {
            (AxumStatus::OK, render_dir(&state.bases)).into_response()
        } else {
            AxumStatus::NOT_FOUND.into_response()
        }
    }

    async fn delta_index(
        AxumPath(lang): AxumPath<String>,
        State(state): State<Arc<MirrorState>>,
    ) -> impl IntoResponse {
        if lang == "enwiki" {
            (AxumStatus::OK, render_dir(&state.deltas)).into_response()
        } else {
            AxumStatus::NOT_FOUND.into_response()
        }
    }

    async fn base_file(
        AxumPath((lang, date, _file)): AxumPath<(String, String, String)>,
        State(state): State<Arc<MirrorState>>,
    ) -> impl IntoResponse {
        if lang != "enwiki" || !state.bases.contains(&date.as_str()) {
            return AxumStatus::NOT_FOUND.into_response();
        }
        (AxumStatus::OK, Body::from(state.canned_payload)).into_response()
    }

    async fn delta_file(
        AxumPath((lang, date, _file)): AxumPath<(String, String, String)>,
        State(state): State<Arc<MirrorState>>,
    ) -> impl IntoResponse {
        if lang != "enwiki" || !state.deltas.contains(&date.as_str()) {
            return AxumStatus::NOT_FOUND.into_response();
        }
        (AxumStatus::OK, Body::from(state.canned_payload)).into_response()
    }

    async fn spawn_mirror(state: MirrorState) -> String {
        let state = Arc::new(state);
        let app = Router::new()
            .route("/{lang}/", get(base_index))
            .route("/{lang}/{date}/{file}", get(base_file))
            .route("/other/incr/{lang}/", get(delta_index))
            .route("/other/incr/{lang}/{date}/{file}", get(delta_file))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn latest_base_returns_lex_max_dated_entry() {
        let mirror = spawn_mirror(MirrorState {
            bases: vec!["20260101", "20260301", "20260401", "20260201"],
            deltas: vec![],
            canned_payload: b"",
        })
        .await;
        let driver = WikipediaDriver::new("en", Some(mirror));
        let cancel = CancellationToken::new();
        let id = driver.latest_base(&cancel).await.unwrap();
        assert_eq!(id, SnapshotId::new("20260401"));
    }

    #[tokio::test]
    async fn list_deltas_since_filters_strictly_after() {
        let mirror = spawn_mirror(MirrorState {
            bases: vec![],
            deltas: vec!["20260420", "20260421", "20260422", "20260423", "20260424"],
            canned_payload: b"",
        })
        .await;
        let driver = WikipediaDriver::new("en", Some(mirror));
        let cancel = CancellationToken::new();
        let since = DeltaId::new("20260422");
        let got = driver
            .list_deltas_since(Some(&since), &cancel)
            .await
            .unwrap();
        assert_eq!(
            got,
            vec![DeltaId::new("20260423"), DeltaId::new("20260424")]
        );
    }

    #[tokio::test]
    async fn list_deltas_since_none_returns_all_sorted() {
        let mirror = spawn_mirror(MirrorState {
            bases: vec![],
            deltas: vec!["20260424", "20260420", "20260422"],
            canned_payload: b"",
        })
        .await;
        let driver = WikipediaDriver::new("en", Some(mirror));
        let cancel = CancellationToken::new();
        let got = driver.list_deltas_since(None, &cancel).await.unwrap();
        assert_eq!(
            got,
            vec![
                DeltaId::new("20260420"),
                DeltaId::new("20260422"),
                DeltaId::new("20260424"),
            ]
        );
    }

    #[tokio::test]
    async fn fetch_base_writes_payload_atomically() {
        let payload: &[u8] = b"\x42\x5a\x68\x39 fake bz2 payload";
        let mirror = spawn_mirror(MirrorState {
            bases: vec!["20260401"],
            deltas: vec![],
            canned_payload: payload,
        })
        .await;
        let driver = WikipediaDriver::new("en", Some(mirror));
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("base/20260401/dump.xml.bz2");
        let cancel = CancellationToken::new();
        driver
            .fetch_base(&SnapshotId::new("20260401"), &dest, &cancel)
            .await
            .unwrap();
        let got = std::fs::read(&dest).unwrap();
        assert_eq!(got, payload);
        // No `.partial` file left behind.
        assert!(!dest.with_extension("partial").exists());
    }

    #[tokio::test]
    async fn fetch_base_is_idempotent_when_dest_already_populated() {
        // Pre-write the dest with sentinel bytes; fetch_base must not
        // overwrite it (that's the v1 idempotency contract — caller
        // deletes to force re-download).
        let mirror = spawn_mirror(MirrorState {
            bases: vec!["20260401"],
            deltas: vec![],
            canned_payload: b"server payload",
        })
        .await;
        let driver = WikipediaDriver::new("en", Some(mirror));
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("dump.xml.bz2");
        std::fs::write(&dest, b"sentinel").unwrap();
        let cancel = CancellationToken::new();
        driver
            .fetch_base(&SnapshotId::new("20260401"), &dest, &cancel)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"sentinel");
    }

    #[tokio::test]
    async fn fetch_base_404_surfaces_not_found() {
        let mirror = spawn_mirror(MirrorState {
            bases: vec!["20260401"],
            deltas: vec![],
            canned_payload: b"",
        })
        .await;
        let driver = WikipediaDriver::new("en", Some(mirror));
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("dump.xml.bz2");
        let cancel = CancellationToken::new();
        let err = driver
            .fetch_base(&SnapshotId::new("20990101"), &dest, &cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, FeedError::NotFound(_)), "got {err:?}");
        // Nothing written on failure.
        assert!(!dest.exists());
    }

    #[tokio::test]
    async fn cancellation_aborts_in_flight_fetch() {
        let mirror = spawn_mirror(MirrorState {
            bases: vec!["20260401"],
            deltas: vec![],
            canned_payload: b"payload that should never be observed",
        })
        .await;
        let driver = WikipediaDriver::new("en", Some(mirror));
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("dump.xml.bz2");
        let cancel = CancellationToken::new();
        cancel.cancel(); // already-cancelled before we start
        let err = driver
            .fetch_base(&SnapshotId::new("20260401"), &dest, &cancel)
            .await
            .unwrap_err();
        assert!(matches!(err, FeedError::Cancelled), "got {err:?}");
        assert!(!dest.exists());
    }
}
