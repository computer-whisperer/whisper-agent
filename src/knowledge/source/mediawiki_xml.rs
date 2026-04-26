//! [`MediaWikiXml`] — source adapter that streams a MediaWiki XML
//! export (`pages-articles` dumps from <https://dumps.wikimedia.org/>)
//! and emits one [`SourceRecord`] per main-namespace, non-redirect,
//! non-empty article.
//!
//! Why streaming: a current `enwiki-latest-pages-articles.xml.bz2` is
//! ~22 GB compressed / ~95 GB uncompressed. DOM parsing is not an
//! option; we walk events with `quick-xml` and hold only one page in
//! memory at a time. The bz2 path uses `bzip2::read::BzDecoder` so
//! decompression streams in lockstep with parsing.
//!
//! Filtering rules baked into v1 — all tunable later via config:
//! - **Namespace 0 only.** Drops `Talk:` / `User:` / `Template:` etc.
//! - **Redirects dropped.** A `<redirect title="…" />` empty tag inside
//!   `<page>` marks the page as a stub pointing elsewhere.
//! - **Empty pages dropped.** A `<text>` element whose body is empty or
//!   whitespace-only contributes nothing to retrieval.
//!
//! What we do *not* do (deliberately, for v1):
//! - **No wikitext → plaintext.** The chunker and embedder see raw
//!   wikitext, including `[[link]]` / `{{template}}` / table markup.
//!   Embeddings tolerate the noise; sparse search still hits keywords.
//!   Plaintext extraction (likely via `parse_wiki_text`) lands when
//!   RAG quality matters more than pipeline shape.
//! - **No configurable namespace whitelist.** Add when the second use
//!   case appears.
//!
//! `source_id` for emitted records is the article title (e.g.
//! `"Apollo program"`). It survives across rebuilds — `content_hash`
//! still depends on the body, so a renamed-but-otherwise-identical
//! article re-uses chunks.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use quick_xml::Reader;
use quick_xml::escape::unescape;
use quick_xml::events::Event;
use quick_xml::name::QName;

use super::{SourceAdapter, SourceError, SourceRecord};

/// Main-article namespace per MediaWiki convention.
const MAIN_NAMESPACE: i32 = 0;

pub struct MediaWikiXml {
    archive_path: PathBuf,
}

impl MediaWikiXml {
    pub fn new(archive_path: impl Into<PathBuf>) -> Self {
        Self {
            archive_path: archive_path.into(),
        }
    }

    pub fn archive_path(&self) -> &Path {
        &self.archive_path
    }
}

impl SourceAdapter for MediaWikiXml {
    fn enumerate(&self) -> Box<dyn Iterator<Item = Result<SourceRecord, SourceError>> + Send + '_> {
        match open_reader(&self.archive_path) {
            Ok(reader) => Box::new(MediaWikiXmlIter::new(reader)),
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    }
}

/// Detect compression by extension and wrap the file accordingly.
/// Returns a single boxed `BufRead` so the iterator type isn't generic
/// over the compression flavor.
fn open_reader(path: &Path) -> Result<Box<dyn BufRead + Send>, SourceError> {
    let file = File::open(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => SourceError::NotFound(path.display().to_string()),
        _ => SourceError::Io {
            path: path.display().to_string(),
            error: e,
        },
    })?;
    if is_bz2(path) {
        let decoder = bzip2::read::BzDecoder::new(BufReader::new(file));
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn is_bz2(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("bz2"))
}

/// State accumulated while walking the events for a single `<page>`.
#[derive(Default)]
struct PageState {
    title: Option<String>,
    ns: Option<i32>,
    is_redirect: bool,
    text: Option<String>,
}

impl PageState {
    /// Convert a fully-walked page into a `SourceRecord` if it passes
    /// the v1 filters (ns=0, not a redirect, non-empty text).
    fn into_record(self) -> Option<SourceRecord> {
        if self.is_redirect {
            return None;
        }
        if self.ns != Some(MAIN_NAMESPACE) {
            return None;
        }
        let title = self.title?;
        let text = self.text?;
        if text.trim().is_empty() {
            return None;
        }
        Some(SourceRecord::new(title, text))
    }
}

struct MediaWikiXmlIter {
    reader: Reader<Box<dyn BufRead + Send>>,
    /// Reusable scratch buffer for `read_event_into`. Cleared between
    /// events so peak allocation tracks the largest single XML token,
    /// not the cumulative document.
    buf: Vec<u8>,
    /// Separate scratch buffer for `read_text_into` — must not alias
    /// `buf` because the read-text path is invoked from inside a Start
    /// event handler whose event borrows `buf`.
    text_buf: Vec<u8>,
    /// `Some` when we're inside a `<page>` element walking its
    /// children; `None` between pages.
    current: Option<PageState>,
    /// Sticky terminal: once a parser error fires, we surface it once
    /// and then return `None` forever — same convention `MarkdownDir`
    /// uses for its initial-IO error.
    failed: bool,
}

impl MediaWikiXmlIter {
    fn new(reader: Box<dyn BufRead + Send>) -> Self {
        let mut xml = Reader::from_reader(reader);
        // Defaults are sensible for MediaWiki dumps: trim_text=false
        // preserves wikitext whitespace verbatim, expand_empty_elements=
        // false keeps `<redirect ... />` as `Empty` events so we can
        // distinguish self-closing from start/end pairs.
        xml.config_mut().trim_text(false);
        Self {
            reader: xml,
            buf: Vec::new(),
            text_buf: Vec::new(),
            current: None,
            failed: false,
        }
    }
}

impl Iterator for MediaWikiXmlIter {
    type Item = Result<SourceRecord, SourceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        loop {
            self.buf.clear();
            // Copy the tag name out of the event before dispatching —
            // the event borrows from `self.buf`, but `handle_start`
            // re-borrows `self` mutably to drive `read_text_into`.
            let action = match self.reader.read_event_into(&mut self.buf) {
                Ok(Event::Start(start)) => StartAction::Start(start.name().as_ref().to_vec()),
                Ok(Event::Empty(start)) => StartAction::Empty(start.name().as_ref().to_vec()),
                Ok(Event::End(end)) => StartAction::End(end.name().as_ref().to_vec()),
                Ok(Event::Eof) => return None,
                Err(e) => {
                    self.failed = true;
                    return Some(Err(SourceError::Other(format!(
                        "mediawiki xml parse error at byte {}: {e}",
                        self.reader.buffer_position(),
                    ))));
                }
                // Text / CData / Comment / DocType / Decl / PI: ignored
                // outright. The text bodies we care about are read
                // inline by `read_text_into` inside `handle_start`,
                // not via the surrounding event loop.
                _ => StartAction::Skip,
            };

            match action {
                StartAction::Start(name) => {
                    if let Err(e) = self.handle_start(&name) {
                        self.failed = true;
                        return Some(Err(e));
                    }
                }
                StartAction::Empty(name) => {
                    // `<redirect ... />` is the only empty tag we care
                    // about. Other self-closing tags get ignored.
                    if name == b"redirect"
                        && let Some(state) = self.current.as_mut()
                    {
                        state.is_redirect = true;
                    }
                }
                StartAction::End(name) => {
                    if name == b"page"
                        && let Some(state) = self.current.take()
                        && let Some(record) = state.into_record()
                    {
                        return Some(Ok(record));
                    }
                }
                StartAction::Skip => {}
            }
        }
    }
}

/// Simple owned shape for the per-event dispatch in `next()`. Lets us
/// drop the borrowed event before re-borrowing `self` mutably for
/// `read_text_into`.
enum StartAction {
    Start(Vec<u8>),
    Empty(Vec<u8>),
    End(Vec<u8>),
    Skip,
}

impl MediaWikiXmlIter {
    fn handle_start(&mut self, tag: &[u8]) -> Result<(), SourceError> {
        match tag {
            b"page" => {
                self.current = Some(PageState::default());
            }
            b"title" if self.current.is_some() => {
                let title = self.read_inner_text(b"title")?;
                if let Some(state) = self.current.as_mut() {
                    state.title = Some(title);
                }
            }
            b"ns" if self.current.is_some() => {
                let raw = self.read_inner_text(b"ns")?;
                if let Some(state) = self.current.as_mut() {
                    state.ns = raw.trim().parse::<i32>().ok();
                }
            }
            b"text" if self.current.is_some() => {
                // `<text>` appears at `<page>/<revision>/<text>`. We
                // don't track depth — `<text>` doesn't appear elsewhere
                // in the schema, so an unconditional read is correct.
                let text = self.read_inner_text(b"text")?;
                if let Some(state) = self.current.as_mut() {
                    state.text = Some(text);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Read until the matching end tag and decode the inner text. The
    /// quick-xml buffered-reader API uses `read_text_into` (a separate
    /// buffer); after this call the reader sits just past `</tag>`.
    /// Inner XML entities are unescaped (so `&amp;` arrives as `&`).
    fn read_inner_text(&mut self, tag: &[u8]) -> Result<String, SourceError> {
        self.text_buf.clear();
        let qname = QName(tag);
        let bytes_text = self
            .reader
            .read_text_into(qname, &mut self.text_buf)
            .map_err(|e| {
                SourceError::Other(format!(
                    "mediawiki xml: read_text_into({}) at byte {}: {e}",
                    String::from_utf8_lossy(tag),
                    self.reader.buffer_position(),
                ))
            })?;
        let decoded = bytes_text.decode().map_err(|e| {
            SourceError::Other(format!(
                "mediawiki xml: decode({}) at byte {}: {e}",
                String::from_utf8_lossy(tag),
                self.reader.buffer_position(),
            ))
        })?;
        let unescaped = unescape(&decoded).map_err(|e| {
            SourceError::Other(format!(
                "mediawiki xml: unescape({}) at byte {}: {e}",
                String::from_utf8_lossy(tag),
                self.reader.buffer_position(),
            ))
        })?;
        Ok(unescaped.into_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    /// Hand-rolled MediaWiki XML covering the cases we care about:
    /// - main-namespace article with wikitext markup → kept
    /// - Talk: page (ns=1) → dropped
    /// - redirect (`<redirect />`) → dropped
    /// - empty body → dropped
    /// - main-namespace article with multi-line / unicode text → kept
    const FIXTURE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" version="0.10">
  <siteinfo>
    <sitename>Test Wiki</sitename>
    <namespaces>
      <namespace key="0" case="first-letter" />
      <namespace key="1" case="first-letter">Talk</namespace>
    </namespaces>
  </siteinfo>
  <page>
    <title>Apollo program</title>
    <ns>0</ns>
    <id>123</id>
    <revision>
      <id>456</id>
      <text bytes="42" xml:space="preserve">The '''Apollo program''' was a [[NASA]] effort.</text>
    </revision>
  </page>
  <page>
    <title>Talk:Apollo program</title>
    <ns>1</ns>
    <id>124</id>
    <revision>
      <text>This page is the talk discussion. Should be filtered.</text>
    </revision>
  </page>
  <page>
    <title>Apollo (disambiguation)</title>
    <ns>0</ns>
    <id>125</id>
    <redirect title="Apollo" />
    <revision>
      <text>#REDIRECT [[Apollo]]</text>
    </revision>
  </page>
  <page>
    <title>Empty page</title>
    <ns>0</ns>
    <id>126</id>
    <revision>
      <text>   </text>
    </revision>
  </page>
  <page>
    <title>Über méssy unicode</title>
    <ns>0</ns>
    <id>127</id>
    <revision>
      <text xml:space="preserve">Line one.
Line two with unicode: αβγ
Line three: {{infobox}}</text>
    </revision>
  </page>
</mediawiki>
"#;

    fn write_fixture_to(path: &Path, body: &str) {
        let mut f = File::create(path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    fn write_fixture_bz2_to(path: &Path, body: &str) {
        let f = File::create(path).unwrap();
        let mut enc = bzip2::write::BzEncoder::new(f, bzip2::Compression::default());
        enc.write_all(body.as_bytes()).unwrap();
        enc.finish().unwrap();
    }

    fn collect(adapter: &MediaWikiXml) -> Vec<SourceRecord> {
        adapter
            .enumerate()
            .collect::<Result<Vec<_>, _>>()
            .expect("no errors in test fixture")
    }

    #[test]
    fn missing_file_surfaces_not_found() {
        let bogus = std::env::temp_dir().join("whisper-agent-mwxml-missing.xml");
        let _ = std::fs::remove_file(&bogus);
        let adapter = MediaWikiXml::new(&bogus);
        let mut iter = adapter.enumerate();
        let first = iter.next().expect("error item").unwrap_err();
        assert!(matches!(first, SourceError::NotFound(_)), "{first:?}");
        assert!(iter.next().is_none());
    }

    #[test]
    fn xml_filters_to_keep_only_main_articles() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("dump.xml");
        write_fixture_to(&path, FIXTURE_XML);
        let records = collect(&MediaWikiXml::new(&path));

        let titles: Vec<&str> = records.iter().map(|r| r.source_id.as_str()).collect();
        assert_eq!(records.len(), 2, "expected 2 kept pages, got {titles:?}");
        assert!(titles.contains(&"Apollo program"));
        assert!(titles.contains(&"Über méssy unicode"));
    }

    #[test]
    fn keeps_raw_wikitext_in_text() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("dump.xml");
        write_fixture_to(&path, FIXTURE_XML);
        let records = collect(&MediaWikiXml::new(&path));
        let apollo = records
            .iter()
            .find(|r| r.source_id == "Apollo program")
            .unwrap();
        // `'''bold'''` and `[[link]]` markup must survive — wikitext-
        // to-plaintext extraction is deferred, the chunker sees raw.
        assert!(apollo.text.contains("'''Apollo program'''"));
        assert!(apollo.text.contains("[[NASA]]"));
    }

    #[test]
    fn preserves_multiline_and_unicode_in_text() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("dump.xml");
        write_fixture_to(&path, FIXTURE_XML);
        let records = collect(&MediaWikiXml::new(&path));
        let unicode = records
            .iter()
            .find(|r| r.source_id == "Über méssy unicode")
            .unwrap();
        assert!(unicode.text.contains("Line one."));
        assert!(unicode.text.contains("αβγ"));
        assert!(unicode.text.contains("{{infobox}}"));
    }

    #[test]
    fn content_hash_is_deterministic_per_text() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("dump.xml");
        write_fixture_to(&path, FIXTURE_XML);
        let first = collect(&MediaWikiXml::new(&path));
        let second = collect(&MediaWikiXml::new(&path));
        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.content_hash, b.content_hash);
            assert_eq!(a.source_id, b.source_id);
        }
    }

    #[test]
    fn bz2_path_yields_same_records_as_raw() {
        let tmp = tempfile::tempdir().unwrap();
        let raw = tmp.path().join("dump.xml");
        let bz2 = tmp.path().join("dump.xml.bz2");
        write_fixture_to(&raw, FIXTURE_XML);
        write_fixture_bz2_to(&bz2, FIXTURE_XML);

        let raw_records = collect(&MediaWikiXml::new(&raw));
        let bz2_records = collect(&MediaWikiXml::new(&bz2));
        assert_eq!(raw_records.len(), bz2_records.len());
        for (a, b) in raw_records.iter().zip(bz2_records.iter()) {
            assert_eq!(a.source_id, b.source_id);
            assert_eq!(a.text, b.text);
            assert_eq!(a.content_hash, b.content_hash);
        }
    }

    #[test]
    fn bz2_extension_match_is_case_insensitive() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("dump.XML.BZ2");
        write_fixture_bz2_to(&path, FIXTURE_XML);
        let records = collect(&MediaWikiXml::new(&path));
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn malformed_xml_surfaces_an_error_then_terminates() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.xml");
        // Unclosed `<page>` — quick-xml flags this when it hits Eof
        // before seeing the matching end tag.
        write_fixture_to(
            &path,
            r#"<?xml version="1.0"?><mediawiki><page><title>Apollo</title><ns>0</ns><revision><text>body"#,
        );
        let adapter = MediaWikiXml::new(&path);
        let mut iter = adapter.enumerate();
        // Walk until we get an Err or run out — the exact event count
        // depends on quick-xml's lookahead; we only require that an
        // error eventually surfaces and the iterator then terminates.
        let mut saw_error = false;
        for _ in 0..16 {
            match iter.next() {
                Some(Err(_)) => {
                    saw_error = true;
                    break;
                }
                Some(Ok(_)) => continue,
                None => break,
            }
        }
        assert!(saw_error, "expected a parse error before exhaustion");
        assert!(iter.next().is_none(), "iterator must terminate after error");
    }

    #[test]
    fn empty_dump_yields_no_records() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty.xml");
        write_fixture_to(
            &path,
            r#"<?xml version="1.0"?><mediawiki><siteinfo><sitename>X</sitename></siteinfo></mediawiki>"#,
        );
        let records = collect(&MediaWikiXml::new(&path));
        assert!(records.is_empty());
    }
}
