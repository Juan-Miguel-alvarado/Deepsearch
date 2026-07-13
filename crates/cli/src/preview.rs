//! Off-thread preview rendering.
//!
//! The TUI must never block on IO or syntax highlighting, so previews are built
//! on a dedicated worker thread. The UI sends a [`PreviewRequest`] tagged with a
//! monotonically increasing generation; the worker coalesces to the newest
//! pending request and replies with the built [`Preview`]. The UI applies a
//! reply only if its generation still matches the current selection, discarding
//! stale work.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};

use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SynStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;

use deepsearch_core::FileType;

use crate::util::{format_timestamp, human_size};

const MAX_PREVIEW_LINES: usize = 500;
const MAX_PREVIEW_BYTES: usize = 512 * 1024;

/// A built preview ready to render.
pub enum Preview {
    /// Highlighted text/code or extracted document text.
    Text(Text<'static>),
    /// Key/value metadata (binaries, or the header shown above text).
    Meta(Text<'static>),
    /// A decoded, downscaled image. Decoding happens on the worker thread; the
    /// UI thread only wraps it in a terminal-protocol widget.
    Image(Box<image::DynamicImage>),
    /// Nothing selected / still loading.
    Loading,
    /// Something went wrong building the preview.
    Error(String),
}

/// A request to build a preview for one result.
pub struct PreviewRequest {
    pub generation: u64,
    pub path: PathBuf,
    pub file_type: FileType,
    pub size: u64,
    pub mtime: i64,
    /// Raw (unstemmed) query words, lowercased, for match highlighting.
    pub terms: Vec<String>,
}

/// Handle to the background preview worker.
pub struct PreviewWorker {
    tx: Sender<PreviewRequest>,
    pub rx: Receiver<(u64, Preview)>,
}

impl PreviewWorker {
    pub fn spawn() -> Self {
        let (req_tx, req_rx) = std::sync::mpsc::channel::<PreviewRequest>();
        let (resp_tx, resp_rx) = std::sync::mpsc::channel::<(u64, Preview)>();

        std::thread::spawn(move || {
            // Loading these defaults is ~tens of ms; do it once per worker.
            let syntaxes = SyntaxSet::load_defaults_newlines();
            let themes = ThemeSet::load_defaults();
            let theme = themes
                .themes
                .get("base16-ocean.dark")
                .cloned()
                .or_else(|| themes.themes.values().next().cloned())
                .expect("syntect ships default themes");

            while let Ok(mut req) = req_rx.recv() {
                // Coalesce: if the user moved the selection several times while we
                // were busy, skip straight to the latest request.
                while let Ok(newer) = req_rx.try_recv() {
                    req = newer;
                }
                let preview = build_preview(&syntaxes, &theme, &req);
                if resp_tx.send((req.generation, preview)).is_err() {
                    break; // UI gone
                }
            }
        });

        PreviewWorker { tx: req_tx, rx: resp_rx }
    }

    pub fn request(&self, req: PreviewRequest) {
        let _ = self.tx.send(req);
    }
}

fn build_preview(syntaxes: &SyntaxSet, theme: &Theme, req: &PreviewRequest) -> Preview {
    match req.file_type {
        FileType::Image => image_preview(req),
        FileType::Binary => meta_preview(req),
        FileType::Text | FileType::Code => text_preview(syntaxes, theme, req),
        FileType::Pdf | FileType::Docx => document_preview(req),
    }
}

/// Largest image dimension we keep; larger images are downscaled so building
/// the terminal protocol on the UI thread stays cheap.
const MAX_IMAGE_DIM: u32 = 1200;

/// Decode (and downscale) an image on the worker thread.
fn image_preview(req: &PreviewRequest) -> Preview {
    let reader = match image::ImageReader::open(&req.path).and_then(|r| r.with_guessed_format()) {
        Ok(r) => r,
        Err(e) => return Preview::Error(format!("cannot open image: {e}")),
    };
    match reader.decode() {
        Ok(img) => {
            let img = if img.width() > MAX_IMAGE_DIM || img.height() > MAX_IMAGE_DIM {
                img.thumbnail(MAX_IMAGE_DIM, MAX_IMAGE_DIM)
            } else {
                img
            };
            Preview::Image(Box::new(img))
        }
        // Corrupt/unsupported image: fall back to metadata, never crash.
        Err(_) => meta_preview(req),
    }
}

/// Metadata-only preview for binaries.
fn meta_preview(req: &PreviewRequest) -> Preview {
    let mut lines = Vec::new();
    let key = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let val = Style::default().fg(Color::Gray);
    let kv = |k: &str, v: String| {
        Line::from(vec![
            Span::styled(format!("{k:<10}"), key),
            Span::styled(v, val),
        ])
    };
    lines.push(kv("Type", req.file_type.as_str().to_string()));
    lines.push(kv("Size", human_size(req.size)));
    lines.push(kv("Modified", format_timestamp(req.mtime)));
    lines.push(kv("Path", req.path.display().to_string()));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "(binary file — no text preview)",
        Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
    )));
    Preview::Meta(Text::from(lines))
}

/// Syntax-highlighted preview for text/code.
fn text_preview(syntaxes: &SyntaxSet, theme: &Theme, req: &PreviewRequest) -> Preview {
    let bytes = match read_capped(&req.path) {
        Ok(b) => b,
        Err(e) => return Preview::Error(format!("cannot read file: {e}")),
    };
    let content = String::from_utf8_lossy(&bytes);

    let syntax = req
        .path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(|ext| syntaxes.find_syntax_by_extension(ext))
        .or_else(|| {
            content
                .lines()
                .next()
                .and_then(|l| syntaxes.find_syntax_by_first_line(l))
        })
        .unwrap_or_else(|| syntaxes.find_syntax_plain_text());

    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut lines: Vec<Line<'static>> = Vec::new();

    for raw_line in content.lines().take(MAX_PREVIEW_LINES) {
        let ranges: Vec<(SynStyle, &str)> = highlighter
            .highlight_line(raw_line, syntaxes)
            .unwrap_or_default();
        let spans: Vec<(Style, String)> = ranges
            .into_iter()
            .map(|(st, text)| (syn_to_ratatui(st), text.to_string()))
            .collect();
        lines.push(overlay_matches(spans, &req.terms));
    }

    if content.lines().count() > MAX_PREVIEW_LINES {
        lines.push(Line::from(Span::styled(
            format!("… (truncated at {MAX_PREVIEW_LINES} lines)"),
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )));
    }

    Preview::Text(Text::from(lines))
}

/// Preview for PDF/DOCX: re-extract text via the core extractor and show it
/// plain (no syntax coloring) with query matches highlighted.
fn document_preview(req: &PreviewRequest) -> Preview {
    match deepsearch_core::extract::extract(&req.path) {
        Ok(ext) => match ext.text {
            Some(text) if !text.trim().is_empty() => {
                let lines: Vec<Line<'static>> = text
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .take(MAX_PREVIEW_LINES)
                    .map(|l| {
                        let base = Style::default().fg(Color::Gray);
                        overlay_matches(vec![(base, l.to_string())], &req.terms)
                    })
                    .collect();
                Preview::Text(Text::from(lines))
            }
            _ => meta_preview(req),
        },
        Err(e) => Preview::Error(format!("cannot extract document: {e}")),
    }
}

fn read_capped(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.by_ref().take(MAX_PREVIEW_BYTES as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

fn syn_to_ratatui(s: SynStyle) -> Style {
    Style::default().fg(Color::Rgb(
        s.foreground.r,
        s.foreground.g,
        s.foreground.b,
    ))
}

/// Overlay query-match highlighting on a line's styled spans.
///
/// Matches are found case-insensitively as substrings of the visible text and
/// rendered reversed+bold. Works by expanding to per-character styles, marking
/// matched positions, then run-length encoding back into spans.
fn overlay_matches(spans: Vec<(Style, String)>, terms: &[String]) -> Line<'static> {
    if terms.is_empty() {
        return Line::from(
            spans
                .into_iter()
                .map(|(st, t)| Span::styled(t, st))
                .collect::<Vec<_>>(),
        );
    }

    // Expand spans to parallel per-char arrays.
    let mut chars: Vec<char> = Vec::new();
    let mut styles: Vec<Style> = Vec::new();
    for (st, t) in &spans {
        for c in t.chars() {
            chars.push(c);
            styles.push(*st);
        }
    }
    if chars.is_empty() {
        return Line::from("");
    }

    // Lowercased view for case-insensitive matching (approx: first lowercase char).
    let lower: Vec<char> = chars
        .iter()
        .map(|c| c.to_lowercase().next().unwrap_or(*c))
        .collect();

    let mut matched = vec![false; chars.len()];
    for term in terms {
        let needle: Vec<char> = term.chars().collect();
        if needle.is_empty() || needle.len() > lower.len() {
            continue;
        }
        let mut i = 0;
        while i + needle.len() <= lower.len() {
            if lower[i..i + needle.len()] == needle[..] {
                matched[i..i + needle.len()].fill(true);
                i += needle.len();
            } else {
                i += 1;
            }
        }
    }

    let hl = Modifier::REVERSED | Modifier::BOLD;
    // Run-length encode back into spans of equal (style, matched) pairs.
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut cur = String::new();
    let mut cur_style = styles[0];
    let mut cur_matched = matched[0];
    let flush = |buf: &mut String, style: Style, is_match: bool, out: &mut Vec<Span<'static>>| {
        if buf.is_empty() {
            return;
        }
        let style = if is_match { style.add_modifier(hl) } else { style };
        out.push(Span::styled(std::mem::take(buf), style));
    };
    for i in 0..chars.len() {
        if styles[i] != cur_style || matched[i] != cur_matched {
            flush(&mut cur, cur_style, cur_matched, &mut out);
            cur_style = styles[i];
            cur_matched = matched[i];
        }
        cur.push(chars[i]);
    }
    flush(&mut cur, cur_style, cur_matched, &mut out);
    Line::from(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_marks_matches() {
        let base = Style::default();
        let line = overlay_matches(vec![(base, "hello world".to_string())], &["world".to_string()]);
        // Expect two spans: "hello " (plain) and "world" (highlighted).
        assert!(line.spans.len() >= 2);
        let highlighted: String = line
            .spans
            .iter()
            .filter(|s| s.style.add_modifier.contains(Modifier::REVERSED))
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(highlighted, "world");
    }

    #[test]
    fn overlay_noop_without_terms() {
        let base = Style::default();
        let line = overlay_matches(vec![(base, "abc".to_string())], &[]);
        assert_eq!(line.spans.len(), 1);
    }
}
