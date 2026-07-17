//! Content-based file typing and text extraction.
//!
//! Type is decided by *content* (magic bytes / byte inspection), never by the
//! filename extension. Extraction strategy per type:
//!   * Text / code -> read directly (size-capped).
//!   * PDF          -> `pdf-extract`.
//!   * DOCX         -> unzip and pull `<w:t>` runs out of `word/document.xml`.
//!   * Images       -> no text, metadata only.
//!   * Other binary -> no text, metadata only.

use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// How many bytes we sniff to decide the file type.
const SNIFF_LEN: usize = 8192;
/// Never pull more than this much text out of a single file. Bounds memory and
/// keeps one pathological file from dominating the index.
const MAX_TEXT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileType {
    Text,
    Code,
    Pdf,
    Docx,
    Image,
    Binary,
}

impl FileType {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::Text => "text",
            FileType::Code => "code",
            FileType::Pdf => "pdf",
            FileType::Docx => "docx",
            FileType::Image => "image",
            FileType::Binary => "binary",
        }
    }

    /// Whether this type carries extractable text (drives preview behaviour).
    pub fn is_textual(&self) -> bool {
        matches!(self, FileType::Text | FileType::Code)
    }
}

/// Result of inspecting a file: its type and any extracted text.
pub struct Extracted {
    pub file_type: FileType,
    /// `None` for binaries/images (metadata only).
    pub text: Option<String>,
}

/// Inspect and extract text from `path`.
///
/// Returns `Ok` even when there is no text (binaries): only unrecoverable IO
/// errors surface as `Err`, so a single unreadable file can be skipped by the
/// caller without aborting the whole run.
pub fn extract(path: &Path) -> anyhow::Result<Extracted> {
    let mut file = std::fs::File::open(path)?;
    let mut head = vec![0u8; SNIFF_LEN];
    let n = read_up_to(&mut file, &mut head)?;
    head.truncate(n);

    // 1. Magic-byte detection for structured formats.
    if let Some(kind) = infer::get(&head) {
        match kind.matcher_type() {
            infer::MatcherType::Image => {
                return Ok(Extracted {
                    file_type: FileType::Image,
                    text: None,
                });
            }
            _ => {
                if kind.mime_type() == "application/pdf" {
                    return Ok(extract_pdf(path));
                }
                // DOCX is a zip; only treat it as such if it really carries a
                // word document part.
                if kind.mime_type() == "application/zip" {
                    if let Some(text) = try_extract_docx(path) {
                        return Ok(Extracted {
                            file_type: FileType::Docx,
                            text: Some(text),
                        });
                    }
                }
                // Some other known binary (archive, exe, media, ...).
                return Ok(Extracted {
                    file_type: FileType::Binary,
                    text: None,
                });
            }
        }
    }

    // 2. No magic match: decide text vs binary from the sniffed bytes.
    if content_inspector::inspect(&head).is_binary() {
        return Ok(Extracted {
            file_type: FileType::Binary,
            text: None,
        });
    }

    // 3. Plain text / source code: read the rest (capped) and keep it.
    let mut rest = Vec::new();
    let remaining = MAX_TEXT_BYTES.saturating_sub(head.len());
    file.take(remaining as u64).read_to_end(&mut rest)?;
    let mut bytes = head;
    bytes.extend_from_slice(&rest);
    let text = String::from_utf8_lossy(&bytes).into_owned();
    Ok(Extracted {
        file_type: FileType::Text,
        text: Some(text),
    })
}

fn read_up_to(file: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = file.read(&mut buf[filled..])?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    Ok(filled)
}

/// Extract PDF text. `pdf-extract` can panic on malformed files, so we isolate
/// it behind `catch_unwind`; on any failure we degrade to metadata-only.
fn extract_pdf(path: &Path) -> Extracted {
    let path = path.to_path_buf();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text(&path)
    }));
    match result {
        Ok(Ok(text)) if !text.trim().is_empty() => Extracted {
            file_type: FileType::Pdf,
            text: Some(text),
        },
        _ => Extracted {
            file_type: FileType::Pdf,
            text: None,
        },
    }
}

/// Pull the concatenated `<w:t>` text runs out of a DOCX. Returns `None` if the
/// archive is not actually a Word document.
fn try_extract_docx(path: &Path) -> Option<String> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let file = std::fs::File::open(path).ok()?;
    let mut zip = zip::ZipArchive::new(file).ok()?;
    let mut doc = zip.by_name("word/document.xml").ok()?;
    let mut xml = String::new();
    doc.read_to_string(&mut xml).ok()?;

    let mut reader = Reader::from_str(&xml);
    let mut text = String::new();
    let mut in_text = false;
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) if e.name().as_ref() == b"w:t" => in_text = true,
            Ok(Event::End(e)) if e.name().as_ref() == b"w:t" => in_text = false,
            Ok(Event::Text(e)) if in_text => {
                if let Ok(t) = e.unescape() {
                    text.push_str(&t);
                }
            }
            // Paragraph breaks -> whitespace so words don't run together.
            Ok(Event::End(e)) if e.name().as_ref() == b"w:p" => text.push('\n'),
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn detects_plain_text() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "hello world this is plain text").unwrap();
        let e = extract(f.path()).unwrap();
        assert_eq!(e.file_type, FileType::Text);
        assert!(e.text.unwrap().contains("plain text"));
    }

    #[test]
    fn detects_binary() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0u8, 159, 146, 150, 0, 1, 2, 3, 255, 254])
            .unwrap();
        let e = extract(f.path()).unwrap();
        assert_eq!(e.file_type, FileType::Binary);
        assert!(e.text.is_none());
    }

    #[test]
    fn missing_file_errors() {
        assert!(extract(Path::new("/nonexistent/deepsearch/xyz")).is_err());
    }
}
