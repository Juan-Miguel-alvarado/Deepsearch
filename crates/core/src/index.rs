//! The inverted index and its on-disk representation.
//!
//! Two separate inverted indexes are kept: one over file *contents* and one
//! over file *names*. Keeping them apart lets the ranker boost name matches
//! independently (see `query.rs`).
//!
//! Incremental updates use a tombstone scheme: when a file changes or
//! disappears its old `doc_id` is marked dead instead of being surgically
//! removed from every postings list. Dead docs are skipped at query time and
//! excluded from BM25 statistics, so results stay correct; a `compact()` pass
//! can reclaim the space when churn grows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::extract::FileType;

/// A single posting: a document and the term frequency within the relevant
/// field of that document.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Posting {
    pub doc_id: u32,
    pub tf: u32,
}

/// Per-document metadata kept resident for ranking and display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    pub id: u32,
    pub path: PathBuf,
    pub size: u64,
    /// Modification time, seconds since the Unix epoch. Drives incremental
    /// reindexing.
    pub mtime: i64,
    pub file_type: FileType,
    /// Number of content tokens (document length for BM25).
    pub content_len: u32,
    /// Number of filename tokens.
    pub name_len: u32,
    /// Optional **unit-normalized** semantic embedding of the document, used for
    /// meaning-based (as opposed to keyword) ranking. `None` until the index is
    /// built with embeddings enabled.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

/// A fully-tokenized document ready to be merged into the index.
pub struct PendingDoc {
    pub path: PathBuf,
    pub size: u64,
    pub mtime: i64,
    pub file_type: FileType,
    /// term -> frequency within the content field.
    pub content_tf: HashMap<String, u32>,
    /// term -> frequency within the filename.
    pub name_tf: HashMap<String, u32>,
    /// Distinct **unstemmed** filename tokens, used for fuzzy typo matching.
    pub name_raw: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Index {
    /// Indexed by `doc_id`. `None` marks a tombstoned (dead) slot.
    pub docs: Vec<Option<DocMeta>>,
    /// term -> postings over document contents.
    pub content_index: HashMap<String, Vec<Posting>>,
    /// term -> postings over filenames.
    pub name_index: HashMap<String, Vec<Posting>>,
    /// Unstemmed filename token -> docs containing it. Fuzzy matching scans
    /// these keys so typos are compared against real words, not stems.
    pub name_fuzzy: HashMap<String, Vec<u32>>,
    /// path -> (doc_id, mtime) for the *live* mapping, used by incremental runs.
    pub path_to_id: HashMap<PathBuf, (u32, i64)>,

    // Aggregates over live docs only, maintained incrementally so query-time
    // BM25 needs no full scan.
    pub live_docs: u64,
    pub total_content_len: u64,
    pub total_name_len: u64,
}

impl Index {
    pub fn new() -> Self {
        Index::default()
    }

    /// Average content length across live docs (BM25 `avgdl`). Guards against
    /// division by zero on an empty index.
    pub fn avg_content_len(&self) -> f32 {
        if self.live_docs == 0 {
            0.0
        } else {
            self.total_content_len as f32 / self.live_docs as f32
        }
    }

    pub fn avg_name_len(&self) -> f32 {
        if self.live_docs == 0 {
            0.0
        } else {
            self.total_name_len as f32 / self.live_docs as f32
        }
    }

    pub fn doc(&self, id: u32) -> Option<&DocMeta> {
        self.docs.get(id as usize).and_then(|d| d.as_ref())
    }

    pub fn is_live(&self, id: u32) -> bool {
        self.doc(id).is_some()
    }

    /// Attach a semantic embedding to a live document (no-op for dead slots).
    /// The vector should already be unit-normalized so ranking can use a plain
    /// dot product.
    pub fn set_embedding(&mut self, id: u32, embedding: Vec<f32>) {
        if let Some(Some(meta)) = self.docs.get_mut(id as usize) {
            meta.embedding = Some(embedding);
        }
    }

    /// Whether any live document carries a semantic embedding.
    pub fn has_embeddings(&self) -> bool {
        self.docs.iter().flatten().any(|m| m.embedding.is_some())
    }

    /// Add a freshly tokenized document, returning its new `doc_id`.
    pub fn add(&mut self, pending: PendingDoc) -> u32 {
        let id = self.docs.len() as u32;
        let content_len: u32 = pending.content_tf.values().sum();
        let name_len: u32 = pending.name_tf.values().sum();

        for (term, tf) in pending.content_tf {
            self.content_index
                .entry(term)
                .or_default()
                .push(Posting { doc_id: id, tf });
        }
        for (term, tf) in pending.name_tf {
            self.name_index
                .entry(term)
                .or_default()
                .push(Posting { doc_id: id, tf });
        }
        for raw in pending.name_raw {
            self.name_fuzzy.entry(raw).or_default().push(id);
        }

        self.path_to_id
            .insert(pending.path.clone(), (id, pending.mtime));
        self.docs.push(Some(DocMeta {
            id,
            path: pending.path,
            size: pending.size,
            mtime: pending.mtime,
            file_type: pending.file_type,
            content_len,
            name_len,
            embedding: None,
        }));

        self.live_docs += 1;
        self.total_content_len += content_len as u64;
        self.total_name_len += name_len as u64;
        id
    }

    /// Tombstone a document by path. Its postings remain but are skipped at
    /// query time; aggregates are corrected immediately.
    pub fn remove_path(&mut self, path: &Path) {
        if let Some((id, _)) = self.path_to_id.remove(path) {
            if let Some(slot) = self.docs.get_mut(id as usize) {
                if let Some(meta) = slot.take() {
                    self.live_docs -= 1;
                    self.total_content_len -= meta.content_len as u64;
                    self.total_name_len -= meta.name_len as u64;
                }
            }
        }
    }

    /// Whether `path` is already indexed with the given mtime (i.e. unchanged).
    pub fn is_current(&self, path: &Path, mtime: i64) -> bool {
        matches!(self.path_to_id.get(path), Some(&(_, m)) if m == mtime)
    }

    /// Fraction of doc slots that are tombstoned. High values mean `compact()`
    /// would pay off.
    pub fn dead_ratio(&self) -> f32 {
        if self.docs.is_empty() {
            return 0.0;
        }
        let dead = self.docs.len() as u64 - self.live_docs;
        dead as f32 / self.docs.len() as f32
    }

    /// Rebuild the index dropping all tombstoned docs and renumbering ids.
    pub fn compact(&mut self) {
        let mut fresh = Index::new();
        // Reinsert live docs in id order; postings are rebuilt from scratch is
        // expensive, so instead remap ids and rewrite postings lists.
        let mut remap = HashMap::new();
        for meta in self.docs.iter().flatten() {
            {
                let new_id = fresh.docs.len() as u32;
                remap.insert(meta.id, new_id);
                fresh
                    .path_to_id
                    .insert(meta.path.clone(), (new_id, meta.mtime));
                let mut m = meta.clone();
                m.id = new_id;
                fresh.total_content_len += m.content_len as u64;
                fresh.total_name_len += m.name_len as u64;
                fresh.live_docs += 1;
                fresh.docs.push(Some(m));
            }
        }
        fresh.content_index = remap_postings(&self.content_index, &remap);
        fresh.name_index = remap_postings(&self.name_index, &remap);
        for (term, ids) in &self.name_fuzzy {
            let kept: Vec<u32> = ids.iter().filter_map(|id| remap.get(id).copied()).collect();
            if !kept.is_empty() {
                fresh.name_fuzzy.insert(term.clone(), kept);
            }
        }
        *self = fresh;
    }
}

fn remap_postings(
    index: &HashMap<String, Vec<Posting>>,
    remap: &HashMap<u32, u32>,
) -> HashMap<String, Vec<Posting>> {
    let mut out = HashMap::with_capacity(index.len());
    for (term, postings) in index {
        let mut kept: Vec<Posting> = postings
            .iter()
            .filter_map(|p| {
                remap.get(&p.doc_id).map(|&new_id| Posting {
                    doc_id: new_id,
                    tf: p.tf,
                })
            })
            .collect();
        if !kept.is_empty() {
            kept.shrink_to_fit();
            out.insert(term.clone(), kept);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending(path: &str, content: &[(&str, u32)], name: &[(&str, u32)]) -> PendingDoc {
        PendingDoc {
            path: PathBuf::from(path),
            size: 10,
            mtime: 1,
            file_type: FileType::Text,
            content_tf: content.iter().map(|(t, f)| (t.to_string(), *f)).collect(),
            name_tf: name.iter().map(|(t, f)| (t.to_string(), *f)).collect(),
            name_raw: name.iter().map(|(t, _)| t.to_string()).collect(),
        }
    }

    #[test]
    fn add_updates_aggregates() {
        let mut idx = Index::new();
        idx.add(pending("/a", &[("foo", 2), ("bar", 1)], &[("a", 1)]));
        assert_eq!(idx.live_docs, 1);
        assert_eq!(idx.total_content_len, 3);
        assert_eq!(idx.avg_content_len(), 3.0);
        assert_eq!(idx.content_index["foo"][0].tf, 2);
    }

    #[test]
    fn remove_tombstones_and_fixes_stats() {
        let mut idx = Index::new();
        idx.add(pending("/a", &[("foo", 2)], &[("a", 1)]));
        idx.add(pending("/b", &[("foo", 1)], &[("b", 1)]));
        idx.remove_path(Path::new("/a"));
        assert_eq!(idx.live_docs, 1);
        assert_eq!(idx.total_content_len, 1);
        assert!(!idx.is_live(0));
        assert!(idx.is_live(1));
        // postings still physically present for the dead doc
        assert_eq!(idx.content_index["foo"].len(), 2);
    }

    #[test]
    fn is_current_tracks_mtime() {
        let mut idx = Index::new();
        let mut p = pending("/a", &[("foo", 1)], &[("a", 1)]);
        p.mtime = 42;
        idx.add(p);
        assert!(idx.is_current(Path::new("/a"), 42));
        assert!(!idx.is_current(Path::new("/a"), 43));
        assert!(!idx.is_current(Path::new("/b"), 42));
    }

    #[test]
    fn compact_reclaims_dead_docs() {
        let mut idx = Index::new();
        idx.add(pending("/a", &[("foo", 2)], &[("a", 1)]));
        idx.add(pending("/b", &[("foo", 1), ("bar", 3)], &[("b", 1)]));
        idx.remove_path(Path::new("/a"));
        idx.compact();
        assert_eq!(idx.docs.len(), 1);
        assert_eq!(idx.live_docs, 1);
        assert_eq!(idx.content_index["foo"].len(), 1);
        assert_eq!(idx.content_index["foo"][0].doc_id, 0);
        assert!(idx.is_live(0));
    }
}
