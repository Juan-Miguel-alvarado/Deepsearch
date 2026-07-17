//! # deepsearch-core
//!
//! The indexing + search engine behind `deepsearch`, usable entirely without a
//! TUI. Two layers live here:
//!
//! * **Indexer** ([`indexer`]): walks the filesystem, extracts text, and builds
//!   a persistent inverted index.
//! * **Query** ([`query`]): tokenizes a query and ranks documents with a
//!   hand-written BM25.
//!
//! The [`DeepSearch`] type ties them together with load/save persistence.
//!
//! ```no_run
//! use deepsearch_core::{DeepSearch, indexer::{IndexOptions, Progress}};
//! use deepsearch_core::query::QueryOptions;
//! use std::path::Path;
//!
//! let mut ds = DeepSearch::open_or_empty(None)?;
//! ds.build(Path::new("/home/juan"), &IndexOptions::default(), &Progress::default());
//! ds.save(None)?;
//! for hit in ds.search("bm25 ranking", &QueryOptions::default()) {
//!     println!("{:.3}  {}", hit.score, hit.path.display());
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod extract;
pub mod index;
pub mod indexer;
pub mod query;
pub mod tokenize;

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

pub use extract::FileType;
pub use index::{DocMeta, Index};
pub use indexer::{IndexOptions, IndexStats, Progress};
pub use query::QueryOptions;

/// Serialized index format version. Bumped when the on-disk layout changes so a
/// stale cache is rejected instead of misread.
const FORMAT_VERSION: u32 = 2;

/// Errors surfaced by the high-level API.
#[derive(Debug, thiserror::Error)]
pub enum DeepSearchError {
    #[error("index cache not found at {0}")]
    NotFound(PathBuf),
    #[error("index cache is version {found}, expected {expected}; run `deepsearch index` again")]
    VersionMismatch { found: u32, expected: u32 },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("failed to (de)serialize index: {0}")]
    Codec(String),
}

/// A fully resolved search result (metadata joined onto the ranked hit).
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: u32,
    pub path: PathBuf,
    pub score: f32,
    pub file_type: FileType,
    pub size: u64,
    pub mtime: i64,
}

/// On-disk envelope: a version tag plus the index.
#[derive(Serialize, Deserialize)]
struct IndexFile {
    version: u32,
    index: Index,
}

/// High-level handle bundling an [`Index`] with persistence and search.
pub struct DeepSearch {
    index: Index,
}

impl DeepSearch {
    /// Wrap an in-memory index.
    pub fn new(index: Index) -> Self {
        DeepSearch { index }
    }

    /// Start from an empty index.
    pub fn empty() -> Self {
        DeepSearch {
            index: Index::new(),
        }
    }

    /// Default on-disk cache path: `~/.cache/deepsearch/index.bin`.
    pub fn default_cache_path() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("deepsearch")
            .join("index.bin")
    }

    /// Load an existing index from `path` (or the default cache path), falling
    /// back to an empty index when none exists **or when the cache is from an
    /// older format** — the caller then rebuilds, so a stale cache is a silent
    /// no-op rather than an error.
    pub fn open_or_empty(path: Option<&Path>) -> Result<Self, DeepSearchError> {
        match Self::load(path) {
            Ok(ds) => Ok(ds),
            Err(DeepSearchError::NotFound(_)) | Err(DeepSearchError::VersionMismatch { .. }) => {
                Ok(Self::empty())
            }
            Err(e) => Err(e),
        }
    }

    /// Load an index from disk.
    pub fn load(path: Option<&Path>) -> Result<Self, DeepSearchError> {
        let path = path
            .map(Path::to_path_buf)
            .unwrap_or_else(Self::default_cache_path);
        if !path.exists() {
            return Err(DeepSearchError::NotFound(path));
        }
        let bytes = std::fs::read(&path)?;
        let config = bincode::config::standard();
        // Read just the version prefix first. The version is the first field of
        // the envelope, so a stale cache produces a clear VersionMismatch instead
        // of a confusing decode error when the Index layout has since changed.
        let (version, _) = bincode::serde::decode_from_slice::<u32, _>(&bytes, config)
            .map_err(|e| DeepSearchError::Codec(e.to_string()))?;
        if version != FORMAT_VERSION {
            return Err(DeepSearchError::VersionMismatch {
                found: version,
                expected: FORMAT_VERSION,
            });
        }
        let (file, _): (IndexFile, usize) = bincode::serde::decode_from_slice(&bytes, config)
            .map_err(|e| DeepSearchError::Codec(e.to_string()))?;
        Ok(DeepSearch { index: file.index })
    }

    /// Persist the index to disk (atomic: write to a temp file then rename).
    pub fn save(&self, path: Option<&Path>) -> anyhow::Result<()> {
        let path = path
            .map(Path::to_path_buf)
            .unwrap_or_else(Self::default_cache_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating cache dir {}", parent.display()))?;
        }
        // Serialize by reference so we never clone the (potentially large) index.
        let envelope = IndexFileRef {
            version: FORMAT_VERSION,
            index: &self.index,
        };
        let bytes = bincode::serde::encode_to_vec(&envelope, bincode::config::standard())
            .context("serializing index")?;
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("renaming into {}", path.display()))?;
        Ok(())
    }

    /// Full (re)build over `root`.
    pub fn build(&mut self, root: &Path, opts: &IndexOptions, progress: &Progress) -> IndexStats {
        indexer::build(&mut self.index, root, opts, progress)
    }

    /// Incremental update over `root`.
    pub fn update(&mut self, root: &Path, opts: &IndexOptions, progress: &Progress) -> IndexStats {
        indexer::update(&mut self.index, root, opts, progress)
    }

    /// Number of live (non-tombstoned) documents.
    pub fn len(&self) -> u64 {
        self.index.live_docs
    }

    pub fn is_empty(&self) -> bool {
        self.index.live_docs == 0
    }

    /// Compact if enough of the index is tombstoned.
    pub fn maybe_compact(&mut self, threshold: f32) -> bool {
        if self.index.dead_ratio() >= threshold {
            self.index.compact();
            true
        } else {
            false
        }
    }

    /// Borrow the underlying index (for benchmarks / advanced use).
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Run a ranked keyword search, resolving metadata onto each hit.
    pub fn search(&self, query: &str, opts: &QueryOptions) -> Vec<SearchResult> {
        self.resolve(query::search(&self.index, query, opts))
    }

    /// Hybrid keyword + semantic search. `query_vec` is the (unit-normalized)
    /// embedding of the query; `semantic_weight` (0..1) blends the two signals.
    /// Falls back to pure keyword when `query_vec` is empty.
    pub fn hybrid_search(
        &self,
        query: &str,
        query_vec: &[f32],
        opts: &QueryOptions,
        semantic_weight: f32,
    ) -> Vec<SearchResult> {
        self.resolve(query::hybrid_search(
            &self.index,
            query,
            query_vec,
            opts,
            semantic_weight,
        ))
    }

    /// Whether the index carries semantic embeddings (built with `--semantic`).
    pub fn has_embeddings(&self) -> bool {
        self.index.has_embeddings()
    }

    /// Live `(doc_id, path)` pairs that don't yet have an embedding — the work
    /// list for building semantic vectors.
    pub fn docs_needing_embedding(&self) -> Vec<(u32, PathBuf)> {
        self.index
            .docs
            .iter()
            .flatten()
            .filter(|m| m.embedding.is_none())
            .map(|m| (m.id, m.path.clone()))
            .collect()
    }

    /// Attach a (unit-normalized) embedding to a document.
    pub fn set_embedding(&mut self, doc_id: u32, embedding: Vec<f32>) {
        self.index.set_embedding(doc_id, embedding);
    }

    /// Resolve ranked hits into full results by joining document metadata.
    fn resolve(&self, hits: Vec<query::Hit>) -> Vec<SearchResult> {
        hits.into_iter()
            .filter_map(|hit| {
                self.index.doc(hit.doc_id).map(|m| SearchResult {
                    doc_id: hit.doc_id,
                    path: m.path.clone(),
                    score: hit.score,
                    file_type: m.file_type,
                    size: m.size,
                    mtime: m.mtime,
                })
            })
            .collect()
    }
}

/// Borrowing counterpart of [`IndexFile`] used for zero-copy serialization.
#[derive(Serialize)]
struct IndexFileRef<'a> {
    version: u32,
    index: &'a Index,
}
