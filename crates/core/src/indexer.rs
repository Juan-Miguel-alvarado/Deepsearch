//! Filesystem walking and (parallel, streaming) index construction.
//!
//! Walking uses the `ignore` crate so `.gitignore`, `.ignore`, hidden-file and
//! the usual `node_modules`/`.git` rules are respected for free.
//!
//! To stay within memory bounds on huge trees we do **not** materialize every
//! document at once. The file list is processed in batches: each batch is
//! tokenized in parallel with rayon, then merged into the index and dropped
//! before the next batch starts.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::extract::extract;
use crate::index::{Index, PendingDoc};
use crate::tokenize::tokenize;

/// Files handled per parallel batch. Bounds peak memory: only this many tokenized
/// documents are resident at once during a run.
const BATCH_SIZE: usize = 512;

/// Options controlling a walk.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Honour `.gitignore`/`.ignore` files.
    pub respect_gitignore: bool,
    /// Skip hidden files and directories.
    pub skip_hidden: bool,
    /// Skip files larger than this many bytes entirely (still cheap to change).
    pub max_file_size: u64,
    /// Follow symlinks.
    pub follow_links: bool,
}

impl Default for IndexOptions {
    fn default() -> Self {
        IndexOptions {
            respect_gitignore: true,
            skip_hidden: true,
            max_file_size: 64 * 1024 * 1024,
            follow_links: false,
        }
    }
}

/// Progress counters shared with a caller-driven display (e.g. a progress bar).
#[derive(Default)]
pub struct Progress {
    pub total: AtomicUsize,
    pub done: AtomicUsize,
}

impl Progress {
    pub fn snapshot(&self) -> (usize, usize) {
        (
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }
}

/// Summary of what a build/update run did.
#[derive(Debug, Default, Clone)]
pub struct IndexStats {
    pub scanned: usize,
    pub indexed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub errors: usize,
}

/// Collect the candidate file paths under `root`, applying ignore rules.
pub fn walk(root: &Path, opts: &IndexOptions) -> Vec<PathBuf> {
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .hidden(opts.skip_hidden)
        .git_ignore(opts.respect_gitignore)
        .git_global(opts.respect_gitignore)
        .git_exclude(opts.respect_gitignore)
        .follow_links(opts.follow_links)
        .max_filesize(Some(opts.max_file_size));

    let mut paths = Vec::new();
    for entry in builder.build() {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue, // unreadable dir/permission error: skip, don't abort
        };
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            paths.push(entry.into_path());
        }
    }
    paths
}

/// Full (re)build of the index for `root`. Any previously indexed state in
/// `index` is discarded.
pub fn build(
    index: &mut Index,
    root: &Path,
    opts: &IndexOptions,
    progress: &Progress,
) -> IndexStats {
    *index = Index::new();
    let paths = walk(root, opts);
    progress.total.store(paths.len(), Ordering::Relaxed);
    let mut stats = IndexStats {
        scanned: paths.len(),
        ..Default::default()
    };
    index_paths(index, &paths, progress, &mut stats);
    stats
}

/// Incremental update: reindex only changed/new files, tombstone deleted ones.
pub fn update(
    index: &mut Index,
    root: &Path,
    opts: &IndexOptions,
    progress: &Progress,
) -> IndexStats {
    let paths = walk(root, opts);
    progress.total.store(paths.len(), Ordering::Relaxed);
    let mut stats = IndexStats {
        scanned: paths.len(),
        ..Default::default()
    };

    // Detect deletions: previously indexed paths no longer present on disk.
    use std::collections::HashSet;
    let present: HashSet<&Path> = paths.iter().map(|p| p.as_path()).collect();
    let stale: Vec<PathBuf> = index
        .path_to_id
        .keys()
        .filter(|p| !present.contains(p.as_path()))
        .cloned()
        .collect();
    for p in stale {
        index.remove_path(&p);
        stats.removed += 1;
    }

    // Keep only paths that are new or whose mtime changed.
    let changed: Vec<PathBuf> = paths
        .into_iter()
        .filter(|p| {
            let m = mtime_of(p).unwrap_or(0);
            if index.is_current(p, m) {
                stats.unchanged += 1;
                progress.done.fetch_add(1, Ordering::Relaxed);
                false
            } else {
                // A changed (not new) file: tombstone the old version first.
                if index.path_to_id.contains_key(p) {
                    index.remove_path(p);
                }
                true
            }
        })
        .collect();

    index_paths(index, &changed, progress, &mut stats);
    stats
}

/// Tokenize `paths` in streaming batches and merge them into `index`.
fn index_paths(index: &mut Index, paths: &[PathBuf], progress: &Progress, stats: &mut IndexStats) {
    for batch in paths.chunks(BATCH_SIZE) {
        // Parallel, isolated tokenization. A failure on one file yields `None`
        // and is counted as an error, never aborting the batch.
        let pending: Vec<Result<Option<PendingDoc>, ()>> = batch
            .par_iter()
            .map(|path| {
                let out = tokenize_file(path);
                progress.done.fetch_add(1, Ordering::Relaxed);
                out
            })
            .collect();

        for item in pending {
            match item {
                Ok(Some(doc)) => {
                    index.add(doc);
                    stats.indexed += 1;
                }
                Ok(None) | Err(()) => stats.errors += 1,
            }
        }
    }
}

/// Tokenize a single file into a `PendingDoc`. Returns `Ok(None)` only on hard
/// failures we chose to skip; `Err` is unused today but kept for symmetry.
fn tokenize_file(path: &Path) -> Result<Option<PendingDoc>, ()> {
    // Guard the whole extraction (PDF parsing in particular can panic) so one
    // corrupt file can never bring the run down.
    let extracted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| extract(path)));
    let extracted = match extracted {
        Ok(Ok(e)) => e,
        _ => return Ok(None),
    };

    let meta = std::fs::metadata(path).map_err(|_| ())?;
    let mtime = mtime_of(path).unwrap_or(0);

    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let name_tf = count(tokenize(&name));
    // Distinct unstemmed filename tokens for fuzzy matching.
    let mut name_raw: Vec<String> = crate::tokenize::normalize(&name);
    name_raw.sort();
    name_raw.dedup();

    let content_tf = match &extracted.text {
        Some(text) => count(tokenize(text)),
        None => Default::default(),
    };

    Ok(Some(PendingDoc {
        path: path.to_path_buf(),
        size: meta.len(),
        mtime,
        file_type: extracted.file_type,
        content_tf,
        name_tf,
        name_raw,
    }))
}

fn count(terms: Vec<String>) -> std::collections::HashMap<String, u32> {
    let mut map = std::collections::HashMap::new();
    for t in terms {
        *map.entry(t).or_insert(0) += 1;
    }
    map
}

fn mtime_of(path: &Path) -> Option<i64> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    let secs = mtime
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Some(secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        p
    }

    #[test]
    fn builds_index_over_dir() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "alpha.txt", "the quick brown fox");
        write_file(dir.path(), "beta.txt", "lazy dog sleeps");
        let mut idx = Index::new();
        let stats = build(
            &mut idx,
            dir.path(),
            &IndexOptions::default(),
            &Progress::default(),
        );
        assert_eq!(stats.indexed, 2);
        assert_eq!(idx.live_docs, 2);
        let hits = crate::query::search(&idx, "fox", &crate::query::QueryOptions::default());
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn incremental_skips_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "a.txt", "hello world");
        let mut idx = Index::new();
        build(
            &mut idx,
            dir.path(),
            &IndexOptions::default(),
            &Progress::default(),
        );

        // Second run with no changes: nothing reindexed.
        let stats = update(
            &mut idx,
            dir.path(),
            &IndexOptions::default(),
            &Progress::default(),
        );
        assert_eq!(stats.indexed, 0);
        assert_eq!(stats.unchanged, 1);
    }

    #[test]
    fn incremental_detects_deletion() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_file(dir.path(), "gone.txt", "temporary content here");
        let mut idx = Index::new();
        build(
            &mut idx,
            dir.path(),
            &IndexOptions::default(),
            &Progress::default(),
        );
        assert_eq!(idx.live_docs, 1);

        std::fs::remove_file(&p).unwrap();
        let stats = update(
            &mut idx,
            dir.path(),
            &IndexOptions::default(),
            &Progress::default(),
        );
        assert_eq!(stats.removed, 1);
        assert_eq!(idx.live_docs, 0);
    }
}
