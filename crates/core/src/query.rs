//! Query evaluation: BM25 ranking over the content and name fields, plus a
//! fuzzy fallback on filenames to tolerate typos.
//!
//! BM25 is implemented by hand (no search-engine crate). For a term `t` in a
//! document `d` of a field with average length `avgdl`:
//!
//! ```text
//!   idf(t)      = ln( 1 + (N - df + 0.5) / (df + 0.5) )
//!   score(t,d)  = idf(t) * ( tf * (k1 + 1) )
//!                          / ( tf + k1 * (1 - b + b * |d| / avgdl) )
//! ```
//!
//! `df` is computed from *live* postings only, so tombstoned docs never skew
//! the statistics. Name-field scores are multiplied by `name_boost` before
//! being added to the content score.

use std::collections::HashMap;

use crate::index::Index;
use crate::tokenize::{normalize, stem_word};

pub const DEFAULT_K1: f32 = 1.2;
pub const DEFAULT_B: f32 = 0.75;

/// Tunable ranking parameters.
#[derive(Debug, Clone)]
pub struct QueryOptions {
    pub k1: f32,
    pub b: f32,
    /// Multiplier applied to filename-field scores (matches in the name count
    /// more than matches in the body).
    pub name_boost: f32,
    /// Enable Levenshtein fuzzy matching against filename terms.
    pub fuzzy: bool,
    /// Max edit distance for a fuzzy filename match.
    pub fuzzy_max_dist: usize,
    /// Multiplier applied to fuzzy (approximate) filename matches.
    pub fuzzy_penalty: f32,
    /// Cap on the number of results returned.
    pub limit: usize,
}

impl Default for QueryOptions {
    fn default() -> Self {
        QueryOptions {
            k1: DEFAULT_K1,
            b: DEFAULT_B,
            name_boost: 3.0,
            fuzzy: true,
            fuzzy_max_dist: 1,
            fuzzy_penalty: 0.4,
            limit: 100,
        }
    }
}

/// One ranked hit.
#[derive(Debug, Clone)]
pub struct Hit {
    pub doc_id: u32,
    pub score: f32,
}

fn idf(n: u64, df: u64) -> f32 {
    let n = n as f32;
    let df = df as f32;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

fn bm25_tf(tf: u32, doc_len: u32, avgdl: f32, k1: f32, b: f32) -> f32 {
    let tf = tf as f32;
    let len_norm = if avgdl > 0.0 {
        1.0 - b + b * (doc_len as f32 / avgdl)
    } else {
        1.0
    };
    (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
}

/// Rank documents in `index` for `query`, returning hits sorted by descending
/// score (ties broken by ascending doc_id for determinism).
pub fn search(index: &Index, query: &str, opts: &QueryOptions) -> Vec<Hit> {
    // Keep the unstemmed tokens so the fuzzy pass can compare typos against real
    // filename words; derive the stemmed term per token for exact matching.
    let raw_terms = normalize(query);
    if raw_terms.is_empty() || index.live_docs == 0 {
        return Vec::new();
    }
    let n = index.live_docs;
    let avg_content = index.avg_content_len();
    let avg_name = index.avg_name_len();

    let mut scores: HashMap<u32, f32> = HashMap::new();

    for raw in &raw_terms {
        let term = stem_word(raw);
        // --- content field ---
        if let Some(postings) = index.content_index.get(&term) {
            let live: Vec<_> = postings.iter().filter(|p| index.is_live(p.doc_id)).collect();
            let df = live.len() as u64;
            if df > 0 {
                let term_idf = idf(n, df);
                for p in live {
                    let dl = index.doc(p.doc_id).map(|d| d.content_len).unwrap_or(0);
                    let s = term_idf * bm25_tf(p.tf, dl, avg_content, opts.k1, opts.b);
                    *scores.entry(p.doc_id).or_insert(0.0) += s;
                }
            }
        }

        // --- name field (exact) ---
        let mut name_hit = false;
        if let Some(postings) = index.name_index.get(&term) {
            let live: Vec<_> = postings.iter().filter(|p| index.is_live(p.doc_id)).collect();
            let df = live.len() as u64;
            if df > 0 {
                name_hit = true;
                let term_idf = idf(n, df);
                for p in live {
                    let dl = index.doc(p.doc_id).map(|d| d.name_len).unwrap_or(0);
                    let s = opts.name_boost
                        * term_idf
                        * bm25_tf(p.tf, dl, avg_name, opts.k1, opts.b);
                    *scores.entry(p.doc_id).or_insert(0.0) += s;
                }
            }
        }

        // --- name field (fuzzy fallback) ---
        // Only spend the linear scan when there was no exact name match, so the
        // common case stays fast.
        if opts.fuzzy && !name_hit {
            fuzzy_name_match(index, raw, opts, avg_name, n, &mut scores);
        }
    }

    let mut hits: Vec<Hit> = scores
        .into_iter()
        .map(|(doc_id, score)| Hit { doc_id, score })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.doc_id.cmp(&b.doc_id))
    });
    hits.truncate(opts.limit);
    hits
}

/// Scan the **unstemmed** filename dictionary for tokens within
/// `fuzzy_max_dist` edits of the (unstemmed) query token `raw` and score their
/// docs with a penalty.
///
/// `raw` is compared against real filename words (not stems) because stemming a
/// misspelled word produces unpredictable results and inflates edit distances.
/// Two cheap prefilters keep the scan bounded even on huge dictionaries:
///   * candidate length within the edit budget, and
///   * matching first character — typos rarely change the first letter, and
///     this prunes an entire numeric filename dictionary against a word query.
fn fuzzy_name_match(
    index: &Index,
    raw: &str,
    opts: &QueryOptions,
    avg_name: f32,
    n: u64,
    scores: &mut HashMap<u32, f32>,
) {
    let tlen = raw.chars().count();
    // Fuzzy is a typo fallback: only worth it for reasonably long tokens.
    if tlen < 4 {
        return;
    }
    let tfirst = raw.chars().next();
    for (cand, doc_ids) in &index.name_fuzzy {
        let clen = cand.chars().count();
        if clen.abs_diff(tlen) > opts.fuzzy_max_dist {
            continue;
        }
        if cand.chars().next() != tfirst {
            continue;
        }
        // Damerau-Levenshtein so a transposed pair of letters (the single most
        // common typo, e.g. "gorceries" -> "groceries") counts as one edit.
        let dist = strsim::damerau_levenshtein(raw, cand);
        if dist == 0 || dist > opts.fuzzy_max_dist {
            continue;
        }
        let live: Vec<u32> = doc_ids.iter().copied().filter(|&id| index.is_live(id)).collect();
        let df = live.len() as u64;
        if df == 0 {
            continue;
        }
        // Closer matches keep more of the score.
        let closeness = 1.0 - (dist as f32 / (opts.fuzzy_max_dist as f32 + 1.0));
        let factor = opts.name_boost * opts.fuzzy_penalty * closeness;
        let term_idf = idf(n, df);
        for id in live {
            let dl = index.doc(id).map(|d| d.name_len).unwrap_or(0);
            // Fuzzy hits carry an implicit term frequency of 1.
            let s = factor * term_idf * bm25_tf(1, dl, avg_name, opts.k1, opts.b);
            *scores.entry(id).or_insert(0.0) += s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::FileType;
    use crate::index::PendingDoc;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn tf(pairs: &[(&str, u32)]) -> HashMap<String, u32> {
        pairs.iter().map(|(t, f)| (t.to_string(), *f)).collect()
    }

    fn doc(idx: &mut Index, path: &str, content: &[(&str, u32)], name: &[(&str, u32)]) {
        idx.add(PendingDoc {
            path: PathBuf::from(path),
            size: 1,
            mtime: 1,
            file_type: FileType::Text,
            content_tf: tf(content),
            name_tf: tf(name),
            name_raw: name.iter().map(|(t, _)| t.to_string()).collect(),
        });
    }

    #[test]
    fn idf_rewards_rare_terms() {
        // rarer term (smaller df) must have higher idf
        assert!(idf(1000, 1) > idf(1000, 500));
    }

    #[test]
    fn ranks_by_relevance() {
        let mut idx = Index::new();
        // doc0 mentions "rust" twice, doc1 once, doc2 not at all.
        doc(&mut idx, "/a", &[("rust", 2), ("code", 1)], &[("a", 1)]);
        doc(&mut idx, "/b", &[("rust", 1), ("code", 5)], &[("b", 1)]);
        doc(&mut idx, "/c", &[("code", 3)], &[("c", 1)]);
        let hits = search(&idx, "rust", &QueryOptions::default());
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].doc_id, 0); // more occurrences, shorter doc
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn name_match_outranks_content_match() {
        let mut idx = Index::new();
        // doc0: "report" only in the body; doc1: "report" in the filename.
        doc(&mut idx, "/a", &[("report", 1), ("filler", 20)], &[("a", 1)]);
        doc(&mut idx, "/report", &[("filler", 1)], &[("report", 1)]);
        let hits = search(&idx, "report", &QueryOptions::default());
        assert_eq!(hits[0].doc_id, 1);
    }

    #[test]
    fn fuzzy_tolerates_typo_in_filename() {
        let mut idx = Index::new();
        doc(&mut idx, "/config", &[("stuff", 1)], &[("config", 1)]);
        // "config" is unstemmed here; a transposition typo should still match
        // at the default distance of 1 thanks to Damerau-Levenshtein.
        let hits = search(&idx, "cofnig", &QueryOptions::default());
        assert_eq!(hits[0].doc_id, 0);
    }

    #[test]
    fn fuzzy_ignored_when_disabled() {
        let mut idx = Index::new();
        doc(&mut idx, "/config", &[("stuff", 1)], &[("config", 1)]);
        let opts = QueryOptions { fuzzy: false, ..QueryOptions::default() };
        assert!(search(&idx, "cofnig", &opts).is_empty());
    }

    #[test]
    fn tombstoned_docs_are_excluded() {
        let mut idx = Index::new();
        doc(&mut idx, "/a", &[("rust", 1)], &[("a", 1)]);
        doc(&mut idx, "/b", &[("rust", 1)], &[("b", 1)]);
        idx.remove_path(std::path::Path::new("/a"));
        let hits = search(&idx, "rust", &QueryOptions::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 1);
    }

    #[test]
    fn empty_query_returns_nothing() {
        let mut idx = Index::new();
        doc(&mut idx, "/a", &[("rust", 1)], &[("a", 1)]);
        assert!(search(&idx, "", &QueryOptions::default()).is_empty());
        assert!(search(&idx, "   ", &QueryOptions::default()).is_empty());
    }
}
