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

use crate::extract::FileType;
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
    /// Enable prefix matching against filename terms, so a partially-typed word
    /// (`conf`) matches longer filename tokens (`config`). This is what makes
    /// the interactive UI filter as you type.
    pub prefix: bool,
    /// Multiplier applied to prefix (incomplete-word) filename matches. Kept
    /// below an exact match but above a fuzzy one.
    pub prefix_penalty: f32,
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
            prefix: true,
            prefix_penalty: 0.7,
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
///
/// The query may carry inline **filters** that restrict results by file type or
/// extension: `type:image logo`, `ext:rs parser`, `type:pdf` (filter only). See
/// [`parse_filters`].
pub fn search(index: &Index, query: &str, opts: &QueryOptions) -> Vec<Hit> {
    if index.live_docs == 0 {
        return Vec::new();
    }
    // Pull `type:`/`ext:` tokens out before tokenizing the rest as search terms.
    let (clean, filters) = parse_filters(query);

    // Keep the unstemmed tokens so the fuzzy pass can compare typos against real
    // filename words; derive the stemmed term per token for exact matching.
    let raw_terms = normalize(&clean);
    if raw_terms.is_empty() {
        // A filter with no search terms ("show all my PDFs") browses the corpus:
        // every matching live doc, most-recently-modified first.
        return filter_only_hits(index, &filters, opts.limit);
    }
    let scores = keyword_scores(index, &raw_terms, opts);
    finalize(scores, index, &filters, opts.limit)
}

/// Accumulate BM25 keyword scores for the (unstemmed) `raw_terms` over the
/// content and filename fields, including the prefix and fuzzy filename
/// fallbacks. Shared by keyword and hybrid search.
fn keyword_scores(index: &Index, raw_terms: &[String], opts: &QueryOptions) -> HashMap<u32, f32> {
    let n = index.live_docs;
    let avg_content = index.avg_content_len();
    let avg_name = index.avg_name_len();

    let mut scores: HashMap<u32, f32> = HashMap::new();

    for raw in raw_terms {
        let term = stem_word(raw);
        // --- content field ---
        if let Some(postings) = index.content_index.get(&term) {
            let live: Vec<_> = postings
                .iter()
                .filter(|p| index.is_live(p.doc_id))
                .collect();
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
            let live: Vec<_> = postings
                .iter()
                .filter(|p| index.is_live(p.doc_id))
                .collect();
            let df = live.len() as u64;
            if df > 0 {
                name_hit = true;
                let term_idf = idf(n, df);
                for p in live {
                    let dl = index.doc(p.doc_id).map(|d| d.name_len).unwrap_or(0);
                    let s =
                        opts.name_boost * term_idf * bm25_tf(p.tf, dl, avg_name, opts.k1, opts.b);
                    *scores.entry(p.doc_id).or_insert(0.0) += s;
                }
            }
        }

        // --- name field (prefix fallback) ---
        // When the exact term didn't hit a filename, treat it as a prefix so a
        // half-typed word still matches (`conf` -> `config`). This drives the
        // filter-as-you-type feel of the interactive UI.
        let mut prefix_hit = false;
        if opts.prefix && !name_hit {
            prefix_hit = prefix_name_match(index, raw, opts, avg_name, n, &mut scores);
        }

        // --- name field (fuzzy fallback) ---
        // Only spend the linear scan when neither exact nor prefix matched, so
        // the common case stays fast and typo tolerance is a last resort.
        if opts.fuzzy && !name_hit && !prefix_hit {
            fuzzy_name_match(index, raw, opts, avg_name, n, &mut scores);
        }
    }

    scores
}

/// Apply filters, sort by descending score (ties broken by ascending doc_id for
/// determinism), and cap to `limit`.
fn finalize(scores: HashMap<u32, f32>, index: &Index, filters: &Filters, limit: usize) -> Vec<Hit> {
    let mut hits: Vec<Hit> = scores
        .into_iter()
        .filter(|(doc_id, _)| filters.matches(index, *doc_id))
        .map(|(doc_id, score)| Hit { doc_id, score })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.doc_id.cmp(&b.doc_id))
    });
    hits.truncate(limit);
    hits
}

/// Dot product of two equal-length vectors. For unit-normalized embeddings this
/// equals cosine similarity.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Rank documents purely by semantic similarity to `query_vec` (assumed
/// unit-normalized). Only docs carrying an embedding of matching dimension
/// participate.
pub fn semantic_search(index: &Index, query_vec: &[f32], limit: usize) -> Vec<Hit> {
    if index.live_docs == 0 || query_vec.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<Hit> = index
        .docs
        .iter()
        .flatten()
        .filter_map(|m| {
            let emb = m.embedding.as_ref()?;
            (emb.len() == query_vec.len()).then(|| Hit {
                doc_id: m.id,
                score: dot(query_vec, emb),
            })
        })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.doc_id.cmp(&b.doc_id))
    });
    hits.truncate(limit);
    hits
}

/// Hybrid ranking: blend keyword (BM25) relevance with semantic similarity.
///
/// Keyword scores are min-max normalized to `[0, 1]`; semantic similarity uses
/// the positive part of cosine. The two are combined as
/// `(1 - w)·keyword + w·semantic`, where `w = semantic_weight`. The union of
/// docs that match either signal is returned, so a file can surface by meaning
/// even with no keyword hit (and vice-versa). Inline `type:`/`ext:` filters
/// still apply.
pub fn hybrid_search(
    index: &Index,
    query: &str,
    query_vec: &[f32],
    opts: &QueryOptions,
    semantic_weight: f32,
) -> Vec<Hit> {
    if index.live_docs == 0 {
        return Vec::new();
    }
    let (clean, filters) = parse_filters(query);
    let raw_terms = normalize(&clean);
    let w = semantic_weight.clamp(0.0, 1.0);

    let mut combined: HashMap<u32, f32> = HashMap::new();

    // Keyword side, normalized by its own maximum so the blend is scale-free.
    if !raw_terms.is_empty() {
        let kw = keyword_scores(index, &raw_terms, opts);
        let kw_max = kw.values().copied().fold(0.0_f32, f32::max);
        if kw_max > 0.0 {
            for (id, s) in kw {
                *combined.entry(id).or_insert(0.0) += (1.0 - w) * (s / kw_max);
            }
        }
    }

    // Semantic side: positive cosine similarity against the query vector.
    if !query_vec.is_empty() && w > 0.0 {
        for m in index.docs.iter().flatten() {
            if let Some(emb) = &m.embedding {
                if emb.len() == query_vec.len() {
                    let sim = dot(query_vec, emb).max(0.0);
                    if sim > 0.0 {
                        *combined.entry(m.id).or_insert(0.0) += w * sim;
                    }
                }
            }
        }
    }

    finalize(combined, index, &filters, opts.limit)
}

/// Post-ranking restrictions parsed out of the query string.
#[derive(Debug, Default, PartialEq)]
pub struct Filters {
    /// Allowed extensions (lowercased, no dot). Empty = any.
    pub exts: Vec<String>,
    /// Allowed file types. Empty = any.
    pub types: Vec<FileType>,
}

impl Filters {
    pub fn is_empty(&self) -> bool {
        self.exts.is_empty() && self.types.is_empty()
    }

    /// Whether the doc satisfies every active filter category (types AND exts;
    /// within a category the values are OR-ed).
    fn matches(&self, index: &Index, doc_id: u32) -> bool {
        let Some(meta) = index.doc(doc_id) else {
            return false;
        };
        if !self.types.is_empty() && !self.types.contains(&meta.file_type) {
            return false;
        }
        if !self.exts.is_empty() {
            let ext = meta
                .path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase());
            match ext {
                Some(e) if self.exts.contains(&e) => {}
                _ => return false,
            }
        }
        true
    }
}

/// Split a raw query into (search text, filters), extracting `type:` and `ext:`
/// tokens. `type:` accepts the [`FileType`] names plus a few aliases
/// (`img`→image, `doc`→docx, `txt`→text); unknown values are dropped. Filters
/// can repeat (`ext:rs ext:toml`) and combine with search terms.
pub fn parse_filters(query: &str) -> (String, Filters) {
    let mut filters = Filters::default();
    let mut terms: Vec<&str> = Vec::new();

    for tok in query.split_whitespace() {
        if let Some(v) = strip_prefix_ci(tok, "ext:") {
            let v = v.trim_start_matches('.').to_ascii_lowercase();
            if !v.is_empty() {
                filters.exts.push(v);
            }
        } else if let Some(v) = strip_prefix_ci(tok, "type:") {
            match parse_file_type(&v.to_ascii_lowercase()) {
                Some(ft) => filters.types.push(ft),
                // Unknown kind (e.g. a model guessing `type:rust`): don't drop it
                // silently — fall back to treating the value as a search keyword.
                None => terms.push(v),
            }
        } else {
            terms.push(tok);
        }
    }
    (terms.join(" "), filters)
}

/// Case-insensitive `strip_prefix` for ASCII filter keywords.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

fn parse_file_type(v: &str) -> Option<FileType> {
    Some(match v {
        // Source code is indexed as text (the extractor never emits `Code`), so
        // `type:code` and `type:text` both resolve to the text bucket — combine
        // with `ext:` to narrow to a language.
        "text" | "txt" | "plain" | "code" | "source" => FileType::Text,
        "pdf" => FileType::Pdf,
        "docx" | "doc" | "word" => FileType::Docx,
        "image" | "img" | "picture" => FileType::Image,
        "binary" | "bin" => FileType::Binary,
        _ => return None,
    })
}

/// Results for a filter-only query: all matching live docs, newest first,
/// carrying a score of 0 (there is nothing to rank).
fn filter_only_hits(index: &Index, filters: &Filters, limit: usize) -> Vec<Hit> {
    if filters.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<(i64, u32)> = index
        .docs
        .iter()
        .flatten()
        .filter(|m| filters.matches(index, m.id))
        .map(|m| (m.mtime, m.id))
        .collect();
    // Most-recently-modified first; ties broken by doc_id for determinism.
    hits.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    hits.truncate(limit);
    hits.into_iter()
        .map(|(_, doc_id)| Hit { doc_id, score: 0.0 })
        .collect()
}

/// Scan the **unstemmed** filename dictionary for tokens that have `raw` as a
/// strict prefix (`conf` -> `config`, `configure`) and score their docs with a
/// small penalty. Strict prefix (candidate longer than the query token) keeps
/// this pass disjoint from the exact/fuzzy passes: a fully-typed token is served
/// by those, an incomplete one by this.
///
/// Returns whether anything matched, so the caller can suppress the fuzzy pass.
/// The scan is over the same dictionary as fuzzy matching; a single-character
/// query is skipped so the very first keystroke doesn't select the whole corpus.
fn prefix_name_match(
    index: &Index,
    raw: &str,
    opts: &QueryOptions,
    avg_name: f32,
    n: u64,
    scores: &mut HashMap<u32, f32>,
) -> bool {
    if raw.chars().count() < 2 {
        return false;
    }
    let mut matched = false;
    for (cand, doc_ids) in &index.name_fuzzy {
        // Strict prefix: skip the exact token (handled by the exact pass) and
        // anything that doesn't start with what the user typed.
        if cand.len() <= raw.len() || !cand.starts_with(raw) {
            continue;
        }
        let live: Vec<u32> = doc_ids
            .iter()
            .copied()
            .filter(|&id| index.is_live(id))
            .collect();
        let df = live.len() as u64;
        if df == 0 {
            continue;
        }
        // A prefix that covers more of the candidate is a stronger signal
        // (`config` typed as `confi` beats `config` typed as `co`).
        let coverage = raw.chars().count() as f32 / cand.chars().count() as f32;
        let factor = opts.name_boost * opts.prefix_penalty * (0.5 + 0.5 * coverage);
        let term_idf = idf(n, df);
        for id in live {
            let dl = index.doc(id).map(|d| d.name_len).unwrap_or(0);
            // Prefix hits carry an implicit term frequency of 1.
            let s = factor * term_idf * bm25_tf(1, dl, avg_name, opts.k1, opts.b);
            *scores.entry(id).or_insert(0.0) += s;
        }
        matched = true;
    }
    matched
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
        let live: Vec<u32> = doc_ids
            .iter()
            .copied()
            .filter(|&id| index.is_live(id))
            .collect();
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
        doc(
            &mut idx,
            "/a",
            &[("report", 1), ("filler", 20)],
            &[("a", 1)],
        );
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
    fn prefix_matches_incomplete_word() {
        let mut idx = Index::new();
        doc(
            &mut idx,
            "/config.rs",
            &[("stuff", 1)],
            &[("config", 1), ("rs", 1)],
        );
        doc(
            &mut idx,
            "/other.rs",
            &[("stuff", 1)],
            &[("other", 1), ("rs", 1)],
        );
        // "conf" is a strict prefix of "config" but of nothing in /other.rs.
        let hits = search(&idx, "conf", &QueryOptions::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 0);
    }

    #[test]
    fn prefix_ignored_when_disabled() {
        let mut idx = Index::new();
        doc(&mut idx, "/config.rs", &[("stuff", 1)], &[("config", 1)]);
        let opts = QueryOptions {
            prefix: false,
            fuzzy: false,
            ..QueryOptions::default()
        };
        assert!(search(&idx, "conf", &opts).is_empty());
    }

    #[test]
    fn ext_filter_restricts_results() {
        let mut idx = Index::new();
        doc(&mut idx, "/notes.md", &[("report", 3)], &[("notes", 1)]);
        doc(&mut idx, "/report.txt", &[("report", 3)], &[("report", 1)]);
        // Same term, but only the .md file should survive `ext:md`.
        let hits = search(&idx, "report ext:md", &QueryOptions::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 0);
    }

    #[test]
    fn type_filter_only_browses_corpus() {
        let mut idx = Index::new();
        // No search terms — a bare `type:` filter lists matching docs.
        idx.add(PendingDoc {
            path: PathBuf::from("/a.png"),
            size: 1,
            mtime: 10,
            file_type: FileType::Image,
            content_tf: tf(&[]),
            name_tf: tf(&[("a", 1)]),
            name_raw: vec!["a".to_string()],
        });
        idx.add(PendingDoc {
            path: PathBuf::from("/b.txt"),
            size: 1,
            mtime: 20,
            file_type: FileType::Text,
            content_tf: tf(&[("hi", 1)]),
            name_tf: tf(&[("b", 1)]),
            name_raw: vec!["b".to_string()],
        });
        let hits = search(&idx, "type:image", &QueryOptions::default());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, 0);
    }

    #[test]
    fn parse_filters_splits_terms_and_filters() {
        let (clean, f) = parse_filters("ext:.RS parser TYPE:img");
        assert_eq!(clean, "parser");
        assert_eq!(f.exts, vec!["rs".to_string()]);
        assert_eq!(f.types, vec![FileType::Image]);
    }

    #[test]
    fn unknown_type_becomes_a_keyword() {
        // An invalid `type:` (e.g. a model guessing) is kept as a search term
        // rather than silently dropped, so the query never comes out empty.
        let (clean, f) = parse_filters("type:rust");
        assert_eq!(clean, "rust");
        assert!(f.types.is_empty());
    }

    #[test]
    fn type_code_matches_text() {
        // Source is indexed as text, so `type:code` resolves to the text bucket.
        let mut idx = Index::new();
        doc(
            &mut idx,
            "/main.rs",
            &[("code", 1)],
            &[("main", 1), ("rs", 1)],
        );
        let hits = search(&idx, "type:code", &QueryOptions::default());
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn semantic_ranks_by_cosine() {
        let mut idx = Index::new();
        doc(&mut idx, "/a", &[("x", 1)], &[("a", 1)]);
        doc(&mut idx, "/b", &[("y", 1)], &[("b", 1)]);
        idx.set_embedding(0, vec![1.0, 0.0]);
        idx.set_embedding(1, vec![0.0, 1.0]);
        // Query vector leans toward doc0's embedding.
        let hits = semantic_search(&idx, &[0.9, 0.1], 10);
        assert_eq!(hits[0].doc_id, 0);
    }

    #[test]
    fn hybrid_surfaces_semantic_only_match() {
        let mut idx = Index::new();
        // doc0 matches the keyword; doc1 matches only by embedding.
        doc(&mut idx, "/a", &[("login", 1)], &[("a", 1)]);
        doc(&mut idx, "/b", &[("auth", 1)], &[("b", 1)]);
        idx.set_embedding(0, vec![0.0, 1.0]);
        idx.set_embedding(1, vec![1.0, 0.0]);
        // Query vector points at doc1; keyword "login" points at doc0.
        let hits = hybrid_search(&idx, "login", &[1.0, 0.0], &QueryOptions::default(), 0.5);
        let ids: Vec<u32> = hits.iter().map(|h| h.doc_id).collect();
        assert!(ids.contains(&0), "keyword match present");
        assert!(ids.contains(&1), "semantic-only match surfaced");
    }

    #[test]
    fn fuzzy_ignored_when_disabled() {
        let mut idx = Index::new();
        doc(&mut idx, "/config", &[("stuff", 1)], &[("config", 1)]);
        let opts = QueryOptions {
            fuzzy: false,
            ..QueryOptions::default()
        };
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
