//! Criterion benchmarks for indexing throughput and query latency.
//!
//! Run with `cargo bench`. The query benchmark builds a synthetic corpus of
//! ~10k documents in a tempdir and measures single-query latency; scale
//! `CORPUS` up toward 100k to check the <50ms target on real hardware.

use std::io::Write;
use std::path::Path;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use deepsearch_core::indexer::{build, IndexOptions, Progress};
use deepsearch_core::query::{search, QueryOptions};
use deepsearch_core::index::Index;

/// Number of synthetic documents. Kept modest so `cargo bench` stays quick in
/// CI; raise locally to stress-test.
const CORPUS: usize = 10_000;

const WORDS: &[&str] = &[
    "rust", "index", "search", "engine", "token", "vector", "matrix", "ranking",
    "query", "document", "corpus", "score", "relevance", "inverted", "posting",
    "filesystem", "parallel", "stemming", "camel", "snake", "config", "server",
    "client", "async", "future", "thread", "memory", "cache", "buffer", "stream",
];

fn word(seed: usize) -> &'static str {
    WORDS[seed % WORDS.len()]
}

fn make_corpus(dir: &Path) {
    for i in 0..CORPUS {
        let path = dir.join(format!("doc_{i}.txt"));
        let mut f = std::fs::File::create(&path).unwrap();
        // 40 pseudo-random words per doc.
        let mut line = String::new();
        for j in 0..40 {
            line.push_str(word(i * 7 + j * 13));
            line.push(' ');
        }
        f.write_all(line.as_bytes()).unwrap();
    }
}

fn build_index(dir: &Path) -> Index {
    let mut idx = Index::new();
    build(&mut idx, dir, &IndexOptions::default(), &Progress::default());
    idx
}

fn bench_indexing(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    make_corpus(dir.path());
    let mut group = c.benchmark_group("indexing");
    group.sample_size(10);
    group.bench_function(BenchmarkId::new("build", CORPUS), |b| {
        b.iter(|| build_index(dir.path()));
    });
    group.finish();
}

fn bench_query(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    make_corpus(dir.path());
    let idx = build_index(dir.path());
    let opts = QueryOptions::default();

    let mut group = c.benchmark_group("query");
    group.bench_function(BenchmarkId::new("single_term", CORPUS), |b| {
        b.iter(|| search(&idx, "ranking", &opts));
    });
    group.bench_function(BenchmarkId::new("multi_term", CORPUS), |b| {
        b.iter(|| search(&idx, "inverted index ranking score", &opts));
    });
    group.finish();
}

criterion_group!(benches, bench_indexing, bench_query);
criterion_main!(benches);
