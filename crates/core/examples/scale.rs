//! Scale check: build a 100k-document in-memory index with a realistic
//! vocabulary and measure query latency. Run with:
//!
//!     cargo run --release --example scale -p deepsearch-core
//!
//! This bypasses the filesystem (the indexer is benchmarked separately in
//! `benches/bench.rs`) to isolate pure query latency at the target scale.

use std::collections::HashMap;
use std::time::Instant;

use deepsearch_core::index::{Index, PendingDoc};
use deepsearch_core::query::{search, QueryOptions};
use deepsearch_core::FileType;

const DOCS: usize = 100_000;
const VOCAB: usize = 8_000;
const WORDS_PER_DOC: usize = 60;

// Cheap deterministic LCG so runs are reproducible without an rng dependency.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 16
    }
}

fn word(i: usize) -> String {
    // Pronounceable-ish tokens like "thomer", "banquo", ... length 5-7.
    const C: &[u8] = b"bcdfghjklmnpqrstvwxyz";
    const V: &[u8] = b"aeiou";
    let mut s = String::new();
    let mut x = i as u64 * 2654435761;
    let len = 5 + (i % 3);
    for k in 0..len {
        x = x.wrapping_mul(48271).wrapping_add(1);
        let c = if k % 2 == 0 { C[(x as usize) % C.len()] } else { V[(x as usize) % V.len()] };
        s.push(c as char);
    }
    s
}

fn main() {
    println!("Building {DOCS} docs (vocab {VOCAB})...");
    let vocab: Vec<String> = (0..VOCAB).map(word).collect();
    let t0 = Instant::now();
    let mut idx = Index::new();
    let mut rng = Lcg(0x1234_5678);
    for d in 0..DOCS {
        let mut content_tf: HashMap<String, u32> = HashMap::new();
        for _ in 0..WORDS_PER_DOC {
            let w = &vocab[(rng.next() as usize) % VOCAB];
            *content_tf.entry(w.clone()).or_insert(0) += 1;
        }
        // Filename: two vocab words + index, so names share real words.
        let a = &vocab[(rng.next() as usize) % VOCAB];
        let b = &vocab[(rng.next() as usize) % VOCAB];
        let mut name_tf = HashMap::new();
        name_tf.insert(a.clone(), 1);
        name_tf.insert(b.clone(), 1);
        idx.add(PendingDoc {
            path: format!("/data/{a}_{b}_{d}.txt").into(),
            size: 1000,
            mtime: 1,
            file_type: FileType::Text,
            content_tf,
            name_tf,
            name_raw: vec![a.clone(), b.clone()],
        });
    }
    println!("  built in {:.2}s, {} content terms", t0.elapsed().as_secs_f64(), idx.content_index.len());

    let opts = QueryOptions::default();
    let bench = |label: &str, q: &str| {
        // warm up
        let _ = search(&idx, q, &opts);
        let n = 50;
        let t = Instant::now();
        let mut hits = 0;
        for _ in 0..n {
            hits = search(&idx, q, &opts).len();
        }
        let per = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  {label:<28} {per:6.2} ms/query   ({hits} hits)");
    };

    println!("Query latency over {DOCS} docs:");
    bench("single common term", &vocab[3]);
    bench("single rare term", &vocab[VOCAB - 1]);
    let multi = format!("{} {} {} {}", vocab[1], vocab[2], vocab[3], vocab[4]);
    bench("four-term query", &multi);
    // A typo of a real vocab word -> exercises the fuzzy filename path.
    let typo = transpose(&vocab[5]);
    bench("fuzzy typo (word not in idx)", &typo);
    let missing = "zzzqqword";
    bench("no-match term (fuzzy scan)", missing);
}

/// Swap the middle two characters to make a transposition typo.
fn transpose(s: &str) -> String {
    let mut c: Vec<char> = s.chars().collect();
    if c.len() >= 3 {
        let m = c.len() / 2;
        c.swap(m - 1, m);
    }
    c.into_iter().collect()
}
