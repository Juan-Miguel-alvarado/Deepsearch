//! `deepsearch` — relevance-ranked full-text search over your files.
//!
//! Subcommands:
//!   * `index [PATH]`   build (or `--incremental` update) the index.
//!   * `query "<q>"`    one-shot ranked search (`--json` for scripts).
//!   * `ask "<q>"`      answer a question from your files (local Ollama).
//!   * `tui [PATH]`     interactive fuzzy search UI.
//!   * `stats`          report on the current index.

mod ai;
mod clip;
mod open;
mod preview;
mod tui;
mod util;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use deepsearch_core::{DeepSearch, IndexOptions, Progress, QueryOptions};

#[derive(Parser)]
#[command(
    name = "deepsearch",
    version,
    about = "Relevance-ranked full-text search over all your files"
)]
struct Cli {
    /// Path to the index cache (default: ~/.cache/deepsearch/index.bin).
    #[arg(long, global = true)]
    cache: Option<PathBuf>,

    /// Subcommand. When omitted, `deepsearch` launches the interactive TUI.
    #[command(subcommand)]
    cmd: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Build or update the index.
    Index {
        /// Root directory to index (default: your home directory).
        path: Option<PathBuf>,
        /// Reindex only changed files instead of a full rebuild.
        #[arg(long)]
        incremental: bool,
        /// Also compute semantic embeddings (search by meaning). Needs a local
        /// Ollama with an embedding model (`ollama pull nomic-embed-text`).
        #[arg(long)]
        semantic: bool,
    },
    /// Run a one-shot query and print ranked results.
    Query {
        /// The search query (all words joined).
        #[arg(required = true)]
        query: Vec<String>,
        /// Maximum number of results.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Emit results as JSON.
        #[arg(long)]
        json: bool,
        /// Force keyword-only ranking even if the index has embeddings.
        #[arg(long)]
        keyword: bool,
    },
    /// Ask a question about your files; a local model answers from their
    /// contents and cites the sources.
    Ask {
        /// The question (all words joined).
        #[arg(required = true)]
        request: Vec<String>,
        /// Maximum number of results.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// Emit results as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Launch the interactive TUI.
    Tui {
        /// Root to index if the cache is empty (default: your home directory).
        path: Option<PathBuf>,
    },
    /// Print statistics about the current index.
    Stats,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cache = cli.cache.as_deref();

    match cli.cmd {
        // No subcommand → refresh the index of your home dir, then open the TUI.
        None => cmd_default(cache),
        Some(Command::Index {
            path,
            incremental,
            semantic,
        }) => cmd_index(cache, path, incremental, semantic),
        Some(Command::Query {
            query,
            limit,
            json,
            keyword,
        }) => cmd_query(cache, &query.join(" "), limit, json, keyword),
        Some(Command::Ask {
            request,
            limit,
            json,
        }) => cmd_ask(cache, &request.join(" "), limit, json),
        Some(Command::Tui { path }) => cmd_tui(cache, path),
        Some(Command::Stats) => cmd_stats(cache),
    }
}

fn default_root() -> Result<PathBuf> {
    dirs::home_dir().context("could not determine your home directory; pass a path explicitly")
}

fn cmd_index(
    cache: Option<&std::path::Path>,
    path: Option<PathBuf>,
    incremental: bool,
    semantic: bool,
) -> Result<()> {
    let root = match path {
        Some(p) => p,
        None => default_root()?,
    };
    let opts = IndexOptions::default();

    // For an incremental run we need the previous index; otherwise start fresh.
    let mut ds = if incremental {
        DeepSearch::open_or_empty(cache)?
    } else {
        DeepSearch::empty()
    };

    println!(
        "{} {}",
        if incremental {
            "Updating index for"
        } else {
            "Indexing"
        },
        root.display()
    );

    let stats = with_progress("Indexing", |progress| {
        if incremental {
            ds.update(&root, &opts, progress)
        } else {
            ds.build(&root, &opts, progress)
        }
    });

    // Reclaim space if incremental churn left many tombstones.
    if ds.maybe_compact(0.3) {
        println!("Compacted index (reclaimed tombstoned documents).");
    }

    ds.save(cache).context("saving index")?;
    println!(
        "Done: {} indexed, {} unchanged, {} removed, {} errors. {} live documents.",
        stats.indexed,
        stats.unchanged,
        stats.removed,
        stats.errors,
        ds.len()
    );

    if semantic {
        build_embeddings(&mut ds, cache)?;
    }
    Ok(())
}

/// Weight of the semantic signal in hybrid ranking (0 = keyword only, 1 = pure
/// semantic).
const SEMANTIC_WEIGHT: f32 = 0.5;

/// Compute a semantic embedding for every document that lacks one, via Ollama,
/// then persist. Documents with no extractable text fall back to embedding their
/// file name so they can still be found by meaning.
fn build_embeddings(ds: &mut DeepSearch, cache: Option<&std::path::Path>) -> Result<()> {
    if !ai::embed_available() {
        println!("Skipping embeddings: {}", ai::embed_setup_hint());
        return Ok(());
    }
    let todo = ds.docs_needing_embedding();
    let total = todo.len();
    if total == 0 {
        println!("Embeddings already up to date.");
        return Ok(());
    }
    println!("Building semantic embeddings for {total} documents (local Ollama)...");

    let mut done = 0usize;
    let mut failed = 0usize;
    for (id, path) in todo {
        let text = match deepsearch_core::extract::extract(&path) {
            Ok(e) => e.text.unwrap_or_default(),
            Err(_) => String::new(),
        };
        // No body text (image/binary): embed the file name instead.
        let basis = if text.trim().is_empty() {
            path.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            text
        };
        if basis.trim().is_empty() {
            continue;
        }
        match ai::embed(&basis, false) {
            Ok(v) => ds.set_embedding(id, v),
            Err(_) => failed += 1,
        }
        done += 1;
        if done.is_multiple_of(10) || done == total {
            eprint!("\rembedded {done}/{total}      ");
        }
    }
    eprintln!("\rembedded {done}/{total} documents ({failed} failed).      ");
    ds.save(cache).context("saving embeddings")?;
    Ok(())
}

/// Run a search, using hybrid keyword+semantic ranking when the index has
/// embeddings and Ollama is reachable; otherwise plain keyword search.
fn run_search(
    ds: &DeepSearch,
    query: &str,
    opts: &QueryOptions,
    keyword_only: bool,
) -> (Vec<deepsearch_core::SearchResult>, bool) {
    if !keyword_only && ds.has_embeddings() && ai::available() {
        if let Ok(qv) = ai::embed(query, true) {
            return (ds.hybrid_search(query, &qv, opts, SEMANTIC_WEIGHT), true);
        }
    }
    (ds.search(query, opts), false)
}

fn cmd_query(
    cache: Option<&std::path::Path>,
    query: &str,
    limit: usize,
    json: bool,
    keyword: bool,
) -> Result<()> {
    let ds = load_or_hint(cache)?;
    let opts = QueryOptions {
        limit,
        ..QueryOptions::default()
    };
    let (results, semantic) = run_search(&ds, query, &opts, keyword);
    if semantic && !json {
        eprintln!("(hybrid keyword + semantic)");
    }

    if json {
        print_json(&results);
    } else if results.is_empty() {
        println!("No results for {query:?}.");
    } else {
        for (i, r) in results.iter().enumerate() {
            println!(
                "{:>2}. {:>7.3}  [{}]  {}",
                i + 1,
                r.score,
                r.file_type.as_str(),
                r.path.display()
            );
        }
    }
    Ok(())
}

/// How many files are fed to the model as context when answering a question.
/// Small on purpose: a local CPU model slows down fast as context grows.
const ANSWER_SOURCES: usize = 3;
/// Characters of each file handed to the model — a *relevant* window, not the
/// head of the file.
const SNIPPET_CHARS: usize = 700;

/// Locate a question word in `haystack`, tolerating simple inflection: if the
/// whole word isn't there, retry with successively shorter prefixes so
/// `passwords` still finds `password`.
fn find_term(haystack: &str, term: &str) -> Option<usize> {
    let mut candidate: &str = term;
    loop {
        if let Some(pos) = haystack.find(candidate) {
            return Some(pos);
        }
        if candidate.chars().count() <= 4 {
            return None;
        }
        let mut chars = candidate.chars();
        chars.next_back();
        candidate = chars.as_str();
    }
}

/// Pick the most useful window of `text` for `question`.
///
/// Feeding the first N characters of a file usually means feeding its imports
/// and licence header. Centring the window on the first place a question word
/// appears gives the model the part that actually bears on the question — which
/// both improves the answer and keeps the prompt (and so the latency) small.
fn relevant_snippet(text: &str, question: &str, max_chars: usize) -> String {
    let lower = text.to_lowercase();
    let hit = question
        .to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.chars().count() >= 3)
        .filter_map(|w| find_term(&lower, &w))
        .min();

    let start = match hit {
        // Back up a little so the match has context before it.
        Some(byte_pos) => text[..byte_pos]
            .chars()
            .count()
            .saturating_sub(max_chars / 3),
        None => 0,
    };
    text.chars().skip(start).take(max_chars).collect()
}

/// Ask a question about your files and get an **answer**, not just a file list.
///
/// Retrieval-augmented: the question is used to find the most relevant documents
/// (semantically when embeddings exist), their text is extracted, and a local
/// model answers from those excerpts, citing which ones it used.
fn cmd_ask(
    cache: Option<&std::path::Path>,
    question: &str,
    limit: usize,
    json: bool,
) -> Result<()> {
    if !ai::available() {
        anyhow::bail!(
            "asking questions needs a local Ollama server.\n\
             Install it from https://ollama.com, run `ollama serve`, and pull a model \
             (e.g. `ollama pull llama3.2`). deepsearch works without it — use `query` instead."
        );
    }
    let ds = load_or_hint(cache)?;

    // 1. Find the files most likely to hold the answer.
    let opts = QueryOptions {
        limit: limit.max(ANSWER_SOURCES),
        ..QueryOptions::default()
    };
    let (results, semantic) = run_search(&ds, question, &opts, false);
    if results.is_empty() {
        println!("No files matched that question.");
        return Ok(());
    }

    // 2. Read the top candidates; skip anything with no extractable text.
    let mut sources: Vec<(String, String)> = Vec::new();
    for r in &results {
        if sources.len() >= ANSWER_SOURCES {
            break;
        }
        if let Ok(ext) = deepsearch_core::extract::extract(&r.path) {
            if let Some(text) = ext.text {
                if !text.trim().is_empty() {
                    let snippet = relevant_snippet(&text, question, SNIPPET_CHARS);
                    sources.push((r.path.display().to_string(), snippet));
                }
            }
        }
    }
    if sources.is_empty() {
        println!("Found matching files, but none had readable text to answer from.");
        return Ok(());
    }

    eprintln!(
        "reading {} file(s){}…",
        sources.len(),
        if semantic { " (semantic match)" } else { "" }
    );

    // 3. Let the local model answer from those excerpts.
    let answer = match ai::answer_with_context(question, &sources) {
        Ok(a) => a,
        Err(e) => anyhow::bail!("{e}"),
    };

    if json {
        print_answer_json(&answer, &sources);
    } else {
        println!("\n{answer}\n");
        println!("Sources:");
        for (i, (label, _)) in sources.iter().enumerate() {
            println!("  [{}] {}", i + 1, label);
        }
    }
    Ok(())
}

/// Hand-rolled JSON for `ask --json` (keeps stdout machine-readable).
fn print_answer_json(answer: &str, sources: &[(String, String)]) {
    let esc = |s: &str| {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    };
    println!("{{");
    println!("  \"answer\": \"{}\",", esc(answer));
    println!("  \"sources\": [");
    for (i, (label, _)) in sources.iter().enumerate() {
        let comma = if i + 1 < sources.len() { "," } else { "" };
        println!("    \"{}\"{}", esc(label), comma);
    }
    println!("  ]");
    println!("}}");
}

/// Default action (bare `deepsearch`): incrementally refresh the index of the
/// home directory, then launch the TUI. The first run is a full build; later
/// runs only reprocess files whose mtime changed, so startup stays quick.
fn cmd_default(cache: Option<&std::path::Path>) -> Result<()> {
    let root = default_root()?;
    let mut ds = DeepSearch::open_or_empty(cache)?;

    let fresh = ds.is_empty();
    println!(
        "{} {} ...",
        if fresh {
            "Indexing"
        } else {
            "Refreshing index for"
        },
        root.display()
    );
    with_progress("Indexing", |progress| {
        ds.update(&root, &IndexOptions::default(), progress)
    });
    if ds.maybe_compact(0.3) {
        println!("Compacted index.");
    }
    ds.save(cache).context("saving index")?;

    tui::App::new(ds, QueryOptions::default()).run()
}

fn cmd_tui(cache: Option<&std::path::Path>, path: Option<PathBuf>) -> Result<()> {
    let mut ds = DeepSearch::open_or_empty(cache)?;

    // First run with an empty cache: build the index before entering the UI.
    if ds.is_empty() {
        let root = match path {
            Some(p) => p,
            None => default_root()?,
        };
        println!(
            "No index found — building one for {} first...",
            root.display()
        );
        with_progress("Indexing", |progress| {
            ds.build(&root, &IndexOptions::default(), progress)
        });
        ds.save(cache).context("saving index")?;
    }

    tui::App::new(ds, QueryOptions::default()).run()
}

fn cmd_stats(cache: Option<&std::path::Path>) -> Result<()> {
    let ds = load_or_hint(cache)?;
    let idx = ds.index();
    println!("Live documents:     {}", idx.live_docs);
    println!("Doc slots (w/ dead):{}", idx.docs.len());
    println!("Content terms:      {}", idx.content_index.len());
    println!("Filename terms:     {}", idx.name_index.len());
    println!("Avg content length: {:.1} tokens", idx.avg_content_len());
    println!("Tombstone ratio:    {:.1}%", idx.dead_ratio() * 100.0);
    Ok(())
}

/// Load the index, or print a friendly hint if it hasn't been built yet.
fn load_or_hint(cache: Option<&std::path::Path>) -> Result<DeepSearch> {
    match DeepSearch::load(cache) {
        Ok(ds) => Ok(ds),
        Err(deepsearch_core::DeepSearchError::NotFound(_)) => {
            anyhow::bail!("no index found. Run `deepsearch index` first.")
        }
        Err(e) => Err(e.into()),
    }
}

/// Run `f`, displaying a live progress line driven by the shared counter.
fn with_progress<F, R>(label: &str, f: F) -> R
where
    F: FnOnce(&Progress) -> R,
{
    let progress = Arc::new(Progress::default());
    let done_flag = Arc::new(AtomicBool::new(false));

    let p = progress.clone();
    let flag = done_flag.clone();
    let label_owned = label.to_string();
    let printer = std::thread::spawn(move || {
        while !flag.load(Ordering::Relaxed) {
            let (done, total) = p.snapshot();
            if total > 0 {
                let pct = (done as f64 / total as f64 * 100.0).min(100.0);
                eprint!("\r{label_owned}: {done}/{total} ({pct:.0}%)      ");
            } else {
                eprint!("\r{label_owned}: scanning...      ");
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    });

    let result = f(&progress);

    done_flag.store(true, Ordering::Relaxed);
    let _ = printer.join();
    let (done, total) = progress.snapshot();
    eprintln!("\r{label}: {done}/{total} files scanned.            ");
    result
}

fn print_json(results: &[deepsearch_core::SearchResult]) {
    // Hand-rolled to avoid a serde_json dependency for a tiny output.
    println!("[");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        println!(
            "  {{\"score\": {:.4}, \"type\": \"{}\", \"size\": {}, \"path\": {:?}}}{}",
            r.score,
            r.file_type.as_str(),
            r.size,
            r.path.display().to_string(),
            comma
        );
    }
    println!("]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_centres_on_the_match() {
        let text = format!(
            "{}PASSWORD hashing happens here{}",
            "x".repeat(500),
            "y".repeat(500)
        );
        let snip = relevant_snippet(&text, "how are passwords handled", 200);
        assert!(
            snip.contains("PASSWORD hashing"),
            "window should include the match"
        );
        assert!(snip.chars().count() <= 200);
    }

    #[test]
    fn snippet_falls_back_to_head_without_a_match() {
        let text = "alpha beta gamma delta";
        let snip = relevant_snippet(text, "zzzz nothing matches", 10);
        assert_eq!(snip, "alpha beta");
    }
}
