//! `deepsearch` — relevance-ranked full-text search over your files.
//!
//! Subcommands:
//!   * `index [PATH]`   build (or `--incremental` update) the index.
//!   * `query "<q>"`    one-shot ranked search (`--json` for scripts).
//!   * `tui [PATH]`     interactive fuzzy search UI.
//!   * `stats`          report on the current index.

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
        Some(Command::Index { path, incremental }) => cmd_index(cache, path, incremental),
        Some(Command::Query { query, limit, json }) => {
            cmd_query(cache, &query.join(" "), limit, json)
        }
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
    Ok(())
}

fn cmd_query(cache: Option<&std::path::Path>, query: &str, limit: usize, json: bool) -> Result<()> {
    let ds = load_or_hint(cache)?;
    let opts = QueryOptions {
        limit,
        ..QueryOptions::default()
    };
    let results = ds.search(query, &opts);

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
