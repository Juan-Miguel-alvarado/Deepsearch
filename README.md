# deepsearch

Relevance-ranked full-text search over **all** your files, from the terminal.

`deepsearch` is not `find` or `grep`. It scans your filesystem **once**, builds a
persistent inverted index, and then answers queries in **sub-millisecond** time,
ranked by **BM25 relevance** — searching both file *names* and file *contents*.
An interactive TUI gives you incremental search with live, syntax-highlighted
previews (and image previews via the terminal graphics protocol).

```
deepsearch index ~/projects      # scan & index once (incremental afterwards)
deepsearch query "bm25 ranking"  # ranked results, name + content
deepsearch tui                   # interactive fuzzy search
```

---



https://github.com/user-attachments/assets/0860a6b1-cf3e-4f9d-b826-25801c7f3596



## Architecture

Three layers, with a hard boundary between the engine and the UI:

```
┌──────────────────────────────────────────────┐
│ crates/cli   deepsearch (binary)              │
│   · clap CLI  (index / query / tui / stats)   │
│   · ratatui TUI + async preview worker        │
└───────────────▲──────────────────────────────┘
                │  public API only
┌───────────────┴──────────────────────────────┐
│ crates/core  deepsearch-core (library)        │
│   1. Indexer  — walk, extract, build index    │
│   2. Query    — tokenize, BM25 rank, fuzzy     │
│   (usable with no TUI, no CLI)                 │
└──────────────────────────────────────────────┘
```

Layers 1 and 2 live in `deepsearch-core`, a standalone library with a public
API ([`DeepSearch`], [`indexer`], [`query`]). The TUI depends on the library; the
library depends on nothing UI-related. You can embed the engine in any program:

```rust
use deepsearch_core::{DeepSearch, IndexOptions, Progress, QueryOptions};

let mut ds = DeepSearch::open_or_empty(None)?;
ds.build("/home/juan".as_ref(), &IndexOptions::default(), &Progress::default());
ds.save(None)?;
for hit in ds.search("inverted index", &QueryOptions::default()) {
    println!("{:.3}  {}", hit.score, hit.path.display());
}
```

---

## Design decisions (and why)

### Two-crate workspace, engine separated from UI
A required constraint, and a good one: the engine is testable and reusable
without a terminal. The binary is a thin shell over the library's public API.

### Content-based file typing, not by extension
Type detection reads magic bytes ([`infer`]) and, failing that, inspects the
byte stream ([`content_inspector`]) to classify text vs binary. A `.txt` full of
NUL bytes is treated as binary; a code file with no extension is still indexed.
Extraction is per-type: plain text/code read directly, **PDF** via `pdf-extract`,
**DOCX** by unzipping and pulling `<w:t>` runs out of `word/document.xml`,
everything else indexed by **metadata only**.

### Inverted index with two fields (content + name)
`term → [(doc_id, term_freq)]`, kept as **two** separate dictionaries — one over
contents, one over filenames. Keeping them apart is what lets the ranker boost a
name match independently of a body match (configurable `name_boost`, default
3×). Per-document metadata (path, size, mtime, token lengths) is held resident
for ranking and display.

### BM25, implemented by hand
No search-engine crate. The classic Okapi BM25 with `k1 = 1.2`, `b = 0.75`:

```
idf(t)     = ln(1 + (N − df + 0.5) / (df + 0.5))
score(t,d) = idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
```

Document-frequency `df` is computed from **live** postings only (see tombstones
below), so incremental churn never skews `idf`. Name-field scores are multiplied
by `name_boost` and added to content scores.

### Tokenizer: normalize, then stem — but fuzzy uses the *un*stemmed form
The pipeline splits on non-alphanumerics, splits `camelCase`/`snake_case` and
letter↔digit boundaries (`getUserName → get user name`, `HTTPServer → http
server`), lowercases, then applies English Snowball stemming. The **same**
pipeline runs at index and query time so terms always align.

Fuzzy filename matching deliberately runs on the **unstemmed** tokens. Stemming a
*misspelled* word is unpredictable (`deployment → deploy`, but `deployemnt →
deployemnt`), which would inflate the edit distance and defeat the whole point.
So the index keeps a second, unstemmed filename dictionary purely for fuzzy
lookups.

### Fuzzy matching: Damerau-Levenshtein, bounded
Typos are matched against real filename words with **Damerau-Levenshtein**
distance (default ≤ 1), so a single adjacent transposition — the most common typo
— counts as one edit (`gorceries → groceries`). A naive scan of a 100k-term
dictionary would blow the latency budget, so two cheap prefilters run before the
edit-distance: candidate length within the budget, and **matching first
character**. The first-char filter alone prunes an entire numeric filename
dictionary against a word query and keeps the scan bounded.
*Trade-off:* a typo in the very first character won't fuzzy-match. In practice
that is rare, and it is what buys the latency guarantee.

### Incremental indexing via tombstones + compaction
Re-indexing uses `mtime` to skip unchanged files. When a file changes or is
deleted, its old `doc_id` is **tombstoned** (marked dead) rather than surgically
removed from every postings list — surgical removal is O(dictionary) per file.
Dead docs are skipped at query time and excluded from BM25 stats, so results stay
correct. When the tombstone ratio crosses 30 %, a `compact()` pass renumbers ids
and drops the dead weight. This keeps the *common* case (a few changed files)
cheap while bounding worst-case bloat.

### Streaming, parallel indexing (bounded memory)
Walking uses [`ignore`], so `.gitignore`/`.ignore`, hidden files, `node_modules`,
`.git`, etc. are respected for free. Files are tokenized in **parallel** with
`rayon`, but in **batches** of 512: each batch is tokenized, merged into the
index, and dropped before the next. Only a batch's worth of tokenized documents
is ever resident, so a tree of 100k+ files does not balloon memory.

### Persistence: versioned bincode, atomic write
The index serializes to `~/.cache/deepsearch/index.bin` with `bincode`, wrapped
in a small envelope carrying a **format version** (a stale cache is rejected, not
misread). Writes go to a temp file and are `rename`d into place, so a crash mid-save
never corrupts the existing index.

### Previews never block the UI
The TUI runs a dedicated **preview worker thread**. Each request carries a
generation number; the worker coalesces to the newest pending request and the UI
applies a reply only if its generation still matches the current selection —
stale work is discarded. Text/code is highlighted with `syntect` and query
matches are overlaid (reversed+bold); PDFs/DOCX show extracted text; binaries
show metadata. **Image decoding also happens on the worker** (decoded and
downscaled there); the UI thread only wraps the result in the terminal-graphics
widget.

### Images: native protocols, no system dependency
`ratatui-image` renders via Kitty / Sixel / iTerm2 when the terminal supports
them, falling back to **Unicode half-blocks** everywhere else. Its default
`chafa` backend needs the `libchafa` system library, so it is disabled — the
native protocols plus the half-block fallback cover every terminal with zero
system deps.

### Errors: one bad file can never take down the run
`anyhow` for the application, `thiserror` for the library's typed errors. Text
extraction — `pdf-extract` in particular, which can *panic* on malformed PDFs —
is isolated behind `catch_unwind` **per file**. A corrupt document is counted as
an error and skipped; the index build continues.

---

## Usage

```
deepsearch index [PATH]            Build the index (PATH defaults to $HOME).
deepsearch index [PATH] --incremental
                                   Reindex only changed files; drop deleted ones.
deepsearch query "<words>"         Ranked results (name + content).
        --limit N                  Cap results (default 20).
        --json                     Machine-readable output.
deepsearch tui [PATH]              Interactive UI (indexes PATH first if empty).
deepsearch stats                   Index size, term counts, tombstone ratio.
        --cache <FILE>             Use a non-default index location (global flag).
```

### TUI keys
| Key | Action |
|-----|--------|
| type | edit the query (incremental, debounced) |
| `↑`/`↓`, `Ctrl-n`/`Ctrl-p` | move selection |
| `Esc` | switch to Normal mode |
| `j`/`k`, `g`/`G` | move / jump (Normal mode) |
| `i` or `/` | back to Insert mode |
| `Enter` | open the file in `$EDITOR` |
| `Ctrl-U` | clear the query |
| `q` / `Esc` (Normal), `Ctrl-C` | quit |

Two modes exist so the vim keys (`j`/`k`) can coexist with typing letters into
the query box.

---

## Performance

Measured on this machine (release build):

| Workload | Result |
|----------|--------|
| Query latency, 100k docs, single term | **~0.06 ms** |
| Query latency, 100k docs, 4-term      | **~0.38 ms** |
| Query latency, 100k docs, fuzzy typo  | **~0.25 ms** |
| Query latency, 10k on-disk corpus (criterion) | ~3.8 ms single / ~8.8 ms multi |

Comfortably under the **50 ms @ 100k documents** target.

Reproduce:
```
cargo run --release --example scale -p deepsearch-core   # 100k in-memory latency
cargo bench                                              # criterion: indexing + query
```

---

## Testing

```
cargo test        # unit tests: tokenizer, index, BM25 scoring, extraction, incremental
cargo clippy --workspace --all-targets
```

Unit tests cover the tokenizer (camel/snake splitting, stemming, edge cases), the
inverted index (aggregates, tombstoning, compaction), BM25 scoring (idf
monotonicity, relevance ordering, name-boost, fuzzy tolerance, tombstone
exclusion), text extraction, incremental add/modify/delete, and the preview
match-overlay.

---

## Limitations & future work

- **Stemming is English-only.** Non-English content is still indexed and matched
  exactly, just not stemmed.
- **First-character typos** are not fuzzy-matched (a deliberate latency
  trade-off; see above).
- The index is loaded fully into RAM for querying. For *very* large corpora a
  memory-mapped, block-compressed postings format would be the next step.
- Fuzzy matching is filename-only, by design (typo tolerance is most useful for
  names; content is matched via stemming/BM25).
