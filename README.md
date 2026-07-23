# deepsearch

[![CI](https://github.com/Juan-Miguel-alvarado/Deepsearch/actions/workflows/ci.yml/badge.svg)](https://github.com/Juan-Miguel-alvarado/Deepsearch/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Juan-Miguel-alvarado/Deepsearch?sort=semver)](https://github.com/Juan-Miguel-alvarado/Deepsearch/releases/latest)

Relevance-ranked full-text search over **all** your files, from the terminal.

`deepsearch` is not `find` or `grep`. It scans your filesystem **once**, builds a
persistent inverted index, and then answers queries in **sub-millisecond** time,
ranked by **BM25 relevance** — searching both file *names* and file *contents*.
An interactive TUI gives you incremental search — matching prefixes as you type,
with `type:`/`ext:` filters — live, syntax-highlighted previews (and image
previews via the terminal graphics protocol), an **open-with** menu that
launches the selected file in whatever app you have installed (editor, image/PDF
viewer, media player, or the OS default), one-key **copy-path**, and optional
AI features via a local [Ollama](https://ollama.com) model (free, offline, no API
keys): **natural-language queries** and **semantic search** that finds files by
meaning, not just keywords.

```
deepsearch index ~/projects      # scan & index once (incremental afterwards)
deepsearch query "bm25 ranking"  # ranked results, name + content
deepsearch tui                   # interactive fuzzy search
```

---

<img width="1352" height="718" alt="image" src="https://github.com/user-attachments/assets/81edafd6-d82f-406a-98d0-77a58a76a175" />



https://github.com/user-attachments/assets/f68476ca-0f2e-4101-a1b6-f9b6e720894e



---

## Installation

Pick whichever fits you. After installing, run `deepsearch` from anywhere.

### 1. Download a prebuilt binary (no Rust needed)

Grab the archive for your platform from the
[**latest release**](https://github.com/Juan-Miguel-alvarado/Deepsearch/releases/latest),
extract it, and put the `deepsearch` binary on your `PATH`.

**Linux / macOS:**

```bash
# pick the file matching your platform from the releases page, e.g.:
#   deepsearch-x86_64-unknown-linux-gnu.tar.gz   (Linux, Intel/AMD)
#   deepsearch-aarch64-apple-darwin.tar.gz       (macOS, Apple Silicon)
#   deepsearch-x86_64-apple-darwin.tar.gz        (macOS, Intel)
tar -xzf deepsearch-*.tar.gz
sudo mv deepsearch /usr/local/bin/     # or: mv deepsearch ~/.local/bin/
deepsearch --version
```

**Windows:** download `deepsearch-x86_64-pc-windows-msvc.zip`, unzip it, and put
`deepsearch.exe` in a folder that's on your `PATH`.

Each archive ships a `.sha256` file so you can verify the download
(`shasum -a 256 -c deepsearch-*.tar.gz.sha256`).

### 2. Install with Cargo (from source, needs [Rust](https://rustup.rs))

```bash
cargo install --git https://github.com/Juan-Miguel-alvarado/Deepsearch deepsearch
```

This builds and drops `deepsearch` into `~/.cargo/bin` (already on your `PATH`
if you use rustup). Re-run the same command to update.

### 3. Build from a local clone

```bash
git clone https://github.com/Juan-Miguel-alvarado/Deepsearch
cd Deepsearch
cargo install --path crates/cli --force
```

---

## Quickstart

New here? Try it on a single folder in under a minute — no need to index your
whole home directory. The `--cache` flag keeps this trial index separate from
your real one:

```bash
# 1. build the project
cargo build --release

# 2. index just this repo into a throwaway cache
./target/release/deepsearch --cache /tmp/ds-demo.bin index .

# 3a. one-shot query from the shell
./target/release/deepsearch --cache /tmp/ds-demo.bin query "bm25" 

# 3b. or the interactive UI
./target/release/deepsearch --cache /tmp/ds-demo.bin tui
```

Things to try inside the TUI:

- **Type a partial word** (`conf`) — results filter as you type, before you
  finish `config`.
- **Filter** with `ext:rs parser` or `type:image` right in the query box.
- Press **`Enter`** to open a file in the right app for its type (text in your
  editor, an image in an image viewer, a PDF in a PDF reader…).
- Press **`o`** for the **open-with menu** and hit a number to launch instantly.
- Press **`y`** to **copy the file's path**, and **`F1`** for the key help.

For real use, just run `deepsearch` with no arguments: it indexes your home
directory the first time (incrementally after that), then opens the UI.

---

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
│   2. Query    — tokenize, BM25, prefix, fuzzy  │
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

### Prefix matching: filter as you type
An interactive search has to show useful results *before* you finish a word.
When a query token isn't an exact filename term, it is treated as a **strict
prefix** of the filename dictionary — so `conf` already surfaces `config.rs` and
`configuration.rs`. Prefix hits score below an exact match but above a fuzzy one
(`name_boost · prefix_penalty`, weighted by how much of the candidate word the
prefix covers), so completing the word only sharpens the ranking rather than
changing which files appear. The three name-field passes form a fallback chain —
**exact → prefix → fuzzy** — and each stops the next, so the common case never
pays for the approximate ones. Prefix matching runs on the same unstemmed
filename dictionary as fuzzy, for the same reason (a half-typed word has no
meaningful stem).

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
deepsearch index [PATH] --semantic Also build semantic embeddings (see below).
deepsearch query "<words>"         Ranked results (name + content).
        --limit N                  Cap results (default 20).
        --keyword                  Force keyword-only (skip semantic).
        --json                     Machine-readable output.
deepsearch ask "<plain text>"      Natural-language search (needs local Ollama).
deepsearch tui [PATH]              Interactive UI (indexes PATH first if empty).
deepsearch stats                   Index size, term counts, tombstone ratio.
        --cache <FILE>             Use a non-default index location (global flag).
```

### Ask questions about your files (optional, local & free)

With [Ollama](https://ollama.com) running locally, `ask` doesn't just find files
— it **answers**, from what's actually inside them, and cites the sources:

```bash
deepsearch ask "how are passwords handled in my projects?"
```
```
reading 3 file(s)…

The password is typed as a text input, but it can be switched to display the
password by clicking a button. The password itself is not stored anywhere in
this excerpt. [3]

Sources:
  [1] ~/development/flutter/docs/.../Gradle-for-Android.md
  [2] ~/Documents/proyects/.../ChangePasswordForm.tsx
  [3] ~/Documents/proyects/.../password-input.tsx
```

It's retrieval-augmented generation over your own index: the question finds the
most relevant documents (semantically when embeddings exist), deepsearch pulls
the **passage that bears on the question** out of each one — not the file's
imports — and a local model answers from those excerpts only, saying so plainly
when they don't contain the answer. Nothing leaves the machine.

`--json` emits `{ "answer": ..., "sources": [...] }` for scripting.

**Speed.** This is the one slow feature: on a CPU-only machine expect roughly a
minute per question (context length dominates, which is why only a few short
excerpts are sent). A smaller model is markedly faster:

```bash
ollama pull llama3.2:1b
export DEEPSEARCH_OLLAMA_MODEL=llama3.2:1b
```

It's entirely optional: with no Ollama installed, `ask` prints a friendly hint
and everything else works exactly the same. deepsearch picks a locally installed
model that can generate text (or set `DEEPSEARCH_OLLAMA_MODEL`); point at a
non-default server with `OLLAMA_HOST`.

### Semantic search (search by meaning)

Keyword search only finds the words you type. **Semantic search** finds files by
*meaning*, so a search for `login` surfaces a document about "authentication,
credentials and session tokens" even though it never contains the word "login".

Build it once (needs a local embedding model — free, offline):

```bash
ollama pull nomic-embed-text
deepsearch index --semantic          # embeds every document (a one-time pass)
```

After that it just works:

- `deepsearch query "login"` automatically blends keyword + semantic ranking
  (add `--keyword` to force the old behaviour).
- In the **TUI**, keyword results appear instantly and are re-ranked by meaning a
  moment later; a green **`semantic`** tag in the search box shows it's active.

How it works: each document is embedded into a vector with the local model; a
query is embedded the same way and scored by cosine similarity, then blended with
BM25 (`(1 − w)·keyword + w·semantic`, `w = 0.5`). Everything runs locally through
Ollama — nothing leaves the machine. Pick a different embedding model with
`DEEPSEARCH_OLLAMA_EMBED_MODEL`.

> **Note:** enabling embeddings bumps the on-disk index format. After upgrading,
> run `deepsearch index` again to rebuild the cache.

### Search filters

Any query — from the shell or in the TUI — can carry inline filters that narrow
results by file type or extension. They combine with search terms, and a filter
on its own browses the corpus (newest first):

| Filter | Matches |
|--------|---------|
| `type:image` (`img`) | images |
| `type:pdf` | PDFs |
| `type:code` / `type:text` | source / plain text |
| `type:docx` (`doc`) | Word documents |
| `type:binary` (`bin`) | other binaries |
| `ext:rs`, `ext:.md` | files with that extension (dot optional) |

```
deepsearch query "parser ext:rs"     # 'parser' in .rs files only
deepsearch query "invoice type:pdf"  # 'invoice' among PDFs
deepsearch query "type:image"        # every image, most recent first
```

### TUI keys
| Key | Action |
|-----|--------|
| type | edit the query (incremental, debounced) |
| `↑`/`↓`, `Ctrl-n`/`Ctrl-p` | move selection |
| `Esc` | switch to Normal mode |
| `j`/`k`, `g`/`G` | move / jump (Normal mode) |
| `i` or `/` | back to Insert mode |
| `Enter` | **open** the file in the right app for its type (see below) |
| `o` (Normal) / `Ctrl-o` | **open with…** — a clean, numbered menu of installed apps |
| `Ctrl-a` | **ask AI** — rewrite the query in plain language (needs local Ollama) |
| `y` (Normal) / `Ctrl-y` | **copy** the selected file's path to the clipboard |
| `F1` (any mode) / `?` (Normal) | show a **help overlay** listing every key |
| `Ctrl-U` | clear the query |
| `q` / `Esc` (Normal), `Ctrl-C` | quit |

Two modes exist so the vim keys (`j`/`k`) can coexist with typing letters into
the query box. Press `F1` any time (or `?` in Normal mode) for the full key list.

**Smart open (`Enter`).** Opening does the sensible thing for the file type:
text and source open in your `$EDITOR`, but an image opens in an image viewer, a
PDF in a PDF reader, a video in a media player, and a Word doc / other file in
the OS default handler — so you never get an image dumped as garbled text in the
editor again.

**Open-with menu (`o`).** A tidy popup that detects the applications actually
installed on your `PATH` and orders them by relevance to the selected file
(image viewers for an image, PDF readers for a PDF, …), then the OS default
handler, then **Reveal in folder** and **Terminal here**. Press the number next
to an entry to launch it instantly, or move with `↑`/`↓` and press `Enter`.
Terminal apps (vim, helix, …) suspend the UI while they run; GUI apps launch
detached so the search stays open.

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
monotonicity, relevance ordering, name-boost, prefix matching, fuzzy tolerance,
`type:`/`ext:` filter parsing, tombstone exclusion), text extraction,
incremental add/modify/delete, the open-with app detection, and the preview
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
