# Contributing to deepsearch

Thanks for taking a look! This is a small, dependency-light Rust workspace, so
getting set up takes one command.

## Prerequisites

- A recent stable Rust toolchain (`rustup` recommended). Install from
  <https://rustup.rs>.
- No system libraries are required. Image previews use the terminal's own
  graphics protocol; the optional `chafa` backend is deliberately disabled.

## Workspace layout

```
crates/core   deepsearch-core — indexing + search engine, no UI (the library)
crates/cli    deepsearch      — the CLI + interactive TUI
```

Start in `crates/core/src` for ranking/indexing changes (`query.rs`,
`index.rs`, `indexer.rs`, `tokenize.rs`, `extract.rs`) and `crates/cli/src` for
UI/UX (`tui.rs`, `preview.rs`, `open.rs`, `clip.rs`).

## Everyday commands

```bash
# Build (debug) / run
cargo build
cargo run -- tui                     # run the binary; args go after `--`

# Try changes on a throwaway index instead of your real one
cargo run -- --cache /tmp/ds-dev.bin index .
cargo run -- --cache /tmp/ds-dev.bin query "some words"

# Tests — unit tests live inline in each module
cargo test

# Lints & formatting — please run both before opening a PR
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

# Benchmarks (criterion) and the 100k-doc latency example
cargo bench
cargo run --release --example scale -p deepsearch-core
```

## Before you open a PR

1. `cargo test` passes.
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean.
3. `cargo fmt --all` has been run.
4. New behaviour has a test. The engine is well covered by unit tests in
   `crates/core`; match that style (`#[cfg(test)] mod tests` at the bottom of
   the file, `tempfile` for anything touching the filesystem).

## Releasing

Releases are automated. Bump `version` in the workspace `Cargo.toml`, commit,
then tag and push:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The `Release` workflow builds binaries for Linux, macOS (Intel + Apple Silicon)
and Windows and attaches them (with `.sha256` checksums) to a GitHub Release for
that tag. `CI` runs fmt + clippy + tests on every push and PR.

## Design notes

The README's **"Design decisions (and why)"** section explains the choices
behind the ranker, the tokenizer, incremental indexing, and the preview worker.
Skim it before a non-trivial change — most "why is it done this way?" questions
are answered there.
