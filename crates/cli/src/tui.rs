//! Interactive terminal UI.
//!
//! Layout: a query box on top, a ranked results list on the left, and a preview
//! pane on the right. Search is incremental with a short debounce; previews are
//! built on a worker thread (see [`crate::preview`]) so keystrokes never block.
//!
//! Two modes (vim-flavoured):
//!   * **Insert** (default): typing edits the query, with the caret movement and
//!     word-deletion keys a shell gives you (see [`crate::input`]). Up/Down move
//!     the selection, Enter opens the file, Esc drops to Normal mode.
//!   * **Normal**: `j`/`k` (and arrows) move, `g`/`G` jump, `i` or `/` returns to
//!     Insert, `q`/Esc quits, Enter opens the file.
//!
//! The screen is repainted only when something changed, so an open but idle
//! deepsearch costs nothing.

use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers,
};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, ListState, Padding, Paragraph, Wrap,
};
use ratatui::{DefaultTerminal, Frame};

use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::StatefulImage;

use deepsearch_core::{DeepSearch, QueryOptions, SearchResult};

use crate::input::LineInput;
use crate::open::{candidates_for, AppKind, OpenApp};
use crate::preview::{Preview, PreviewRequest, PreviewWorker};

/// The UI borrows the terminal's own palette instead of imposing one.
///
/// Colours are the **ANSI slots**, not fixed RGB: the terminal maps them to
/// whatever theme is active, so deepsearch matches the desktop (on Omarchy, the
/// theme that sets your wallpaper also rewrites the terminal palette) and stays
/// readable on light and dark backgrounds alike.
///
/// Secondary text is the default foreground with the *dim* attribute rather than
/// a grey: many themes make "bright black" nearly invisible against their
/// background, and a faded foreground degrades gracefully everywhere.
mod theme {
    use ratatui::style::{Color, Modifier, Style};

    /// Primary accent: focus, selection, headings.
    pub const ACCENT: Color = Color::Cyan;
    /// Secondary accent, used for the AI affordances.
    pub const ALT: Color = Color::Magenta;
    /// Positive state (semantic search active).
    pub const OK: Color = Color::Green;
    /// In-progress / attention.
    pub const WARN: Color = Color::Yellow;
    /// Errors.
    pub const ERR: Color = Color::Red;

    /// Body text: the terminal's own foreground.
    pub fn body() -> Style {
        Style::default().fg(Color::Reset)
    }

    /// Secondary text — paths, labels, hints, and unfocused panel outlines.
    pub fn dim() -> Style {
        Style::default().add_modifier(Modifier::DIM)
    }

    /// The selected row. No background fill: a coloured bar plus bold text reads
    /// clearly on any palette, whereas a fixed highlight colour does not.
    pub fn selected() -> Style {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    }
}

/// A rounded, dim-bordered panel with a little breathing room — the base for
/// every pane so they look like one system.
fn panel(title: Line<'_>, focused: bool) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(if focused {
            Style::default().fg(theme::ACCENT)
        } else {
            theme::dim()
        })
        .padding(Padding::horizontal(1))
        .title(title)
}

/// A dim `·` used to separate title badges and footer hints.
fn sep() -> Span<'static> {
    Span::styled(" · ", theme::dim())
}

/// Render a directory for display, collapsing the home directory to `~`. Keeps
/// the narrow results pane readable instead of burning half of it on `/home/you`.
fn pretty_dir(path: &std::path::Path) -> String {
    let s = path.to_string_lossy();
    if let Some(home) = dirs::home_dir() {
        let home = home.to_string_lossy();
        if !home.is_empty() {
            if let Some(rest) = s.strip_prefix(home.as_ref()) {
                return format!("~{rest}");
            }
        }
    }
    s.into_owned()
}

/// Enter the alternate screen, with bracketed paste turned on.
///
/// Pasting into a search box is an everyday move — a path from another window, a phrase from a
/// document. Without bracketed paste the terminal delivers it as a burst of individual keystrokes,
/// which the debounce then turns into a stutter of half-typed searches.
fn init_terminal() -> DefaultTerminal {
    let terminal = ratatui::init();
    let _ = crossterm::execute!(std::io::stdout(), EnableBracketedPaste);
    terminal
}

/// Hand the terminal back exactly as it was found.
fn restore_terminal() {
    let _ = crossterm::execute!(std::io::stdout(), DisableBracketedPaste);
    ratatui::restore();
}

/// How long the query must be idle before we run the search.
const DEBOUNCE: Duration = Duration::from_millis(120);
/// Event-poll tick; also the max UI latency for applying a finished preview.
const TICK: Duration = Duration::from_millis(40);

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Insert,
    Normal,
}

enum Action {
    None,
    Quit,
    SmartOpen,
    OpenWith(OpenApp),
}

/// The "open with" popup: a list of installed apps for the selected file.
struct OpenMenu {
    apps: Vec<OpenApp>,
    state: ListState,
}

impl OpenMenu {
    fn move_selection(&mut self, delta: isize) {
        if self.apps.is_empty() {
            return;
        }
        let cur = self.state.selected().unwrap_or(0) as isize;
        let max = self.apps.len() as isize - 1;
        self.state
            .select(Some((cur + delta).clamp(0, max) as usize));
    }

    fn selected(&self) -> Option<&OpenApp> {
        self.state.selected().and_then(|i| self.apps.get(i))
    }
}

pub struct App {
    ds: DeepSearch,
    opts: QueryOptions,

    mode: Mode,
    input: LineInput,
    results: Vec<SearchResult>,
    list_state: ListState,
    /// Rows visible in the results pane, captured while rendering so PageUp and PageDown move by
    /// a screenful of *this* terminal rather than a hardcoded guess.
    list_height: u16,

    worker: PreviewWorker,
    generation: u64,
    preview: Preview,
    showing_image: bool,
    image_state: Option<StatefulProtocol>,
    picker: Option<Picker>,
    picker_tried: bool,

    open_menu: Option<OpenMenu>,
    show_help: bool,

    /// In-flight natural-language translation (if any). `Some` means an "ask AI"
    /// request is running on a background thread; the result arrives here.
    ai_rx: Option<Receiver<Result<String, String>>>,
    /// Whether a local Ollama server was detected (probed once at startup). When
    /// true, the UI advertises the `Ctrl-a` "ask AI" shortcut.
    ai_available: bool,
    /// Receives the one-shot startup probe of Ollama availability.
    ai_probe: Option<Receiver<bool>>,

    /// Whether the index carries semantic embeddings (built with `--semantic`).
    has_embeddings: bool,
    /// In-flight query embedding for semantic re-ranking, tagged with the
    /// generation it belongs to so stale replies are dropped.
    embed_rx: Option<Receiver<(u64, Vec<f32>)>>,
    embed_gen: u64,

    dirty: bool,
    last_edit: Instant,
    status: String,
}

impl App {
    pub fn new(ds: DeepSearch, opts: QueryOptions) -> Self {
        let n = ds.len();
        let has_embeddings = ds.has_embeddings();
        // Probe Ollama once, off-thread, so startup never blocks on it. The
        // result lights up the "ask AI" hint if a local server is reachable.
        let (probe_tx, probe_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = probe_tx.send(crate::ai::available());
        });
        App {
            ds,
            opts,
            mode: Mode::Insert,
            input: LineInput::new(),
            results: Vec::new(),
            list_state: ListState::default(),
            list_height: 10,
            worker: PreviewWorker::spawn(),
            generation: 0,
            preview: Preview::Loading,
            showing_image: false,
            image_state: None,
            picker: None,
            picker_tried: false,
            open_menu: None,
            show_help: false,
            ai_rx: None,
            ai_available: false,
            ai_probe: Some(probe_rx),
            has_embeddings,
            embed_rx: None,
            embed_gen: 0,
            dirty: false,
            last_edit: Instant::now(),
            status: format!("{n} documents indexed — start typing to search"),
        }
    }

    /// Run the UI to completion.
    pub fn run(mut self) -> Result<()> {
        let mut terminal = init_terminal();
        let res = self.event_loop(&mut terminal);
        restore_terminal();
        res
    }

    /// The main loop.
    ///
    /// Drawing happens only when something actually changed. Polling for input is a blocking wait
    /// that costs nothing, but redrawing on every tick means waking the CPU 25 times a second to
    /// paint an identical screen — for a tool that sits open in a terminal all day, that is real
    /// battery for no benefit. Every branch that changes what is on screen says so explicitly,
    /// including terminal resizes, which the old unconditional redraw was quietly covering up.
    fn event_loop(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        let mut needs_redraw = true;
        loop {
            if needs_redraw {
                terminal.draw(|f| self.render(f))?;
                needs_redraw = false;
            }

            needs_redraw |= self.drain_previews();
            needs_redraw |= self.drain_ai();

            if event::poll(TICK)? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        needs_redraw = true;
                        match self.handle_key(key.code, key.modifiers) {
                            Action::Quit => break,
                            Action::SmartOpen => self.smart_open(terminal)?,
                            Action::OpenWith(app) => self.launch_app(terminal, app)?,
                            Action::None => {}
                        }
                    }
                    Event::Paste(text) => {
                        needs_redraw = true;
                        self.paste(&text);
                    }
                    Event::Resize(..) => needs_redraw = true,
                    _ => {}
                }
            }

            if self.dirty && self.last_edit.elapsed() >= DEBOUNCE {
                self.run_search();
                self.dirty = false;
                needs_redraw = true;
            }
        }
        Ok(())
    }

    // --- input handling ---------------------------------------------------

    fn handle_key(&mut self, code: KeyCode, mods: KeyModifiers) -> Action {
        // Ctrl-c comes first and is never swallowed. An overlay that can trap the one key every
        // terminal user reaches for to get out is a trap, not a dialog.
        if mods.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
            return Action::Quit;
        }

        // Overlays capture the remaining keys while they are up.
        if self.show_help {
            // Any key dismisses the help overlay.
            self.show_help = false;
            return Action::None;
        }
        if self.open_menu.is_some() {
            return self.handle_menu_key(code);
        }

        // F1 opens help from any mode (no need to leave Insert first).
        if code == KeyCode::F(1) {
            self.show_help = true;
            return Action::None;
        }

        // Global bindings first.
        if mods.contains(KeyModifiers::CONTROL) {
            match code {
                KeyCode::Char('n') => {
                    self.move_selection(1);
                    return Action::None;
                }
                KeyCode::Char('p') => {
                    self.move_selection(-1);
                    return Action::None;
                }
                KeyCode::Char('o') => {
                    self.toggle_open_menu();
                    return Action::None;
                }
                KeyCode::Char('y') => {
                    self.copy_path();
                    return Action::None;
                }
                KeyCode::Char('a') => {
                    self.ask_ai();
                    return Action::None;
                }
                // Readline habits, available from either mode.
                KeyCode::Char('u') => return self.edit(|input| input.clear()),
                KeyCode::Char('w') => return self.edit(LineInput::delete_word_back),
                KeyCode::Char('k') if self.mode == Mode::Insert => {
                    return self.edit(LineInput::kill_to_end);
                }
                KeyCode::Char('h') if self.mode == Mode::Insert => {
                    return self.edit(LineInput::backspace);
                }
                _ => {}
            }
        }

        // Alt-Backspace also deletes a word, which is what most terminals send.
        if mods.contains(KeyModifiers::ALT) && code == KeyCode::Backspace {
            return self.edit(LineInput::delete_word_back);
        }

        let page = isize::from(self.list_height.max(1) as i16);
        match code {
            KeyCode::Down => self.move_selection(1),
            KeyCode::Up => self.move_selection(-1),
            KeyCode::PageDown => self.move_selection(page),
            KeyCode::PageUp => self.move_selection(-page),
            KeyCode::Enter => {
                if self.selected().is_some() {
                    return Action::SmartOpen;
                }
            }
            _ => match self.mode {
                Mode::Insert => return self.handle_insert(code),
                Mode::Normal => return self.handle_normal(code),
            },
        }
        Action::None
    }

    /// Insert pasted text at the caret.
    ///
    /// Newlines and tabs collapse to single spaces: a query is one line, and a multi-line paste
    /// should search for those words rather than mangle the box.
    fn paste(&mut self, text: &str) {
        let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if self.mode == Mode::Normal {
            self.mode = Mode::Insert;
        }
        self.edit(|input| input.insert_str(&flattened));
    }

    /// Apply an edit to the query, re-running the search only if the text actually changed.
    ///
    /// Moving the caret must not restart the search: it would throw away results and make the
    /// list flicker for a keystroke that asked for nothing new.
    fn edit(&mut self, op: impl FnOnce(&mut LineInput) -> bool) -> Action {
        if op(&mut self.input) {
            self.mark_dirty();
        }
        Action::None
    }

    fn handle_insert(&mut self, code: KeyCode) -> Action {
        match code {
            KeyCode::Char(c) => return self.edit(|input| input.insert(c)),
            KeyCode::Backspace => return self.edit(LineInput::backspace),
            KeyCode::Delete => return self.edit(LineInput::delete),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Esc => self.mode = Mode::Normal,
            _ => {}
        }
        Action::None
    }

    fn handle_normal(&mut self, code: KeyCode) -> Action {
        match code {
            KeyCode::Char('j') => self.move_selection(1),
            KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::Char('g') => self.select(0),
            KeyCode::Char('G') => self.select(self.results.len().saturating_sub(1)),
            KeyCode::Char('i') | KeyCode::Char('/') => self.mode = Mode::Insert,
            KeyCode::Char('o') => self.toggle_open_menu(),
            KeyCode::Char('y') => self.copy_path(),
            KeyCode::Char('?') => self.show_help = true,
            KeyCode::Char('q') | KeyCode::Esc => return Action::Quit,
            _ => {}
        }
        Action::None
    }

    /// Keys while the "open with" popup is up.
    fn handle_menu_key(&mut self, code: KeyCode) -> Action {
        let Some(menu) = self.open_menu.as_mut() else {
            return Action::None;
        };
        match code {
            KeyCode::Esc | KeyCode::Char('q') => self.open_menu = None,
            KeyCode::Down | KeyCode::Char('j') => menu.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => menu.move_selection(-1),
            KeyCode::Enter => {
                let chosen = menu.selected().cloned();
                self.open_menu = None;
                if let Some(app) = chosen {
                    return Action::OpenWith(app);
                }
            }
            // Number keys jump straight to that entry and launch it.
            KeyCode::Char(c @ '1'..='9') => {
                let idx = c as usize - '1' as usize;
                if let Some(app) = menu.apps.get(idx).cloned() {
                    self.open_menu = None;
                    return Action::OpenWith(app);
                }
            }
            _ => {}
        }
        Action::None
    }

    /// Open (or close) the "open with" popup for the current selection.
    fn toggle_open_menu(&mut self) {
        if self.open_menu.is_some() {
            self.open_menu = None;
            return;
        }
        let Some((path, file_type)) = self.selected().map(|r| (r.path.clone(), r.file_type)) else {
            self.status = "nothing selected to open".to_string();
            return;
        };
        let apps = candidates_for(&path, file_type);
        if apps.is_empty() {
            self.status = "no applications found to open this file".to_string();
            return;
        }
        let mut state = ListState::default();
        state.select(Some(0));
        self.open_menu = Some(OpenMenu { apps, state });
    }

    /// Copy the selected file's full path to the system clipboard.
    fn copy_path(&mut self) {
        let Some(path) = self.selected().map(|r| r.path.display().to_string()) else {
            self.status = "nothing selected to copy".to_string();
            return;
        };
        self.status = match crate::clip::copy(&path) {
            Ok(tool) => format!("copied path to clipboard ({tool})"),
            Err(e) => format!("copy failed: {e}"),
        };
    }

    /// Send the current query to a local Ollama model to be rewritten as a
    /// deepsearch query. Runs on a background thread so the UI never blocks; the
    /// reply is picked up by [`Self::drain_ai`].
    fn ask_ai(&mut self) {
        if self.ai_rx.is_some() {
            return; // a request is already in flight
        }
        let request = self.input.text().trim().to_string();
        if request.is_empty() {
            self.status = "type what you're looking for, then Ctrl-a".to_string();
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.ai_rx = Some(rx);
        self.status = "asking AI…".to_string();
        std::thread::spawn(move || {
            let _ = tx.send(crate::ai::translate_query(&request));
        });
    }

    /// Apply a finished AI translation: replace the query and search, or report
    /// the error.
    ///
    /// Returns whether anything on screen changed, so the loop knows to repaint.
    fn drain_ai(&mut self) -> bool {
        let mut changed = false;

        // Pick up the one-shot Ollama availability probe.
        if let Some(probe) = self.ai_probe.as_ref() {
            if let Ok(available) = probe.try_recv() {
                self.ai_available = available;
                self.ai_probe = None;
                changed = true;
            }
        }

        // Pick up a finished query embedding and re-rank with semantics.
        if let Some(rx) = self.embed_rx.as_ref() {
            if let Ok((gen, vec)) = rx.try_recv() {
                self.embed_rx = None;
                changed = true;
                if gen == self.embed_gen && !self.input.text().trim().is_empty() {
                    self.apply_semantic(&vec);
                }
            }
        }

        let Some(rx) = self.ai_rx.as_ref() else {
            return changed;
        };
        match rx.try_recv() {
            Ok(Ok(query)) => {
                self.ai_rx = None;
                self.input.set(query);
                self.status = format!("AI → {}", self.input.text());
                self.mark_dirty();
            }
            Ok(Err(e)) => {
                self.ai_rx = None;
                self.status = format!("AI: {e}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => return changed, // still working
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.ai_rx = None;
                self.status = "AI request failed".to_string();
            }
        }
        true
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
        self.last_edit = Instant::now();
    }

    // --- search & selection ----------------------------------------------

    fn run_search(&mut self) {
        if self.input.text().trim().is_empty() {
            self.results.clear();
            self.list_state.select(None);
            self.preview = Preview::Loading;
            self.showing_image = false;
            self.status = format!("{} documents indexed", self.ds.len());
            return;
        }
        let start = Instant::now();
        self.results = self.ds.search(self.input.text(), &self.opts);
        let elapsed = start.elapsed();
        self.status = format!(
            "{} results in {:.1} ms",
            self.results.len(),
            elapsed.as_secs_f64() * 1000.0
        );
        if self.results.is_empty() {
            self.list_state.select(None);
            self.preview = Preview::Loading;
            self.showing_image = false;
        } else {
            self.list_state.select(Some(0));
            self.request_preview();
        }
        // Kick off semantic re-ranking in the background; keyword results are
        // already on screen and get refined when the embedding arrives.
        self.request_semantic();
    }

    /// If the index has embeddings and Ollama is reachable, embed the current
    /// query off-thread. The reply (tagged with a generation) is applied by
    /// [`Self::drain_ai`] via [`Self::hybrid_search`], replacing the keyword
    /// results with meaning-aware ones.
    fn request_semantic(&mut self) {
        if !self.has_embeddings || !self.ai_available {
            return;
        }
        let query = self.input.text().trim().to_string();
        if query.is_empty() {
            return;
        }
        self.embed_gen += 1;
        let gen = self.embed_gen;
        let (tx, rx) = std::sync::mpsc::channel();
        self.embed_rx = Some(rx);
        std::thread::spawn(move || {
            if let Ok(vec) = crate::ai::embed(&query, true) {
                let _ = tx.send((gen, vec));
            }
        });
    }

    /// Re-rank the current query with hybrid keyword+semantic scoring using the
    /// freshly computed query embedding, keeping the selection on the same file
    /// when possible.
    fn apply_semantic(&mut self, query_vec: &[f32]) {
        let keep = self.selected().map(|r| r.doc_id);
        self.results = self.ds.hybrid_search(
            self.input.text(),
            query_vec,
            &self.opts,
            crate::SEMANTIC_WEIGHT,
        );
        if self.results.is_empty() {
            self.list_state.select(None);
            self.preview = Preview::Loading;
            self.showing_image = false;
            return;
        }
        // Try to keep the same file selected; otherwise jump to the top.
        let idx = keep
            .and_then(|id| self.results.iter().position(|r| r.doc_id == id))
            .unwrap_or(0);
        self.list_state.select(Some(idx));
        self.status = format!("{} results · semantic", self.results.len());
        self.request_preview();
    }

    fn selected(&self) -> Option<&SearchResult> {
        self.list_state.selected().and_then(|i| self.results.get(i))
    }

    fn move_selection(&mut self, delta: isize) {
        if self.results.is_empty() {
            return;
        }
        let cur = self.list_state.selected().unwrap_or(0) as isize;
        let max = self.results.len() as isize - 1;
        let next = (cur + delta).clamp(0, max) as usize;
        self.select(next);
    }

    fn select(&mut self, idx: usize) {
        if self.results.is_empty() {
            return;
        }
        let idx = idx.min(self.results.len() - 1);
        if self.list_state.selected() != Some(idx) {
            self.list_state.select(Some(idx));
            self.request_preview();
        }
    }

    /// Query terms for match highlighting: raw, lowercased, length >= 2.
    fn highlight_terms(&self) -> Vec<String> {
        self.input
            .text()
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.chars().count() >= 2)
            .collect()
    }

    fn request_preview(&mut self) {
        let Some(res) = self.selected() else { return };
        let (path, file_type, size, mtime) = (res.path.clone(), res.file_type, res.size, res.mtime);
        let terms = self.highlight_terms();
        self.generation += 1;
        self.preview = Preview::Loading;
        self.worker.request(PreviewRequest {
            generation: self.generation,
            path,
            file_type,
            size,
            mtime,
            terms,
        });
    }

    /// Apply any preview replies whose generation is still current.
    ///
    /// Returns whether anything on screen changed, so the loop knows to repaint.
    fn drain_previews(&mut self) -> bool {
        let mut changed = false;
        while let Ok((gen, preview)) = self.worker.rx.try_recv() {
            if gen != self.generation {
                continue; // stale
            }
            changed = true;
            match preview {
                Preview::Image(img) => match self.ensure_picker() {
                    Some(picker) => {
                        self.image_state = Some(picker.new_resize_protocol(*img));
                        self.showing_image = true;
                    }
                    None => {
                        self.showing_image = false;
                        self.preview = Preview::Error("terminal cannot render images".to_string());
                    }
                },
                other => {
                    self.showing_image = false;
                    self.image_state = None;
                    self.preview = other;
                }
            }
        }
        changed
    }

    /// Lazily initialize the image picker, probing the terminal for its
    /// graphics protocol; falls back to a fixed font size (Unicode blocks).
    fn ensure_picker(&mut self) -> Option<&mut Picker> {
        if !self.picker_tried {
            self.picker_tried = true;
            // Probe the terminal for a graphics protocol (Kitty/Sixel/iTerm2);
            // if that fails, fall back to Unicode half-blocks which work
            // anywhere.
            self.picker = Some(Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks()));
        }
        self.picker.as_mut()
    }

    fn open_editor(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        let Some(path) = self.selected().map(|r| r.path.clone()) else {
            return Ok(());
        };
        let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());

        restore_terminal();
        let status = std::process::Command::new(&editor).arg(&path).status();
        *terminal = init_terminal();
        terminal.clear()?;

        if let Err(e) = status {
            self.status = format!("failed to launch {editor}: {e}");
        }
        Ok(())
    }

    /// Enter on a result: open it in the *right* app for its type — text/code in
    /// `$EDITOR`, but images, PDFs, video and Office docs in a real viewer
    /// instead of opening them as garbled text in the editor.
    fn smart_open(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        let Some((path, file_type)) = self.selected().map(|r| (r.path.clone(), r.file_type)) else {
            return Ok(());
        };
        // Text and source belong in the editor.
        if file_type.is_textual() {
            return self.open_editor(terminal);
        }
        // Everything else: prefer a type-specific viewer / the OS default over a
        // generic code editor; fall back to the editor only if nothing else is
        // available.
        let apps = candidates_for(&path, file_type);
        let choice = apps
            .iter()
            .find(|a| {
                !matches!(
                    a.kind,
                    AppKind::Editor | AppKind::Reveal | AppKind::Terminal
                )
            })
            .or_else(|| apps.first())
            .cloned();
        match choice {
            Some(app) => self.launch_app(terminal, app),
            None => self.open_editor(terminal),
        }
    }

    /// Launch a chosen [`OpenApp`]. The command (program + args, path included)
    /// is ready to run. Terminal apps suspend the TUI for the duration; GUI apps
    /// are spawned detached so the UI keeps running.
    fn launch_app(&mut self, terminal: &mut DefaultTerminal, app: OpenApp) -> Result<()> {
        if app.terminal {
            restore_terminal();
            let status = std::process::Command::new(&app.program)
                .args(&app.args)
                .status();
            *terminal = init_terminal();
            terminal.clear()?;
            self.status = match status {
                Ok(_) => format!("opened in {}", app.label),
                Err(e) => format!("failed to launch {}: {e}", app.program),
            };
        } else {
            // Detach: silence the child's stdio so a chatty GUI launcher can't
            // scribble over the terminal, and don't wait on it.
            use std::process::Stdio;
            let spawned = std::process::Command::new(&app.program)
                .args(&app.args)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn();
            self.status = match spawned {
                Ok(_) => format!("opened in {}", app.label),
                Err(e) => format!("failed to launch {}: {e}", app.program),
            };
        }
        Ok(())
    }

    // --- rendering --------------------------------------------------------

    fn render(&mut self, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // query
                Constraint::Min(0),    // body
                Constraint::Length(1), // status
            ])
            .split(frame.area());

        self.render_query(frame, chunks[0]);

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(chunks[1]);

        self.render_results(frame, body[0]);
        self.render_preview(frame, body[1]);
        self.render_status(frame, chunks[2]);

        // Overlays draw on top of everything else.
        if self.open_menu.is_some() {
            self.render_open_menu(frame);
        }
        if self.show_help {
            render_help(frame);
        }
    }

    fn render_open_menu(&mut self, frame: &mut Frame) {
        // Compute the title before borrowing the menu mutably for its state.
        let filename = self
            .selected()
            .and_then(|r| r.path.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        let Some(menu) = self.open_menu.as_mut() else {
            return;
        };

        // A clean, numbered list: press the number to launch, or arrow + Enter.
        let items: Vec<ListItem> = menu
            .apps
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let num = if i < 9 {
                    format!(" {} ", i + 1)
                } else {
                    "   ".to_string()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(num, theme::dim()),
                    Span::raw(" "),
                    Span::styled(a.label.clone(), Style::default().fg(kind_color(a.kind))),
                ]))
            })
            .collect();

        // Size to content (each app is one row) plus borders, clamped to screen.
        let rows = menu.apps.len() as u16 + 2;
        let width = 52u16;
        let area = centered_rect(
            width,
            rows.min(frame.area().height.saturating_sub(2)),
            frame.area(),
        );

        let block = panel(
            Line::from(vec![
                Span::styled(
                    " Open ",
                    Style::default()
                        .fg(theme::ACCENT)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(filename, theme::body()),
                Span::raw(" "),
            ]),
            true,
        )
        .title_bottom(Line::from(Span::styled(
            " 1-9 open · ↑↓ move · ⏎ · esc ",
            theme::dim(),
        )));

        let list = List::new(items)
            .block(block)
            .highlight_style(theme::selected())
            .highlight_symbol("▌");

        frame.render_widget(Clear, area);
        frame.render_stateful_widget(list, area, &mut menu.state);
    }

    fn render_query(&self, frame: &mut Frame, area: Rect) {
        // Badges are dim text with a coloured marker rather than filled blocks:
        // they inform without shouting over the query itself.
        let mut title = vec![Span::styled(
            " deepsearch",
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )];

        let (mode_label, mode_color) = match self.mode {
            Mode::Insert => ("INSERT", theme::OK),
            Mode::Normal => ("NORMAL", theme::ACCENT),
        };
        title.push(sep());
        title.push(Span::styled(mode_label, Style::default().fg(mode_color)));

        if self.has_embeddings && self.ai_available {
            title.push(sep());
            title.push(Span::styled("● semantic", Style::default().fg(theme::OK)));
        }
        if self.ai_rx.is_some() {
            title.push(sep());
            title.push(Span::styled("◌ asking…", Style::default().fg(theme::WARN)));
        } else if self.ai_available {
            title.push(sep());
            title.push(Span::styled("⌃A ask AI", Style::default().fg(theme::ALT)));
        }
        title.push(Span::raw(" "));

        let focused = self.mode == Mode::Insert;
        let block = panel(Line::from(title), focused);
        // Room left for the query itself, once the borders, the padding and the "❯ " prompt are
        // paid for. The input scrolls inside whatever is left rather than running off the edge.
        let inner_width = block.inner(area).width;
        let prompt_width = 2u16;
        let room = inner_width.saturating_sub(prompt_width) as usize;
        let view = self.input.view(room);

        let mut spans = vec![Span::styled(
            "❯ ",
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )];
        if view.scrolled {
            // Tell the user the query continues off-screen to the left; without this a scrolled
            // box looks like it simply lost what they typed.
            spans.push(Span::styled("‹", theme::dim()));
            spans.push(Span::styled(
                view.text.chars().skip(1).collect::<String>(),
                theme::body(),
            ));
        } else if self.input.is_empty() {
            spans.push(Span::styled("search your files…", theme::dim()));
        } else {
            spans.push(Span::styled(view.text.clone(), theme::body()));
        }

        frame.render_widget(Paragraph::new(Line::from(spans)).block(block), area);

        if self.mode == Mode::Insert && area.width > 2 {
            // The caret sits where the view says it does, clamped inside the box so a tiny
            // terminal can never push it onto the border.
            let max_x = area.right().saturating_sub(2);
            let x = (area.x + 2 + prompt_width + view.cursor_col as u16).min(max_x);
            frame.set_cursor_position(Position::new(x, area.y + 1));
        }
    }

    /// What to show in the results pane when there is nothing to list.
    ///
    /// A pane that is merely blank makes the user wonder whether the tool is broken, still
    /// working, or simply found nothing. Say which.
    fn empty_results_message(&self) -> Vec<Line<'static>> {
        let hint = |text: &str| Line::from(Span::styled(text.to_string(), theme::dim()));
        if self.input.text().trim().is_empty() {
            vec![
                Line::from(Span::styled(
                    format!("{} documents indexed", self.ds.len()),
                    theme::body(),
                )),
                Line::from(""),
                hint("Start typing to search."),
                hint("Narrow it down with ext:rs or type:pdf."),
            ]
        } else {
            vec![
                Line::from(Span::styled("No matches.", theme::body())),
                Line::from(""),
                hint("Try fewer or more general words."),
                hint("Filters like ext:md and type:image also apply."),
                hint("Ctrl-u clears the query."),
            ]
        }
    }

    fn render_results(&mut self, frame: &mut Frame, area: Rect) {
        // The filename is what you scan for, so it leads; the kind and the
        // directory are context and stay dim. The score is noise — it's gone.
        let items: Vec<ListItem> = self
            .results
            .iter()
            .map(|r| {
                let name = r
                    .path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let parent = r.path.parent().map(pretty_dir).unwrap_or_default();
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<4}", type_tag(r.file_type)), theme::dim()),
                    Span::styled(name, theme::body()),
                    Span::styled(format!("  {parent}"), theme::dim()),
                ]))
            })
            .collect();

        let title = Line::from(vec![
            Span::styled(" Results ", theme::dim()),
            Span::styled(format!("{} ", self.results.len()), theme::dim()),
        ]);
        let focused = self.mode == Mode::Normal;
        let block = panel(title, focused);

        // Remember the viewport so PageUp/PageDown move by an actual screenful.
        self.list_height = block.inner(area).height;

        if self.results.is_empty() {
            let inner = block.inner(area);
            frame.render_widget(block, area);
            frame.render_widget(
                Paragraph::new(self.empty_results_message()).wrap(Wrap { trim: false }),
                inner,
            );
            return;
        }

        let list = List::new(items)
            .block(block)
            .highlight_style(theme::selected())
            .highlight_symbol("▌ ");
        frame.render_stateful_widget(list, area, &mut self.list_state);
    }

    fn render_preview(&mut self, frame: &mut Frame, area: Rect) {
        // Lead with the filename; the directory trails behind it, dimmed.
        let title = match self.selected() {
            Some(r) => {
                let name = r
                    .path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let parent = r.path.parent().map(pretty_dir).unwrap_or_default();
                Line::from(vec![
                    Span::styled(
                        format!(" {name} "),
                        Style::default()
                            .fg(theme::ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!("{parent} "), theme::dim()),
                ])
            }
            None => Line::from(Span::styled(" Preview ", theme::dim())),
        };
        let block = panel(title, false);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.showing_image {
            if let Some(state) = self.image_state.as_mut() {
                frame.render_stateful_widget(StatefulImage::default(), inner, state);
                return;
            }
        }

        // With nothing selected there is nothing to load, and a permanent "…" reads as a preview
        // that never arrives. Say what the pane is waiting for instead.
        if self.selected().is_none() {
            let message = if self.results.is_empty() && !self.input.text().trim().is_empty() {
                "Nothing to preview — no result matched."
            } else {
                "Select a result to preview it here."
            };
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(message, theme::dim()))),
                inner,
            );
            return;
        }

        let text: Text = match &self.preview {
            Preview::Text(t) | Preview::Meta(t) => t.clone(),
            Preview::Loading => Text::from(Line::from(Span::styled("loading…", theme::dim()))),
            Preview::Error(e) => Text::from(Line::from(Span::styled(
                e.clone(),
                Style::default().fg(theme::ERR),
            ))),
            // Image handled above; if we get here the picker failed.
            Preview::Image(_) => Text::from("image"),
        };
        frame.render_widget(Paragraph::new(text).wrap(Wrap { trim: false }), inner);
    }

    fn render_status(&self, frame: &mut Frame, area: Rect) {
        // Status on the left, key hints right-aligned: keys in the accent,
        // their labels dim, so the bar reads as guidance rather than chrome.
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(area);

        let status = Line::from(Span::styled(format!(" {}", self.status), theme::dim()));
        frame.render_widget(Paragraph::new(status), cols[0]);

        let key = Style::default().fg(theme::ACCENT);
        let label = theme::dim();
        let mut hints: Vec<Span> = Vec::new();
        for (i, (k, l)) in [
            ("↑↓", "move"),
            ("⏎", "open"),
            ("o", "open with"),
            ("F1", "help"),
        ]
        .iter()
        .enumerate()
        {
            if i > 0 {
                hints.push(sep());
            }
            hints.push(Span::styled(*k, key));
            hints.push(Span::raw(" "));
            hints.push(Span::styled(*l, label));
        }
        hints.push(Span::raw(" "));
        frame.render_widget(
            Paragraph::new(Line::from(hints)).alignment(Alignment::Right),
            cols[1],
        );
    }
}

/// Colour used for an open-with entry's label, grouping it by kind at a glance.
fn kind_color(kind: AppKind) -> Color {
    match kind {
        AppKind::Editor => theme::ACCENT,
        AppKind::Image => theme::OK,
        AppKind::Pdf => theme::ERR,
        AppKind::Media => theme::ALT,
        AppKind::Default => theme::WARN,
        AppKind::Reveal | AppKind::Terminal => Color::Reset,
    }
}

/// A centered overlay listing every keybinding.
fn render_help(frame: &mut Frame) {
    let rows: &[(&str, &str)] = &[
        ("type", "edit the query (filters as you type)"),
        ("ext:rs / type:pdf", "filter results by extension or type"),
        ("← / →  ·  Home / End", "move the caret in the query"),
        (
            "Ctrl-w  ·  Alt-Backspace",
            "delete the word behind the caret",
        ),
        (
            "Ctrl-k  ·  Delete",
            "delete to end of query · delete forwards",
        ),
        ("Ctrl-u", "clear the query"),
        ("↑ / ↓  ·  Ctrl-n / Ctrl-p", "move selection"),
        ("PageUp / PageDown", "move by one screenful"),
        ("Enter", "open in the right app for the file"),
        ("o  ·  Ctrl-o", "open-with menu (choose an app)"),
        ("Ctrl-a", "ask in plain language (local Ollama)"),
        ("y  ·  Ctrl-y", "copy the file path to the clipboard"),
        ("Esc", "Insert → Normal mode"),
        ("i  ·  /", "Normal → Insert mode"),
        ("j / k  ·  g / G", "move / jump (Normal mode)"),
        ("F1  ·  ? (Normal)", "show this help"),
        ("q  ·  Esc  ·  Ctrl-c", "quit"),
    ];

    let lines: Vec<Line> = rows
        .iter()
        .map(|(key, desc)| {
            Line::from(vec![
                Span::styled(format!("{key:<26}"), Style::default().fg(theme::ACCENT)),
                Span::styled(*desc, theme::dim()),
            ])
        })
        .collect();

    let height = rows.len() as u16 + 2;
    let area = centered_rect(66, height.min(frame.area().height), frame.area());
    let block = panel(
        Line::from(Span::styled(
            " Keys ",
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )),
        true,
    )
    .title_bottom(Line::from(Span::styled(" any key to close ", theme::dim())));

    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// A `Rect` of the given width/height centered inside `area` (clamped to fit).
fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    Rect {
        x,
        y,
        width,
        height,
    }
}

fn type_tag(t: deepsearch_core::FileType) -> &'static str {
    use deepsearch_core::FileType::*;
    match t {
        Text => "txt",
        Code => "code",
        Pdf => "pdf",
        Docx => "doc",
        Image => "img",
        Binary => "bin",
        Dir => "dir",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepsearch_core::index::{Index, PendingDoc};
    use deepsearch_core::FileType;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Term frequencies built through the real tokenizer, so query-time stemming
    /// lines up with what the fixture indexed.
    fn tf_of(text: &str) -> HashMap<String, u32> {
        let mut m = HashMap::new();
        for t in deepsearch_core::tokenize::tokenize(text) {
            *m.entry(t).or_insert(0) += 1;
        }
        m
    }

    /// Build a tiny in-memory index so the UI has something to draw.
    fn demo_app() -> App {
        let mut idx = Index::new();
        for (path, ft) in [
            ("/home/juan/notes/meeting-notes.md", FileType::Text),
            ("/home/juan/src/deepsearch/notes-parser.rs", FileType::Text),
            ("/home/juan/Pictures/notes-diagram.png", FileType::Image),
            ("/home/juan/docs/notes-archive.pdf", FileType::Pdf),
        ] {
            let p = PathBuf::from(path);
            let file_name = p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            idx.add(PendingDoc {
                path: p,
                size: 1024,
                mtime: 1,
                file_type: ft,
                content_tf: tf_of("meeting notes about the project"),
                name_tf: tf_of(&file_name),
                name_raw: deepsearch_core::tokenize::normalize(&file_name),
            });
        }
        let mut app = App::new(DeepSearch::new(idx), QueryOptions::default());
        app.input.set("notes");
        app.run_search();
        app
    }

    /// Feed a key to the app the way the event loop would.
    fn press(app: &mut App, code: KeyCode) {
        app.handle_key(code, KeyModifiers::NONE);
    }

    fn ctrl(app: &mut App, c: char) -> Action {
        app.handle_key(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn type_str(app: &mut App, text: &str) {
        for c in text.chars() {
            press(app, KeyCode::Char(c));
        }
    }

    /// Render the UI into an off-screen buffer and return it as plain text, so
    /// the layout can be inspected (and regressions caught) without a terminal.
    fn render_to_text(app: &mut App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn renders_main_layout() {
        let mut app = demo_app();
        let out = render_to_text(&mut app, 100, 18);
        println!("\n{out}\n");
        // Chrome, query and results are all present.
        assert!(out.contains("deepsearch"), "title bar");
        assert!(out.contains("INSERT"), "mode badge");
        assert!(out.contains("Results"), "results pane");
        assert!(out.contains("meeting-notes.md"), "a result row");
        assert!(out.contains("╭") && out.contains("╮"), "rounded corners");
    }

    #[test]
    fn renders_ai_badges_when_available() {
        let mut app = demo_app();
        // Pretend a local Ollama with embeddings was found.
        app.ai_available = true;
        app.has_embeddings = true;
        let out = render_to_text(&mut app, 100, 8);
        println!("\n{out}\n");
        assert!(out.contains("semantic"), "semantic badge");
        assert!(out.contains("ask AI"), "ask-AI badge");
    }

    #[test]
    fn uses_only_the_terminal_palette() {
        // deepsearch must not impose colours of its own: a hardcoded RGB value
        // ignores the user's theme (and, on Omarchy, their wallpaper). Every
        // cell should use an ANSI slot or the terminal default.
        let mut app = demo_app();
        app.ai_available = true;
        app.has_embeddings = true;

        let mut terminal = Terminal::new(TestBackend::new(100, 16)).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
        let buf = terminal.backend().buffer().clone();

        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                let cell = &buf[(x, y)];
                for (what, color) in [("fg", cell.fg), ("bg", cell.bg)] {
                    assert!(
                        !matches!(color, Color::Rgb(..) | Color::Indexed(..)),
                        "cell ({x},{y}) {what} is a fixed colour ({color:?}); \
                         use an ANSI slot so the terminal theme drives it",
                    );
                }
            }
        }
    }

    #[test]
    fn the_query_box_edits_at_the_caret_not_only_at_the_end() {
        let mut app = demo_app();
        app.input.set("meting notes");
        // Walk back to just after "me" and repair the typo in place.
        app.input.home();
        for _ in 0..2 {
            press(&mut app, KeyCode::Right);
        }
        type_str(&mut app, "e");
        assert_eq!(app.input.text(), "meeting notes");
    }

    #[test]
    fn ctrl_w_deletes_a_word_and_ctrl_u_clears_the_query() {
        let mut app = demo_app();
        app.input.set("annual report draft");
        ctrl(&mut app, 'w');
        assert_eq!(app.input.text(), "annual report ");
        ctrl(&mut app, 'u');
        assert_eq!(app.input.text(), "");
    }

    #[test]
    fn moving_the_caret_does_not_restart_the_search() {
        let mut app = demo_app();
        app.dirty = false;
        press(&mut app, KeyCode::Left);
        press(&mut app, KeyCode::Home);
        press(&mut app, KeyCode::End);
        assert!(
            !app.dirty,
            "caret movement changes no text, so it must not queue another search"
        );

        press(&mut app, KeyCode::Char('x'));
        assert!(app.dirty, "typing must queue a search");
    }

    #[test]
    fn deleting_nothing_does_not_queue_a_search() {
        let mut app = demo_app();
        app.input.clear();
        app.dirty = false;
        press(&mut app, KeyCode::Backspace);
        press(&mut app, KeyCode::Delete);
        ctrl(&mut app, 'w');
        assert!(!app.dirty, "no-op edits must not trigger work");
    }

    #[test]
    fn ctrl_c_quits_even_with_an_overlay_open() {
        let mut app = demo_app();
        app.show_help = true;
        assert!(
            matches!(ctrl(&mut app, 'c'), Action::Quit),
            "help must not trap Ctrl-c"
        );

        let mut app = demo_app();
        app.toggle_open_menu();
        assert!(
            matches!(ctrl(&mut app, 'c'), Action::Quit),
            "the menu must not trap Ctrl-c"
        );
    }

    #[test]
    fn page_keys_move_by_the_visible_height_not_a_fixed_guess() {
        let mut app = demo_app();
        // Rendering is what teaches the app how tall the list pane is.
        render_to_text(&mut app, 100, 40);
        let tall = app.list_height;
        render_to_text(&mut app, 100, 10);
        let short = app.list_height;
        assert!(
            tall > short,
            "the page size must follow the terminal ({tall} vs {short})"
        );
        assert!(
            short >= 1,
            "even a cramped pane must page by at least one row"
        );
    }

    #[test]
    fn a_long_query_scrolls_instead_of_running_off_the_box() {
        let mut app = demo_app();
        app.input.set("x".repeat(300));
        let out = render_to_text(&mut app, 60, 12);
        let query_row = out.lines().nth(1).unwrap_or_default();
        assert!(
            query_row.chars().count() <= 60,
            "the query must stay inside the terminal: {query_row:?}"
        );
        assert!(
            query_row.contains('\u{2039}'),
            "a scrolled query must show the ‹ marker"
        );
    }

    #[test]
    fn an_empty_result_set_explains_itself() {
        let mut app = demo_app();
        app.input.set("zzzznothingmatchesthis");
        app.run_search();
        let out = render_to_text(&mut app, 100, 16);
        assert!(app.results.is_empty(), "the fixture should match nothing");
        assert!(
            out.contains("No matches"),
            "the results pane must say so: {out}"
        );
        assert!(
            out.contains("ext:"),
            "and point at the filters that might help"
        );
        assert!(
            out.contains("Nothing to preview"),
            "the preview pane must not sit on a bare ellipsis"
        );
    }

    #[test]
    fn an_untouched_query_invites_the_user_in() {
        let mut app = demo_app();
        app.input.clear();
        app.run_search();
        let out = render_to_text(&mut app, 100, 16);
        assert!(
            out.contains("documents indexed"),
            "say what is searchable: {out}"
        );
        assert!(out.contains("Start typing"), "and what to do about it");
    }

    #[test]
    fn a_pasted_multi_line_selection_becomes_one_query() {
        let mut app = demo_app();
        app.input.clear();
        app.mode = Mode::Normal;
        app.paste("annual\n report\t draft\n");
        assert_eq!(app.input.text(), "annual report draft");
        assert!(
            app.mode == Mode::Insert,
            "pasting should put you back in the query"
        );
    }

    #[test]
    fn tiny_terminals_render_without_panicking() {
        // A pane can end up with zero usable width; every layout calculation has to survive it.
        for (w, h) in [(1u16, 1u16), (3, 2), (8, 4), (20, 6), (40, 10)] {
            let mut app = demo_app();
            app.show_help = true;
            render_to_text(&mut app, w, h);
        }
    }

    #[test]
    fn a_query_of_wide_characters_keeps_the_caret_inside_the_box() {
        // Each of these occupies two columns. Counting characters instead of cells would put the
        // caret roughly twice as far right as the text actually reaches — off the border, and in
        // a narrow terminal off the screen entirely.
        let mut app = demo_app();
        app.input.set("日本語のテキストをたくさん入力する");

        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();

        let caret = terminal.get_cursor_position().unwrap();
        assert_eq!(caret.y, 1, "the caret belongs on the query line");
        assert!(
            caret.x < 39,
            "the caret escaped the query box at x={}",
            caret.x
        );
    }

    #[test]
    fn renders_help_overlay() {
        let mut app = demo_app();
        app.show_help = true;
        let out = render_to_text(&mut app, 100, 22);
        println!("\n{out}\n");
        assert!(out.contains("Keys"), "help title");
        assert!(out.contains("any key to close"), "help footer");
    }
}
