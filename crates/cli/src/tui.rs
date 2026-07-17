//! Interactive terminal UI.
//!
//! Layout: a query box on top, a ranked results list on the left, and a preview
//! pane on the right. Search is incremental with a short debounce; previews are
//! built on a worker thread (see [`crate::preview`]) so keystrokes never block.
//!
//! Two modes (vim-flavoured):
//!   * **Insert** (default): typing edits the query. Arrows move the selection,
//!     Enter opens the file, Esc drops to Normal mode.
//!   * **Normal**: `j`/`k` (and arrows) move, `g`/`G` jump, `i` or `/` returns to
//!     Insert, `q`/Esc quits, Enter opens the file.

use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};

use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::StatefulImage;

use deepsearch_core::{DeepSearch, QueryOptions, SearchResult};

use crate::open::{candidates_for, AppKind, OpenApp};
use crate::preview::{Preview, PreviewRequest, PreviewWorker};

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
    input: String,
    results: Vec<SearchResult>,
    list_state: ListState,

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

    dirty: bool,
    last_edit: Instant,
    status: String,
}

impl App {
    pub fn new(ds: DeepSearch, opts: QueryOptions) -> Self {
        let n = ds.len();
        App {
            ds,
            opts,
            mode: Mode::Insert,
            input: String::new(),
            results: Vec::new(),
            list_state: ListState::default(),
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
            dirty: false,
            last_edit: Instant::now(),
            status: format!("{n} documents indexed — start typing to search"),
        }
    }

    /// Run the UI to completion.
    pub fn run(mut self) -> Result<()> {
        let mut terminal = ratatui::init();
        let res = self.event_loop(&mut terminal);
        ratatui::restore();
        res
    }

    fn event_loop(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        loop {
            terminal.draw(|f| self.render(f))?;
            self.drain_previews();
            self.drain_ai();

            if event::poll(TICK)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    match self.handle_key(key.code, key.modifiers) {
                        Action::Quit => break,
                        Action::SmartOpen => self.smart_open(terminal)?,
                        Action::OpenWith(app) => self.launch_app(terminal, app)?,
                        Action::None => {}
                    }
                }
            }

            if self.dirty && self.last_edit.elapsed() >= DEBOUNCE {
                self.run_search();
                self.dirty = false;
            }
        }
        Ok(())
    }

    // --- input handling ---------------------------------------------------

    fn handle_key(&mut self, code: KeyCode, mods: KeyModifiers) -> Action {
        // Overlays capture keys while they are up.
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
                KeyCode::Char('c') => return Action::Quit,
                KeyCode::Char('n') => {
                    self.move_selection(1);
                    return Action::None;
                }
                KeyCode::Char('p') => {
                    self.move_selection(-1);
                    return Action::None;
                }
                KeyCode::Char('u') => {
                    self.input.clear();
                    self.mark_dirty();
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
                _ => {}
            }
        }

        match code {
            KeyCode::Down => self.move_selection(1),
            KeyCode::Up => self.move_selection(-1),
            KeyCode::PageDown => self.move_selection(10),
            KeyCode::PageUp => self.move_selection(-10),
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

    fn handle_insert(&mut self, code: KeyCode) -> Action {
        match code {
            KeyCode::Char(c) => {
                self.input.push(c);
                self.mark_dirty();
            }
            KeyCode::Backspace => {
                self.input.pop();
                self.mark_dirty();
            }
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
        let request = self.input.trim().to_string();
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
    fn drain_ai(&mut self) {
        let Some(rx) = self.ai_rx.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(query)) => {
                self.ai_rx = None;
                self.input = query;
                self.status = format!("AI → {}", self.input);
                self.mark_dirty();
            }
            Ok(Err(e)) => {
                self.ai_rx = None;
                self.status = format!("AI: {e}");
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {} // still working
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.ai_rx = None;
                self.status = "AI request failed".to_string();
            }
        }
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
        self.last_edit = Instant::now();
    }

    // --- search & selection ----------------------------------------------

    fn run_search(&mut self) {
        if self.input.trim().is_empty() {
            self.results.clear();
            self.list_state.select(None);
            self.preview = Preview::Loading;
            self.showing_image = false;
            self.status = format!("{} documents indexed", self.ds.len());
            return;
        }
        let start = Instant::now();
        self.results = self.ds.search(&self.input, &self.opts);
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
    fn drain_previews(&mut self) {
        while let Ok((gen, preview)) = self.worker.rx.try_recv() {
            if gen != self.generation {
                continue; // stale
            }
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

        ratatui::restore();
        let status = std::process::Command::new(&editor).arg(&path).status();
        *terminal = ratatui::init();
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
            ratatui::restore();
            let status = std::process::Command::new(&app.program)
                .args(&app.args)
                .status();
            *terminal = ratatui::init();
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
                    Span::styled(
                        num,
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(
                        a.label.clone(),
                        Style::default()
                            .fg(kind_color(a.kind))
                            .add_modifier(Modifier::BOLD),
                    ),
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

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(Line::from(vec![
                Span::styled(" Open ", Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(filename, Style::default().fg(Color::Yellow)),
                Span::raw(" "),
            ]))
            .title_bottom(Line::from(Span::styled(
                " 1-9 open · ↑/↓ move · Enter · Esc ",
                Style::default().fg(Color::DarkGray),
            )));

        let list = List::new(items)
            .block(block)
            .highlight_style(Style::default().bg(Color::DarkGray))
            .highlight_symbol("▶ ");

        frame.render_widget(Clear, area);
        frame.render_stateful_widget(list, area, &mut menu.state);
    }

    fn render_query(&self, frame: &mut Frame, area: Rect) {
        let mode_tag = match self.mode {
            Mode::Insert => Span::styled(
                " INSERT ",
                Style::default().bg(Color::Green).fg(Color::Black),
            ),
            Mode::Normal => Span::styled(
                " NORMAL ",
                Style::default().bg(Color::Blue).fg(Color::White),
            ),
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .title(Line::from(vec![Span::raw(" deepsearch "), mode_tag]));
        let text = Line::from(vec![
            Span::styled(
                "❯ ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(&self.input),
        ]);
        frame.render_widget(Paragraph::new(text).block(block), area);

        if self.mode == Mode::Insert && area.width > 2 {
            // Place the cursor right after the prompt + current input, clamped
            // inside the box (saturating math guards against tiny terminals).
            let max_x = area.right().saturating_sub(2);
            let x = (area.x + 4 + self.input.chars().count() as u16).min(max_x);
            frame.set_cursor_position(Position::new(x, area.y + 1));
        }
    }

    fn render_results(&mut self, frame: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self
            .results
            .iter()
            .map(|r| {
                let name = r
                    .path
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let parent = r
                    .path
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let line = Line::from(vec![
                    Span::styled(
                        format!("{:>6.2} ", r.score),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled(
                        format!("[{}] ", type_tag(r.file_type)),
                        Style::default().fg(Color::Magenta),
                    ),
                    Span::styled(name, Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(format!("  {parent}"), Style::default().fg(Color::DarkGray)),
                ]);
                ListItem::new(line)
            })
            .collect();

        let title = format!(" results ({}) ", self.results.len());
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(title))
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▶ ");
        frame.render_stateful_widget(list, area, &mut self.list_state);
    }

    fn render_preview(&mut self, frame: &mut Frame, area: Rect) {
        let title = self
            .selected()
            .map(|r| format!(" {} ", r.path.display()))
            .unwrap_or_else(|| " preview ".to_string());
        let block = Block::default().borders(Borders::ALL).title(title);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.showing_image {
            if let Some(state) = self.image_state.as_mut() {
                frame.render_stateful_widget(StatefulImage::default(), inner, state);
                return;
            }
        }

        let text: Text = match &self.preview {
            Preview::Text(t) | Preview::Meta(t) => t.clone(),
            Preview::Loading => Text::from(Line::from(Span::styled(
                "…",
                Style::default().fg(Color::DarkGray),
            ))),
            Preview::Error(e) => Text::from(Line::from(Span::styled(
                e.clone(),
                Style::default().fg(Color::Red),
            ))),
            // Image handled above; if we get here the picker failed.
            Preview::Image(_) => Text::from("image"),
        };
        frame.render_widget(Paragraph::new(text).wrap(Wrap { trim: false }), inner);
    }

    fn render_status(&self, frame: &mut Frame, area: Rect) {
        let help = "↑/↓ move · Enter open · o open-with · Ctrl-a ask AI · F1 help · Esc/q quit";
        let line = Line::from(vec![
            Span::styled(
                format!(" {} ", self.status),
                Style::default().fg(Color::Black).bg(Color::Cyan),
            ),
            Span::raw("  "),
            Span::styled(help, Style::default().fg(Color::DarkGray)),
        ]);
        frame.render_widget(Paragraph::new(line), area);
    }
}

/// Colour used for an open-with entry's label, grouping it by kind at a glance.
fn kind_color(kind: AppKind) -> Color {
    match kind {
        AppKind::Editor => Color::Cyan,
        AppKind::Image => Color::Green,
        AppKind::Pdf => Color::Red,
        AppKind::Media => Color::Magenta,
        AppKind::Default => Color::Yellow,
        AppKind::Reveal | AppKind::Terminal => Color::Gray,
    }
}

/// A centered overlay listing every keybinding.
fn render_help(frame: &mut Frame) {
    let rows: &[(&str, &str)] = &[
        ("type", "edit the query (filters as you type)"),
        ("ext:rs / type:pdf", "filter results by extension or type"),
        ("↑ / ↓  ·  Ctrl-n / Ctrl-p", "move selection"),
        ("PageUp / PageDown", "move by 10"),
        ("Enter", "open in the right app for the file"),
        ("o  ·  Ctrl-o", "open-with menu (choose an app)"),
        ("Ctrl-a", "ask in plain language (needs local Ollama)"),
        ("y  ·  Ctrl-y", "copy the file path to the clipboard"),
        ("Ctrl-u", "clear the query"),
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
                Span::styled(
                    format!(" {key:<26}"),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(*desc, Style::default().fg(Color::Gray)),
            ])
        })
        .collect();

    let height = rows.len() as u16 + 2;
    let area = centered_rect(64, height.min(frame.area().height), frame.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            " Keys ",
            Style::default().add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Line::from(Span::styled(
            " any key to close ",
            Style::default().fg(Color::DarkGray),
        )));

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
    }
}
