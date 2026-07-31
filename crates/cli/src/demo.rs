//! Frame capture for the README animation.
//!
//! The GIF in the README is not a screen recording. It drives the real [`App`](crate::tui::App)
//! with real keystrokes against a real index, renders each step through the same code path the
//! terminal uses, and writes the resulting cells out as ANSI. `docs/make-demo.py` then turns those
//! frames into the animation.
//!
//! The point is that the picture cannot drift from the product: if the layout changes, so does the
//! GIF the next time it is generated. Nothing here is mocked and nothing is hand-drawn.
//!
//! Regenerate with:
//!
//! ```text
//! cargo run --release -- --cache /tmp/ds-demo.bin index .
//! DEMO_CACHE=/tmp/ds-demo.bin cargo test --release -- --ignored capture_demo_frames
//! python3 docs/make-demo.py
//! ```

use std::fmt::Write as _;

use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};

/// Serialise a rendered buffer as ANSI text, one line per terminal row.
///
/// Only the escape sequences the UI actually emits are handled — the palette is deliberately
/// limited to ANSI slots (there is a test enforcing that), so there are no true-colour cases.
pub fn buffer_to_ansi(buf: &Buffer) -> String {
    let mut out = String::new();
    for y in 0..buf.area.height {
        let mut current = String::new();
        for x in 0..buf.area.width {
            let cell = &buf[(x, y)];
            let wanted = sgr_for(cell.fg, cell.bg, cell.modifier);
            if wanted != current {
                out.push_str("\x1b[0m");
                out.push_str(&wanted);
                current = wanted;
            }
            out.push_str(cell.symbol());
        }
        out.push_str("\x1b[0m");
        out.push('\n');
    }
    out
}

fn sgr_for(fg: Color, bg: Color, modifier: Modifier) -> String {
    let mut codes: Vec<u16> = Vec::new();
    if modifier.contains(Modifier::BOLD) {
        codes.push(1);
    }
    if modifier.contains(Modifier::DIM) {
        codes.push(2);
    }
    if modifier.contains(Modifier::ITALIC) {
        codes.push(3);
    }
    if modifier.contains(Modifier::UNDERLINED) {
        codes.push(4);
    }
    if modifier.contains(Modifier::REVERSED) {
        codes.push(7);
    }
    if let Some(code) = ansi_code(fg) {
        codes.push(code);
    }
    if let Some(code) = ansi_code(bg) {
        codes.push(code + 10);
    }
    if codes.is_empty() {
        return String::new();
    }
    let mut out = String::from("\x1b[");
    for (i, code) in codes.iter().enumerate() {
        if i > 0 {
            out.push(';');
        }
        let _ = write!(out, "{code}");
    }
    out.push('m');
    out
}

/// Foreground SGR number for a colour, or `None` for the terminal default.
fn ansi_code(color: Color) -> Option<u16> {
    Some(match color {
        Color::Reset => return None,
        Color::Black => 30,
        Color::Red => 31,
        Color::Green => 32,
        Color::Yellow => 33,
        Color::Blue => 34,
        Color::Magenta => 35,
        Color::Cyan => 36,
        Color::Gray => 37,
        Color::DarkGray => 90,
        Color::LightRed => 91,
        Color::LightGreen => 92,
        Color::LightYellow => 93,
        Color::LightBlue => 94,
        Color::LightMagenta => 95,
        Color::LightCyan => 96,
        Color::White => 97,
        // The UI never emits these; see `uses_only_the_terminal_palette`.
        Color::Rgb(..) | Color::Indexed(..) => return None,
    })
}

#[cfg(test)]
mod tests {
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::style::{Color, Modifier, Style};

    use super::buffer_to_ansi;

    #[test]
    fn plain_text_carries_no_escapes_beyond_the_resets() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 2, 1));
        buf[(0, 0)].set_symbol("h");
        buf[(1, 0)].set_symbol("i");
        assert_eq!(buffer_to_ansi(&buf), "hi\x1b[0m\n");
    }

    #[test]
    fn colours_and_modifiers_become_sgr() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
        buf[(0, 0)].set_symbol("x").set_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
        let ansi = buffer_to_ansi(&buf);
        assert!(ansi.contains("\x1b[1;36m"), "got {ansi:?}");
    }

    #[test]
    fn a_run_of_one_style_emits_one_escape() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 3, 1));
        for x in 0..3 {
            buf[(x, 0)]
                .set_symbol("=")
                .set_style(Style::default().fg(Color::Green));
        }
        let ansi = buffer_to_ansi(&buf);
        assert_eq!(
            ansi.matches("\x1b[32m").count(),
            1,
            "style should not repeat per cell"
        );
    }
}
