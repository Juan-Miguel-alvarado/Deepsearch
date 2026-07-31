//! A single-line text input with a cursor and horizontal scrolling.
//!
//! Split out from the UI because it is pure logic over a string: no terminal, no ratatui, no
//! state beyond the text and the caret. That makes the fiddly parts — moving through multi-byte
//! characters, deleting a word, keeping the caret visible in a query longer than the box — cheap
//! to test exhaustively, which is exactly where an input box goes wrong.
//!
//! Positions are **character** indices, never byte offsets, so an accented letter behaves like
//! any other. Widths are measured in **terminal cells**, so a CJK character that occupies two
//! columns is accounted for when deciding what fits.

use unicode_width::UnicodeWidthChar;

/// What is currently visible in the box, and where to put the caret.
#[derive(Debug, PartialEq, Eq)]
pub struct View {
    /// The slice of the query that fits.
    pub text: String,
    /// Caret offset from the start of `text`, in terminal cells.
    pub cursor_col: usize,
    /// Whether text is scrolled off the left edge.
    pub scrolled: bool,
}

/// An editable line of text.
#[derive(Debug, Default, Clone)]
pub struct LineInput {
    text: String,
    /// Caret position as a character index in `0..=char_len`.
    cursor: usize,
}

impl LineInput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Number of characters, which is also the caret's maximum position.
    pub fn len(&self) -> usize {
        self.text.chars().count()
    }

    /// Only assertions need this today; the widget derives everything it draws from [`Self::view`].
    #[cfg(test)]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Byte offset of a character index, clamped to the end of the string.
    fn byte_of(&self, char_index: usize) -> usize {
        self.text
            .char_indices()
            .nth(char_index)
            .map_or(self.text.len(), |(byte, _)| byte)
    }

    // --- editing. Each returns whether the text changed, so the caller knows
    // whether a new search is needed rather than guessing. ---

    pub fn insert(&mut self, c: char) -> bool {
        let at = self.byte_of(self.cursor);
        self.text.insert(at, c);
        self.cursor += 1;
        true
    }

    pub fn insert_str(&mut self, s: &str) -> bool {
        if s.is_empty() {
            return false;
        }
        let at = self.byte_of(self.cursor);
        self.text.insert_str(at, s);
        self.cursor += s.chars().count();
        true
    }

    /// Delete the character before the caret.
    pub fn backspace(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        self.cursor -= 1;
        self.text.remove(self.byte_of(self.cursor));
        true
    }

    /// Delete the character under the caret.
    pub fn delete(&mut self) -> bool {
        if self.cursor >= self.len() {
            return false;
        }
        self.text.remove(self.byte_of(self.cursor));
        true
    }

    /// Delete backwards to the start of the previous word, the way a shell does: skip any
    /// whitespace immediately behind the caret, then the run of non-whitespace before it.
    pub fn delete_word_back(&mut self) -> bool {
        if self.cursor == 0 {
            return false;
        }
        let chars: Vec<char> = self.text.chars().collect();
        let mut start = self.cursor;
        while start > 0 && chars[start - 1].is_whitespace() {
            start -= 1;
        }
        while start > 0 && !chars[start - 1].is_whitespace() {
            start -= 1;
        }
        let (from, to) = (self.byte_of(start), self.byte_of(self.cursor));
        self.text.replace_range(from..to, "");
        self.cursor = start;
        true
    }

    /// Delete from the caret to the end of the line.
    pub fn kill_to_end(&mut self) -> bool {
        if self.cursor >= self.len() {
            return false;
        }
        let at = self.byte_of(self.cursor);
        self.text.truncate(at);
        true
    }

    pub fn clear(&mut self) -> bool {
        if self.text.is_empty() {
            return false;
        }
        self.text.clear();
        self.cursor = 0;
        true
    }

    /// Replace the whole line, leaving the caret at the end.
    pub fn set(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.cursor = self.len();
    }

    // --- caret movement. These never change the text. ---

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.len());
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.len();
    }

    /// The portion of the line that fits in `width` cells, with the caret always visible.
    ///
    /// Scrolling is derived from the caret rather than remembered, so it cannot drift out of sync
    /// with the text after an edit. When the caret sits past the right edge the window is
    /// right-anchored to it; the caller is told when content is hidden to the left so it can show
    /// a marker.
    pub fn view(&self, width: usize) -> View {
        if width == 0 {
            return View {
                text: String::new(),
                cursor_col: 0,
                scrolled: false,
            };
        }

        let chars: Vec<char> = self.text.chars().collect();
        let cell_width = |c: char| c.width().unwrap_or(0);
        // One cell is reserved so the caret itself has somewhere to sit at the end of the line.
        let budget = width.saturating_sub(1).max(1);

        // Walk back from the caret until the text between `start` and the caret fills the budget.
        let mut start = self.cursor;
        let mut used = 0usize;
        while start > 0 {
            let w = cell_width(chars[start - 1]);
            if used + w > budget {
                break;
            }
            used += w;
            start -= 1;
        }

        let cursor_col = used;
        let mut text = String::new();
        let mut taken = 0usize;
        // Build the window forward from `start`, which is at most `width` cells wide and always
        // contains the caret.
        for &c in &chars[start..] {
            let w = cell_width(c);
            if taken + w > width {
                break;
            }
            taken += w;
            text.push(c);
        }

        View {
            text,
            cursor_col,
            scrolled: start > 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LineInput;

    fn input(text: &str) -> LineInput {
        let mut input = LineInput::new();
        input.set(text);
        input
    }

    #[test]
    fn typing_inserts_at_the_caret_not_at_the_end() {
        let mut line = input("helo");
        line.home();
        line.right();
        line.right();
        line.right();
        assert!(line.insert('l'));
        assert_eq!(line.text(), "hello");
        assert_eq!(line.cursor(), 4);
    }

    #[test]
    fn backspace_and_delete_work_on_either_side_of_the_caret() {
        let mut line = input("abcd");
        line.home();
        line.right();
        line.right();
        assert!(line.backspace());
        assert_eq!(line.text(), "acd");
        assert!(line.delete());
        assert_eq!(line.text(), "ad");
        assert_eq!(line.cursor(), 1);
    }

    #[test]
    fn deleting_past_either_edge_is_a_no_op_rather_than_a_panic() {
        let mut line = LineInput::new();
        assert!(!line.backspace());
        assert!(!line.delete());
        assert!(!line.kill_to_end());
        assert!(!line.clear());

        let mut line = input("x");
        line.home();
        assert!(!line.backspace());
        line.end();
        assert!(!line.delete());
    }

    #[test]
    fn multi_byte_characters_are_one_step_each() {
        // Every character here is multi-byte; byte-indexing would panic or corrupt the string.
        let mut line = input("añoñ");
        assert_eq!(
            line.cursor(),
            4,
            "the caret counts characters, not the 6 bytes"
        );

        line.left(); // now between 'o' and the final 'ñ'
        assert!(line.backspace());
        assert_eq!(
            line.text(),
            "aññ",
            "backspace removes the 'o', not half of a 'ñ'"
        );

        assert!(line.insert('é'));
        assert_eq!(line.text(), "añéñ");
        assert_eq!(line.cursor(), 3);
    }

    #[test]
    fn word_delete_eats_the_word_and_the_space_behind_it() {
        let mut line = input("hola que tal");
        assert!(line.delete_word_back());
        assert_eq!(line.text(), "hola que ");
        assert!(line.delete_word_back());
        assert_eq!(line.text(), "hola ");
        assert!(line.delete_word_back());
        assert_eq!(line.text(), "");
        assert!(!line.delete_word_back());
    }

    #[test]
    fn word_delete_only_touches_text_behind_the_caret() {
        let mut line = input("uno dos tres");
        line.home();
        for _ in 0..7 {
            line.right();
        }
        assert!(line.delete_word_back());
        assert_eq!(line.text(), "uno  tres");
    }

    #[test]
    fn kill_to_end_keeps_what_is_behind_the_caret() {
        let mut line = input("keep this drop that");
        line.home();
        for _ in 0..10 {
            line.right();
        }
        assert!(line.kill_to_end());
        assert_eq!(line.text(), "keep this ");
    }

    #[test]
    fn caret_movement_is_clamped_to_the_line() {
        let mut line = input("ab");
        line.end();
        line.right();
        line.right();
        assert_eq!(line.cursor(), 2);
        line.home();
        line.left();
        assert_eq!(line.cursor(), 0);
    }

    #[test]
    fn a_short_line_is_shown_whole() {
        let line = input("hello");
        let view = line.view(40);
        assert_eq!(view.text, "hello");
        assert_eq!(view.cursor_col, 5);
        assert!(!view.scrolled);
    }

    #[test]
    fn a_long_line_scrolls_to_keep_the_caret_visible() {
        let line = input(&"abcdefghij".repeat(5)); // 50 characters
        let view = line.view(10);
        assert!(view.scrolled, "content is hidden to the left");
        assert!(view.text.chars().count() <= 10);
        assert!(view.cursor_col < 10, "the caret must stay inside the box");
        assert!(
            line.text().ends_with(&view.text),
            "the window is anchored to the caret"
        );
    }

    #[test]
    fn scrolling_back_to_the_start_stops_scrolling() {
        let mut line = input(&"x".repeat(50));
        line.home();
        let view = line.view(10);
        assert!(!view.scrolled);
        assert_eq!(view.cursor_col, 0);
        assert!(view.text.starts_with('x'));
    }

    #[test]
    fn wide_characters_are_measured_in_cells_not_characters() {
        // Each of these occupies two terminal columns.
        let line = input("日本語テキスト");
        let view = line.view(10);
        let cells: usize = view
            .text
            .chars()
            .map(|c| unicode_width::UnicodeWidthChar::width(c).unwrap_or(0))
            .sum();
        assert!(
            cells <= 10,
            "the window must not overflow the box: {cells} cells"
        );
        assert!(view.cursor_col <= 10);
    }

    #[test]
    fn a_zero_width_box_does_not_panic() {
        let line = input("anything");
        let view = line.view(0);
        assert!(view.text.is_empty());
        assert_eq!(view.cursor_col, 0);
    }

    #[test]
    fn set_moves_the_caret_to_the_end() {
        let mut line = input("short");
        line.home();
        line.set("a much longer replacement");
        assert_eq!(line.cursor(), line.len());
    }
}
