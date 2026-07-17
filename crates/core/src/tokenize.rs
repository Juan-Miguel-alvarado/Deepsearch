//! Text normalization and tokenization.
//!
//! The same pipeline is applied at index time and at query time so that terms
//! always match. Pipeline:
//!   1. Split on any non-alphanumeric character (this handles `snake_case`,
//!      punctuation, whitespace, paths, etc.).
//!   2. Split each remaining chunk on `camelCase` / letter<->digit boundaries.
//!   3. Lowercase.
//!   4. Porter/Snowball (English) stemming.
//!   5. Drop empty tokens and absurdly long ones (usually base64/minified junk).

use rust_stemmers::{Algorithm, Stemmer};

thread_local! {
    // Snowball stemmers are stateless; keep one per thread to avoid rebuilding
    // the function tables on every call during a parallel index run.
    static STEMMER: Stemmer = Stemmer::create(Algorithm::English);
}

/// Longest token we keep. Anything longer is almost always a hash, a data URI
/// or a minified blob, which only pollutes the dictionary.
const MAX_TOKEN_LEN: usize = 40;

/// Tokenize `text` into normalized, **stemmed** terms (the index/query terms).
pub fn tokenize(text: &str) -> Vec<String> {
    normalize(text).iter().map(|t| stem_word(t)).collect()
}

/// Normalize `text` into lowercased, camel/snake-split tokens **without**
/// stemming. Used for fuzzy filename matching, where stemming would distort
/// typo'd words and inflate edit distances.
pub fn normalize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in text.split(|c: char| !c.is_alphanumeric()) {
        if chunk.is_empty() {
            continue;
        }
        for sub in split_identifier(chunk) {
            if !sub.is_empty() && sub.len() <= MAX_TOKEN_LEN {
                out.push(sub.to_lowercase());
            }
        }
    }
    out
}

/// Stem an already-normalized (lowercase) token.
pub fn stem_word(token: &str) -> String {
    STEMMER.with(|s| s.stem(token).into_owned())
}

/// Split an identifier on camelCase and letter<->digit boundaries.
///
/// Examples:
///   `getUserName` -> [get, User, Name]
///   `HTTPServer`  -> [HTTP, Server]
///   `parseUTF8`   -> [parse, UTF, 8]
fn split_identifier(word: &str) -> Vec<&str> {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() <= 1 {
        return vec![word];
    }
    // Compute byte offsets so we can return string slices.
    let mut byte_offsets = Vec::with_capacity(chars.len() + 1);
    let mut acc = 0usize;
    for c in &chars {
        byte_offsets.push(acc);
        acc += c.len_utf8();
    }
    byte_offsets.push(acc);

    let mut parts = Vec::new();
    let mut start = 0usize; // index into `chars`
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let boundary =
            // lower/digit -> upper : "getUser"
            (!prev.is_uppercase() && cur.is_uppercase())
            // acronym end -> word : "HTTPServer" splits before 'S' because next is lower
            || (prev.is_uppercase() && cur.is_uppercase()
                && i + 1 < chars.len() && chars[i + 1].is_lowercase())
            // letter <-> digit boundary
            || (prev.is_alphabetic() && cur.is_numeric())
            || (prev.is_numeric() && cur.is_alphabetic());
        if boundary {
            parts.push(&word[byte_offsets[start]..byte_offsets[i]]);
            start = i;
        }
    }
    parts.push(&word[byte_offsets[start]..byte_offsets[chars.len()]]);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_snake_case() {
        assert_eq!(tokenize("user_name_field"), vec!["user", "name", "field"]);
    }

    #[test]
    fn splits_camel_case() {
        assert_eq!(tokenize("getUserName"), vec!["get", "user", "name"]);
    }

    #[test]
    fn handles_acronyms() {
        assert_eq!(
            tokenize("HTTPServerConfig"),
            vec!["http", "server", "config"]
        );
    }

    #[test]
    fn splits_letter_digit() {
        assert_eq!(tokenize("parseUTF8"), vec!["pars", "utf", "8"]);
    }

    #[test]
    fn lowercases_and_stems() {
        // "running" and "runs" stem to the same root.
        assert_eq!(tokenize("Running"), tokenize("runs"));
    }

    #[test]
    fn drops_punctuation_and_paths() {
        assert_eq!(
            tokenize("/home/juan/main.rs"),
            vec!["home", "juan", "main", "rs"]
        );
    }

    #[test]
    fn drops_overly_long_tokens() {
        let junk = "a".repeat(200);
        assert!(tokenize(&junk).is_empty());
    }

    #[test]
    fn empty_input() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("   \n\t  ").is_empty());
    }
}
