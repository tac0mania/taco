//! The GPT-2 pre-tokenizer: it divides valid UTF-8 text into pieces, as the published regex
//! does, without a regex engine.
//!
//! PLAN(§10): the pattern of `tools/baseline/bpe.py` (published in openai/gpt-2 `encoder.py`):
//!
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! ```
//!
//! The regex engine tries the alternatives from left to right at the end of the previous
//! match, and each `+` is greedy. Every character is in exactly one of four classes: letter
//! (`\p{L}`), number (`\p{N}`), white space (`\s`) or other. Thus the alternatives reduce to
//! these rules, applied at the start of the remaining text:
//!
//! 1. An apostrophe followed by `s`, `t`, `re`, `ve`, `m`, `ll` or `d` is one piece.
//! 2. A letter, number or other character starts a run of characters of the same class.
//! 3. A space (U+0020) followed by a letter, number or other character starts a run of that
//!    class, with the space at its start.
//! 4. Else the text starts with `n` white-space characters. If they reach the end of the text,
//!    or if `n` is 1, they are one piece. Else the first `n - 1` are one piece: `\s+(?!\S)`
//!    gives back the last white-space character, so that it can start a piece of rule 3.
//!
//! The pieces cover the text with no gap, because every character matches an alternative.
//!
//! DETERMINISM: the classes come from the tables in `unicode_tables`, not from the Unicode
//! version of the Rust toolchain.

use super::unicode_tables::{LETTER, NUMBER, Ranges, WHITESPACE};

/// The class of a character in the pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// `\p{L}`.
    Letter,
    /// `\p{N}`.
    Number,
    /// `\s`.
    Space,
    /// Not `\s`, `\p{L}` or `\p{N}`.
    Other,
}

/// `true` if `c` is in one of the sorted, disjoint ranges. Cost: a binary search.
fn in_ranges(c: char, ranges: Ranges) -> bool {
    let c = u32::from(c);
    ranges
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                core::cmp::Ordering::Less
            } else if lo > c {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// The class of `c`. ASCII takes a fast path that gives the same result as the tables.
fn class(c: char) -> Class {
    if c.is_ascii() {
        return if c.is_ascii_alphabetic() {
            Class::Letter
        } else if c.is_ascii_digit() {
            Class::Number
        } else if matches!(c, '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | ' ') {
            Class::Space
        } else {
            Class::Other
        };
    }
    if in_ranges(c, LETTER) {
        Class::Letter
    } else if in_ranges(c, NUMBER) {
        Class::Number
    } else if in_ranges(c, WHITESPACE) {
        Class::Space
    } else {
        Class::Other
    }
}

/// The contraction suffixes of rule 1, in the order of the pattern.
const CONTRACTIONS: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];

/// The byte length of the first piece of `text`. It is 0 only for empty text.
///
/// Cost: one pass over the characters of the piece, and a binary search in a table of at most
/// 694 ranges for each character that is not ASCII.
fn first_piece_len(text: &str) -> usize {
    let mut chars = text.char_indices();
    let Some((_, c0)) = chars.next() else { return 0 };
    if c0 == '\'' {
        let rest = text.get(1..).unwrap_or("");
        if let Some(s) = CONTRACTIONS.iter().find(|s| rest.starts_with(**s)) {
            return 1 + s.len();
        }
    }
    let c0_class = class(c0);
    // Rules 2 and 3: the class of the run, and the byte offset where the run starts.
    let run = if c0_class == Class::Space {
        match chars.clone().next() {
            Some((i, c1)) if c0 == ' ' && class(c1) != Class::Space => Some((class(c1), i)),
            _ => None,
        }
    } else {
        Some((c0_class, 0))
    };
    if let Some((run_class, start)) = run {
        let tail = text.get(start..).unwrap_or("");
        let len = tail
            .char_indices()
            .find(|&(_, c)| class(c) != run_class)
            .map_or(tail.len(), |(i, _)| i);
        return start + len;
    }
    // Rule 4: a run of white space.
    let mut last_end = 0usize;
    let mut before_last_end = 0usize;
    for (count, (end, c)) in text.char_indices().map(|(i, c)| (i + c.len_utf8(), c)).enumerate() {
        if class(c) != Class::Space {
            // The run stops before a character that is not white space. `count` white-space
            // characters come before it.
            return if count >= 2 { before_last_end } else { last_end };
        }
        before_last_end = last_end;
        last_end = end;
    }
    last_end
}

/// An iterator over the pieces of valid UTF-8 text. See the module documentation.
#[derive(Clone, Debug)]
pub(super) struct Pieces<'a> {
    rest: &'a str,
}

impl<'a> Pieces<'a> {
    /// The pieces of `text`, in order. Their concatenation is `text`.
    pub(super) fn new(text: &'a str) -> Self {
        Self { rest: text }
    }
}

impl<'a> Iterator for Pieces<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let len = first_piece_len(self.rest);
        if len == 0 {
            return None;
        }
        // INVARIANT: `len` is the end of a character of `rest`, so both splits are valid.
        let (piece, rest) = self.rest.split_at_checked(len)?;
        self.rest = rest;
        Some(piece)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(s: &str) -> Vec<&str> {
        Pieces::new(s).collect()
    }

    #[test]
    fn splits_words_with_their_leading_space() {
        assert_eq!(pieces("Hello world"), ["Hello", " world"]);
        assert_eq!(pieces("  two  spaces"), [" ", " two", " ", " spaces"]);
    }

    #[test]
    fn contractions_start_only_at_the_apostrophe() {
        assert_eq!(pieces("don't we'll"), ["don", "'t", " we", "'ll"]);
        assert_eq!(pieces("\"'s"), ["\"'", "s"]);
        assert_eq!(pieces("'S"), ["'", "S"]);
    }

    #[test]
    fn white_space_gives_back_its_last_character() {
        assert_eq!(pieces("a\n\n b"), ["a", "\n\n", " b"]);
        assert_eq!(pieces("a\tb"), ["a", "\t", "b"]);
        assert_eq!(pieces("a   "), ["a", "   "]);
        assert_eq!(pieces(" "), [" "]);
        assert_eq!(pieces("x \n"), ["x", " \n"]);
    }

    #[test]
    fn numbers_and_other_characters_make_their_own_runs() {
        assert_eq!(pieces("abc123!?"), ["abc", "123", "!?"]);
        assert_eq!(pieces(" 42 ++"), [" 42", " ++"]);
    }

    #[test]
    fn devanagari_marks_are_not_letters() {
        // U+0939 HA is a letter (Lo). U+093F VOWEL SIGN I is a mark (Mc), so it is "other".
        assert_eq!(pieces("\u{939}\u{93F}"), ["\u{939}", "\u{93F}"]);
    }

    #[test]
    fn ascii_fast_path_agrees_with_the_tables() {
        for b in 0u8..128 {
            let c = char::from(b);
            let table = if in_ranges(c, LETTER) {
                Class::Letter
            } else if in_ranges(c, NUMBER) {
                Class::Number
            } else if in_ranges(c, WHITESPACE) {
                Class::Space
            } else {
                Class::Other
            };
            assert_eq!(class(c), table, "byte {b}");
        }
    }

    #[test]
    fn tables_are_sorted_and_disjoint() {
        for t in [LETTER, NUMBER, WHITESPACE] {
            for w in t.windows(2) {
                assert!(w[0].0 <= w[0].1 && w[0].1 < w[1].0);
            }
        }
    }

    #[test]
    fn std_whitespace_is_the_same_set() {
        // White_Space is a stable property. This test documents the agreement.
        for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            assert_eq!(in_ranges(c, WHITESPACE), c.is_whitespace(), "U+{:04X}", u32::from(c));
        }
    }
}
