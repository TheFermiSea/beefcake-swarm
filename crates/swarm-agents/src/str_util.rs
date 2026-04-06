//! Shared string utilities for UTF-8-safe operations.
//!
//! Consolidated here to eliminate 6+ copies of char-boundary truncation logic
//! scattered across runtime_adapter, context_firewall, dispatch, reformulation,
//! and subtask modules.

/// Truncate a string at a char boundary, returning a `&str` slice.
/// Safe for multi-byte UTF-8 (em dash, Unicode quotes in rustc output, etc.).
///
/// Returns the full string if `max_len >= s.len()`.
pub fn safe_truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Truncate a string at a char boundary, appending "..." if truncated.
/// Returns an owned `String`.
pub fn safe_truncate_with_ellipsis(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", safe_truncate(s, max_len))
    }
}

/// Snap a byte offset forward to the nearest char boundary.
/// Returns `s.len()` if the offset is beyond the string.
pub fn snap_to_char_boundary(s: &str, byte_offset: usize) -> usize {
    let mut pos = byte_offset.min(s.len());
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_truncate_ascii() {
        assert_eq!(safe_truncate("hello", 10), "hello");
        assert_eq!(safe_truncate("hello world", 5), "hello");
        assert_eq!(safe_truncate("", 5), "");
    }

    #[test]
    fn test_safe_truncate_multibyte() {
        // Em dash '—' is 3 bytes (0xE2 0x80 0x94)
        let s = "aaa—bbb"; // bytes: 3 + 3 + 3 = 9
        assert_eq!(safe_truncate(s, 4), "aaa"); // can't cut inside em dash
        assert_eq!(safe_truncate(s, 6), "aaa—"); // after em dash
    }

    #[test]
    fn test_safe_truncate_with_ellipsis() {
        assert_eq!(safe_truncate_with_ellipsis("hello", 10), "hello");
        assert_eq!(safe_truncate_with_ellipsis("hello world", 5), "hello...");
    }

    #[test]
    fn test_snap_to_char_boundary() {
        let s = "aaa—bbb";
        assert_eq!(snap_to_char_boundary(s, 3), 3); // at boundary
        assert_eq!(snap_to_char_boundary(s, 4), 6); // inside em dash → snap forward
        assert_eq!(snap_to_char_boundary(s, 100), 9); // beyond end
    }
}
