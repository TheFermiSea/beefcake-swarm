//! Shell Safety — Command Injection Prevention
//!
//! Provides sanitization and validation utilities for command execution paths.
//! Used wherever subprocess commands are constructed, especially SSH remote
//! execution where arguments are joined into shell strings.
//!
//! # Threat Model
//!
//! - **SSH remote execution:** `ssh host "cmd arg1 arg2"` invokes a shell on the
//!   remote host. Metacharacters in arguments can execute arbitrary commands.
//! - **Direct execution:** `Command::new(cmd).args(args)` does NOT invoke a shell,
//!   so metacharacters are harmless. But we validate anyway as defense-in-depth.
//!
//! # Usage
//!
//! ```rust,ignore
//! use coordination::shell_safety::{escape_for_ssh, validate_arg};
//!
//! // For SSH remote commands: escape each argument
//! let safe_arg = escape_for_ssh(user_input);
//! let cmd = format!("sbatch --parsable {}", safe_arg);
//! Command::new("ssh").args([host, &cmd]).output()?;
//!
//! // For direct commands: validate (defense-in-depth)
//! validate_arg(user_input)?;
//! Command::new("cargo").arg(user_input).output()?;
//! ```

/// Shell metacharacters that can cause command injection when interpreted
/// by a shell (bash/sh/zsh).
const SHELL_METACHARACTERS: &[char] = &[
    ';', '|', '&', '`', '$', '(', ')', '{', '}', '<', '>', '\n', '\r', '!', '#', '~', '*', '?',
    '[', ']', '\\', '"', '\'',
];

/// Subset of metacharacters that indicate chaining/injection intent
/// (vs. globbing characters that might appear in legitimate args).
const INJECTION_CHARACTERS: &[char] = &[';', '|', '&', '`', '$', '(', ')', '\n', '\r'];

/// Validation error for argument checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgValidationError {
    /// The problematic character found.
    pub character: char,
    /// Position in the input string.
    pub position: usize,
    /// The original input (truncated to 100 chars).
    pub input_preview: String,
}

impl std::fmt::Display for ArgValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dangerous character '{}' at position {} in: {}",
            self.character.escape_default(),
            self.position,
            self.input_preview
        )
    }
}

impl std::error::Error for ArgValidationError {}

/// Escape a string for safe inclusion in an SSH command string.
///
/// Wraps the argument in single quotes and escapes any embedded single quotes
/// using the `'\''` pattern (end quote, escaped quote, start quote).
///
/// This is the POSIX-standard way to prevent shell interpretation:
/// - Single-quoted strings pass through literally (no `$`, `\`, `` ` `` expansion)
/// - The only character that needs escaping inside `'...'` is `'` itself
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(escape_for_ssh("hello"), "'hello'");
/// assert_eq!(escape_for_ssh("it's"), "'it'\\''s'");
/// assert_eq!(escape_for_ssh("$(rm -rf /)"), "'$(rm -rf /)'");
/// ```
pub fn escape_for_ssh(arg: &str) -> String {
    // Replace each single quote with: end-quote, backslash-quote, start-quote
    let escaped = arg.replace('\'', "'\\''");
    format!("'{}'", escaped)
}

/// Validate that an argument contains no injection-class metacharacters.
///
/// Use this for defense-in-depth on arguments passed to `Command::new().arg()`,
/// which doesn't invoke a shell but where we still want to catch suspicious input.
///
/// Returns Ok(()) if clean, Err with details if a dangerous character is found.
pub fn validate_arg(arg: &str) -> Result<(), ArgValidationError> {
    for (pos, ch) in arg.chars().enumerate() {
        if INJECTION_CHARACTERS.contains(&ch) {
            return Err(ArgValidationError {
                character: ch,
                position: pos,
                input_preview: if arg.len() > 100 {
                    format!("{}...", &arg[..100])
                } else {
                    arg.to_string()
                },
            });
        }
    }
    Ok(())
}

/// Validate that an argument contains no shell metacharacters at all.
///
/// Stricter than [`validate_arg`] — also rejects globbing characters, quotes,
/// and other characters that have special meaning in shells.
pub fn validate_strict(arg: &str) -> Result<(), ArgValidationError> {
    for (pos, ch) in arg.chars().enumerate() {
        if SHELL_METACHARACTERS.contains(&ch) {
            return Err(ArgValidationError {
                character: ch,
                position: pos,
                input_preview: if arg.len() > 100 {
                    format!("{}...", &arg[..100])
                } else {
                    arg.to_string()
                },
            });
        }
    }
    Ok(())
}

/// Build a safe SSH command string from a program and arguments.
///
/// Each argument is individually escaped with [`escape_for_ssh`], then joined.
/// The result is safe to pass as `ssh host "<result>"`.
///
/// # Example
///
/// ```rust,ignore
/// let cmd = build_ssh_command("sbatch", &["--parsable", "/path/to/script.slurm"]);
/// // Returns: "sbatch '--parsable' '/path/to/script.slurm'"
/// ```
pub fn build_ssh_command(program: &str, args: &[&str]) -> String {
    let mut parts = vec![escape_for_ssh(program)];
    for arg in args {
        parts.push(escape_for_ssh(arg));
    }
    parts.join(" ")
}

/// Sanitize a string for use as a filename or identifier component.
///
/// Replaces any character that is not alphanumeric, `-`, `_`, or `.` with `_`.
/// Also prevents path traversal by replacing `/` and `\`.
pub fn sanitize_identifier(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_for_ssh_simple() {
        assert_eq!(escape_for_ssh("hello"), "'hello'");
        assert_eq!(escape_for_ssh("--parsable"), "'--parsable'");
        assert_eq!(
            escape_for_ssh("/cluster/shared/scripts/run.slurm"),
            "'/cluster/shared/scripts/run.slurm'"
        );
    }

    #[test]
    fn test_escape_for_ssh_single_quotes() {
        assert_eq!(escape_for_ssh("it's"), "'it'\\''s'");
        assert_eq!(escape_for_ssh("'"), "''\\'''");
    }

    #[test]
    fn test_escape_for_ssh_injection_attempts() {
        // Command substitution
        assert_eq!(escape_for_ssh("$(rm -rf /)"), "'$(rm -rf /)'");
        // Backtick substitution
        assert_eq!(escape_for_ssh("`rm -rf /`"), "'`rm -rf /`'");
        // Semicolon chaining
        assert_eq!(escape_for_ssh("; rm -rf /"), "'; rm -rf /'");
        // Pipe
        assert_eq!(escape_for_ssh("| cat /etc/passwd"), "'| cat /etc/passwd'");
        // Ampersand
        assert_eq!(escape_for_ssh("& curl evil.com"), "'& curl evil.com'");
    }

    #[test]
    fn test_escape_for_ssh_empty() {
        assert_eq!(escape_for_ssh(""), "''");
    }

    #[test]
    fn test_escape_for_ssh_spaces() {
        assert_eq!(escape_for_ssh("path with spaces"), "'path with spaces'");
    }

    #[test]
    fn test_validate_arg_clean() {
        assert!(validate_arg("hello").is_ok());
        assert!(validate_arg("--flag=value").is_ok());
        assert!(validate_arg("/path/to/file").is_ok());
        assert!(validate_arg("file.rs").is_ok());
        assert!(validate_arg("").is_ok());
        // Globbing chars are OK for validate_arg (only injection chars blocked)
        assert!(validate_arg("*.rs").is_ok());
        assert!(validate_arg("src/**/*.rs").is_ok());
    }

    #[test]
    fn test_validate_arg_injection() {
        let err = validate_arg("; rm -rf /").unwrap_err();
        assert_eq!(err.character, ';');
        assert_eq!(err.position, 0);

        let err = validate_arg("foo | bar").unwrap_err();
        assert_eq!(err.character, '|');

        let err = validate_arg("foo & bar").unwrap_err();
        assert_eq!(err.character, '&');

        let err = validate_arg("$(evil)").unwrap_err();
        assert_eq!(err.character, '$');

        let err = validate_arg("`evil`").unwrap_err();
        assert_eq!(err.character, '`');

        let err = validate_arg("foo\nbar").unwrap_err();
        assert_eq!(err.character, '\n');
    }

    #[test]
    fn test_validate_strict_rejects_globs() {
        assert!(validate_strict("*.rs").is_err());
        assert!(validate_strict("file[0]").is_err());
        assert!(validate_strict("path?").is_err());
        assert!(validate_strict("$HOME").is_err());
        assert!(validate_strict("\"quoted\"").is_err());
    }

    #[test]
    fn test_validate_strict_clean() {
        assert!(validate_strict("hello").is_ok());
        assert!(validate_strict("--flag").is_ok());
        assert!(validate_strict("123").is_ok());
        assert!(validate_strict("/path/to/file.rs").is_ok());
        assert!(validate_strict("foo-bar_baz.txt").is_ok());
    }

    #[test]
    fn test_build_ssh_command() {
        let cmd = build_ssh_command("sbatch", &["--parsable", "/path/to/script.slurm"]);
        assert_eq!(cmd, "'sbatch' '--parsable' '/path/to/script.slurm'");
    }

    #[test]
    fn test_build_ssh_command_with_malicious_args() {
        let cmd = build_ssh_command("cat", &["/etc/passwd; rm -rf /"]);
        assert_eq!(cmd, "'cat' '/etc/passwd; rm -rf /'");
        // The semicolon is inside single quotes — shell won't interpret it
    }

    #[test]
    fn test_build_ssh_command_no_args() {
        let cmd = build_ssh_command("hostname", &[]);
        assert_eq!(cmd, "'hostname'");
    }

    #[test]
    fn test_sanitize_identifier() {
        assert_eq!(sanitize_identifier("hello-world"), "hello-world");
        assert_eq!(sanitize_identifier("test_123.rs"), "test_123.rs");
        assert_eq!(
            sanitize_identifier("../../etc/passwd"),
            ".._.._etc_passwd"
        );
        assert_eq!(sanitize_identifier("file name"), "file_name");
        assert_eq!(sanitize_identifier("a;b|c&d"), "a_b_c_d");
    }

    #[test]
    fn test_sanitize_identifier_empty() {
        assert_eq!(sanitize_identifier(""), "");
    }

    #[test]
    fn test_arg_validation_error_display() {
        let err = validate_arg("; injection").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("dangerous character"));
        assert!(msg.contains("; injection"));
    }

    #[test]
    fn test_validate_arg_long_input_truncated() {
        let long = "a".repeat(200) + ";";
        let err = validate_arg(&long).unwrap_err();
        assert!(err.input_preview.ends_with("..."));
        assert!(err.input_preview.len() < 110);
    }
}
