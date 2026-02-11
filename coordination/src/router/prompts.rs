//! Prompt templates for different task types
//!
//! Provides structured prompts optimized for each model and task type.

use crate::feedback::error_parser::{ErrorCategory, ParsedError};
use serde::{Deserialize, Serialize};

/// Prompt templates for different task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptTemplate {
    /// Fix compilation errors
    ErrorFix {
        code: String,
        errors: Vec<ErrorInfo>,
        context: Option<String>,
    },
    /// Generate code from description
    Generate {
        description: String,
        signature: Option<String>,
        examples: Vec<String>,
    },
    /// Refactor existing code
    Refactor {
        code: String,
        goal: String,
        constraints: Vec<String>,
    },
}

/// Simplified error info for prompts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub category: String,
    pub message: String,
    pub location: Option<String>,
    pub suggestion: Option<String>,
}

impl From<&ParsedError> for ErrorInfo {
    fn from(error: &ParsedError) -> Self {
        Self {
            category: error.category.to_string(),
            message: error.message.clone(),
            location: error.file.as_ref().map(|f| {
                format!(
                    "{}:{}:{}",
                    f,
                    error.line.unwrap_or(0),
                    error.column.unwrap_or(0)
                )
            }),
            suggestion: error.suggestion.clone(),
        }
    }
}

/// Builder for fix prompts
pub struct FixPromptBuilder {
    code: String,
    errors: Vec<ParsedError>,
    context: Option<String>,
    single_error_focus: bool,
    include_category_hints: bool,
}

impl FixPromptBuilder {
    /// Create a new builder
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            errors: Vec::new(),
            context: None,
            single_error_focus: true,
            include_category_hints: true,
        }
    }

    /// Add errors to fix
    pub fn with_errors(mut self, errors: Vec<ParsedError>) -> Self {
        self.errors = errors;
        self
    }

    /// Add context about the code
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Set whether to focus on single error
    pub fn single_error_focus(mut self, focus: bool) -> Self {
        self.single_error_focus = focus;
        self
    }

    /// Set whether to include category-specific hints
    pub fn include_hints(mut self, include: bool) -> Self {
        self.include_category_hints = include;
        self
    }

    /// Build the prompt string
    pub fn build(&self) -> String {
        let mut prompt = String::new();

        // Header
        prompt.push_str("# Rust Compilation Error Fix\n\n");

        // Context if provided
        if let Some(ctx) = &self.context {
            prompt.push_str(&format!("## Context\n{}\n\n", ctx));
        }

        // Current code
        prompt.push_str("## Current Code\n\n```rust\n");
        prompt.push_str(&self.code);
        prompt.push_str("\n```\n\n");

        // Errors
        prompt.push_str("## Compilation Errors\n\n");

        let errors_to_show = if self.single_error_focus && !self.errors.is_empty() {
            &self.errors[..1]
        } else {
            &self.errors[..]
        };

        for (i, error) in errors_to_show.iter().enumerate() {
            if errors_to_show.len() > 1 {
                prompt.push_str(&format!("### Error {} ({})\n\n", i + 1, error.category));
            }

            prompt.push_str(&format!("**Message:** {}\n", error.message));

            if let Some(code) = &error.code {
                prompt.push_str(&format!("**Code:** {}\n", code));
            }

            if let Some(file) = &error.file {
                prompt.push_str(&format!(
                    "**Location:** {}:{}:{}\n",
                    file,
                    error.line.unwrap_or(0),
                    error.column.unwrap_or(0)
                ));
            }

            if !error.labels.is_empty() {
                prompt.push_str(&format!("**Labels:** {}\n", error.labels.join(", ")));
            }

            if let Some(suggestion) = &error.suggestion {
                prompt.push_str(&format!("**Compiler suggestion:** `{}`\n", suggestion));
            }

            prompt.push_str(&format!(
                "\n**Full diagnostic:**\n```\n{}\n```\n\n",
                error.rendered
            ));
        }

        // Category-specific hints
        if self.include_category_hints {
            let hints = self.generate_category_hints();
            if !hints.is_empty() {
                prompt.push_str("## Hints\n\n");
                for hint in hints {
                    prompt.push_str(&format!("- {}\n", hint));
                }
                prompt.push('\n');
            }
        }

        // Instructions
        prompt.push_str("## Instructions\n\n");
        prompt.push_str("1. Fix the compilation error(s) listed above\n");
        prompt.push_str("2. Return ONLY the corrected Rust code\n");
        prompt.push_str("3. Do not include markdown code fences or explanations\n");
        prompt.push_str("4. Ensure the code compiles cleanly with `cargo check`\n");
        prompt.push_str("5. Preserve the original code structure and style where possible\n");

        prompt
    }

    /// Generate hints based on error categories
    fn generate_category_hints(&self) -> Vec<String> {
        let mut hints = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for error in &self.errors {
            if seen.contains(&error.category) {
                continue;
            }
            seen.insert(error.category);

            match error.category {
                ErrorCategory::Lifetime => {
                    hints.push("Consider adding explicit lifetime parameters".to_string());
                    hints.push(
                        "Try using owned types (String instead of &str) if lifetimes are complex"
                            .to_string(),
                    );
                    hints.push("Check if Clone can simplify ownership".to_string());
                }
                ErrorCategory::BorrowChecker => {
                    hints.push("Split borrows across different scopes".to_string());
                    hints.push("Consider using .clone() to avoid borrow conflicts".to_string());
                    hints.push(
                        "Restructure to avoid simultaneous mutable and immutable borrows"
                            .to_string(),
                    );
                }
                ErrorCategory::TypeMismatch => {
                    hints.push("Check the expected vs actual types carefully".to_string());
                    hints.push("Use .into() or .as_ref() for conversions".to_string());
                    hints.push("Verify generic type parameters".to_string());
                }
                ErrorCategory::TraitBound => {
                    hints.push("Add the required trait bound to generic parameters".to_string());
                    hints.push("Check if the type implements the required trait".to_string());
                    hints.push("Consider using dyn Trait or impl Trait".to_string());
                }
                ErrorCategory::Async => {
                    hints.push("Ensure async functions are awaited".to_string());
                    hints.push(
                        "Check that types are Send + Sync if crossing thread boundaries"
                            .to_string(),
                    );
                    hints.push("Use Pin<Box<dyn Future>> for dynamic futures".to_string());
                }
                ErrorCategory::ImportResolution => {
                    hints.push("Add the missing use statement".to_string());
                    hints.push("Check the crate name in Cargo.toml".to_string());
                    hints.push("Verify the module path is correct".to_string());
                }
                _ => {}
            }
        }

        hints
    }
}

/// Builder for code generation prompts
pub struct GenerationPromptBuilder {
    description: String,
    signature: Option<String>,
    examples: Vec<String>,
    constraints: Vec<String>,
    style_guide: Option<String>,
}

impl GenerationPromptBuilder {
    /// Create a new builder
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            signature: None,
            examples: Vec::new(),
            constraints: Vec::new(),
            style_guide: None,
        }
    }

    /// Set the function signature
    pub fn with_signature(mut self, sig: impl Into<String>) -> Self {
        self.signature = Some(sig.into());
        self
    }

    /// Add an example
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.examples.push(example.into());
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: impl Into<String>) -> Self {
        self.constraints.push(constraint.into());
        self
    }

    /// Set style guide
    pub fn with_style(mut self, style: impl Into<String>) -> Self {
        self.style_guide = Some(style.into());
        self
    }

    /// Build the prompt string
    pub fn build(&self) -> String {
        let mut prompt = String::new();

        prompt.push_str("# Rust Code Generation\n\n");

        prompt.push_str("## Task Description\n\n");
        prompt.push_str(&self.description);
        prompt.push_str("\n\n");

        if let Some(sig) = &self.signature {
            prompt.push_str("## Function Signature\n\n```rust\n");
            prompt.push_str(sig);
            prompt.push_str("\n```\n\n");
        }

        if !self.examples.is_empty() {
            prompt.push_str("## Examples\n\n");
            for (i, example) in self.examples.iter().enumerate() {
                prompt.push_str(&format!("### Example {}\n\n", i + 1));
                prompt.push_str(example);
                prompt.push_str("\n\n");
            }
        }

        if !self.constraints.is_empty() {
            prompt.push_str("## Constraints\n\n");
            for constraint in &self.constraints {
                prompt.push_str(&format!("- {}\n", constraint));
            }
            prompt.push('\n');
        }

        if let Some(style) = &self.style_guide {
            prompt.push_str("## Style Guide\n\n");
            prompt.push_str(style);
            prompt.push_str("\n\n");
        }

        prompt.push_str("## Requirements\n\n");
        prompt.push_str("1. Write idiomatic, safe Rust code\n");
        prompt.push_str("2. Handle all error cases appropriately (use Result or Option)\n");
        prompt.push_str("3. Include necessary imports\n");
        prompt.push_str("4. Code must compile with `cargo check`\n");
        prompt.push_str("5. Return ONLY the code, no explanations\n");

        prompt
    }
}

/// System prompts for different model tiers
pub struct SystemPrompts;

impl SystemPrompts {
    /// System prompt for fast model (Strand)
    pub fn fast() -> &'static str {
        r#"You are an expert Rust coder. Write clean, idiomatic Rust code that compiles.
Focus on correctness first, then clarity. Use proper error handling with Result and Option.
Return ONLY the code, no explanations or markdown formatting."#
    }

    /// System prompt for specialized model (Hydra)
    pub fn specialized() -> &'static str {
        r#"You are HydraCoder, a specialized Rust code generator trained on 180k+ Rust samples.
Your expertise includes:
- Complex lifetime and borrowing patterns
- Async/await with Tokio and futures
- Trait implementations and generic programming
- Error handling patterns (thiserror, anyhow)

Generate idiomatic, zero-cost-abstraction Rust code.
Prioritize compile-time safety and performance.
Return ONLY the code, no explanations or markdown formatting."#
    }

    /// System prompt for reasoning model (OR1)
    pub fn reasoning() -> &'static str {
        r#"You are an expert Rust architect with deep knowledge of:
- Ownership, borrowing, and lifetime patterns
- Async/await and the Tokio ecosystem
- Error handling strategies
- Type-state patterns and compile-time guarantees
- Zero-cost abstractions and performance optimization
- Memory safety principles

Think through the problem carefully. Consider:
1. What are the ownership requirements?
2. What lifetimes are needed?
3. Are there any borrow conflicts?
4. What error cases must be handled?

Then provide the corrected code.
Return the code without markdown formatting."#
    }

    /// Get system prompt for a tier
    pub fn for_tier(tier: &super::task_classifier::ModelTier) -> &'static str {
        match tier {
            super::task_classifier::ModelTier::Fast => Self::fast(),
            super::task_classifier::ModelTier::Specialized => Self::specialized(),
            super::task_classifier::ModelTier::Reasoning => Self::reasoning(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_prompt_builder() {
        let prompt = FixPromptBuilder::new("fn main() { let x: i32 = \"hello\"; }")
            .with_context("This is a simple test")
            .build();

        assert!(prompt.contains("# Rust Compilation Error Fix"));
        assert!(prompt.contains("fn main()"));
        assert!(prompt.contains("This is a simple test"));
    }

    #[test]
    fn test_generation_prompt_builder() {
        let prompt = GenerationPromptBuilder::new("Implement a function to add two numbers")
            .with_signature("fn add(a: i32, b: i32) -> i32")
            .with_constraint("Must not overflow")
            .build();

        assert!(prompt.contains("add two numbers"));
        assert!(prompt.contains("fn add(a: i32, b: i32)"));
        assert!(prompt.contains("Must not overflow"));
    }

    #[test]
    fn test_system_prompts() {
        assert!(SystemPrompts::fast().contains("idiomatic"));
        assert!(SystemPrompts::specialized().contains("HydraCoder"));
        assert!(SystemPrompts::reasoning().contains("architect"));
    }
}
