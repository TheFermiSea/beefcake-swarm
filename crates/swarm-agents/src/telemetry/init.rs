#[cfg(test)]
mod tests {

    use crate::telemetry::*;


    #[test]
    fn test_prune_task_prompt_no_prune() {
        let prompt = "Task description\n---\nIteration 1\n---\nIteration 2";
        // Iteration 2, prune_after 3 → no pruning
        assert_eq!(prune_task_prompt(prompt, 2, 3, 2), prompt);
    }

    #[test]
    fn test_prune_task_prompt_prunes() {
        let prompt = "Task description\n---\nIteration 1\n---\nIteration 2\n---\nIteration 3\n---\nIteration 4";
        let pruned = prune_task_prompt(prompt, 5, 3, 2);
        assert!(pruned.contains("Task description"));
        assert!(pruned.contains("Iteration 4"));
        assert!(pruned.contains("Iteration 3"));
        assert!(!pruned.contains("Iteration 1"));
        assert!(pruned.contains("[Earlier iterations pruned"));
    }

    #[test]
    fn test_append_experiment_tsv_creates_header() {
        let dir = tempfile::TempDir::new().unwrap();
        append_experiment_tsv(
            dir.path(),
            "abc123",
            5,
            &["fmt", "clippy"],
            "keep",
            "partial progress",
        );
        let content = std::fs::read_to_string(dir.path().join("experiments.tsv")).unwrap();
        assert!(content.starts_with("timestamp\t"));
        assert!(content.contains("abc123"));
        assert!(content.contains("keep"));
    }

    #[test]
    fn test_append_experiment_tsv_appends() {
        let dir = tempfile::TempDir::new().unwrap();
        append_experiment_tsv(dir.path(), "abc", 5, &["fmt"], "keep", "first");
        append_experiment_tsv(dir.path(), "def", 3, &["fmt", "clippy"], "revert", "second");
        let content = std::fs::read_to_string(dir.path().join("experiments.tsv")).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 rows
    }

    #[test]
    fn test_append_failure_ledger() {
        let dir = tempfile::TempDir::new().unwrap();
        let entry = FailureLedgerEntry {
            tool: "edit_file".to_string(),
            error_class: "match_failure".to_string(),
            signal_traced: "old_content not found".to_string(),
            file_path: Some("src/main.rs".to_string()),
            iteration: 1,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            success: false,
        };
        append_failure_ledger(dir.path(), &entry);
        let content =
            std::fs::read_to_string(dir.path().join(".swarm-failure-ledger.jsonl")).unwrap();
        assert!(content.contains("edit_file"));
        assert!(content.contains("match_failure"));
    }

    #[test]
    fn test_failure_ledger_success_entry() {
        let dir = tempfile::TempDir::new().unwrap();
        let entry = FailureLedgerEntry {
            tool: "edit_file".to_string(),
            error_class: "anchor_edit".to_string(),
            signal_traced: "lines 10-15 replaced".to_string(),
            file_path: Some("src/lib.rs".to_string()),
            iteration: 2,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            success: true,
        };
        append_failure_ledger(dir.path(), &entry);
        let content =
            std::fs::read_to_string(dir.path().join(".swarm-failure-ledger.jsonl")).unwrap();
        let parsed: FailureLedgerEntry = serde_json::from_str(content.trim()).unwrap();
        assert!(parsed.success);
    }
}
