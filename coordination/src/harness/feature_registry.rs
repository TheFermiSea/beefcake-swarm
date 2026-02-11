//! Feature registry management
//!
//! Handles loading, saving, and querying the features.json registry.

use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::types::{FeatureSpec, FeatureSummary};
use std::path::Path;

/// Feature registry manager
pub struct FeatureRegistry {
    features: Vec<FeatureSpec>,
    path: std::path::PathBuf,
}

impl FeatureRegistry {
    /// Load registry from file
    pub fn load(path: impl AsRef<Path>) -> HarnessResult<Self> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(HarnessError::registry_not_found(&path));
        }

        let content = std::fs::read_to_string(&path)?;
        let features: Vec<FeatureSpec> = serde_json::from_str(&content)
            .map_err(|e| HarnessError::invalid_registry(e.to_string()))?;

        Ok(Self { features, path })
    }

    /// Load registry with automatic recovery from backup if corrupted
    ///
    /// Recovery strategy:
    /// 1. Try loading from primary path
    /// 2. If corrupted, try loading from .backup file
    /// 3. If backup also fails, return empty registry
    pub fn load_with_recovery(path: impl AsRef<Path>) -> HarnessResult<Self> {
        let path = path.as_ref().to_path_buf();
        let backup_path = path.with_extension("json.backup");

        // Try primary file first
        match Self::load(&path) {
            Ok(registry) => return Ok(registry),
            Err(HarnessError::RegistryNotFound { .. }) => {
                // File doesn't exist - check backup
            }
            Err(HarnessError::InvalidRegistry { message }) => {
                // Corrupted - log and try backup
                eprintln!(
                    "Warning: Registry at {:?} is corrupted ({}), trying backup...",
                    path, message
                );
            }
            Err(e) => return Err(e),
        }

        // Try backup file
        if backup_path.exists() {
            match Self::load(&backup_path) {
                Ok(mut registry) => {
                    eprintln!("Recovered registry from backup: {:?}", backup_path);
                    registry.path = path.clone();
                    // Save recovered registry to primary location
                    if let Err(e) = registry.save() {
                        eprintln!("Warning: Failed to save recovered registry: {}", e);
                    }
                    return Ok(registry);
                }
                Err(e) => {
                    eprintln!("Warning: Backup also corrupted: {}", e);
                }
            }
        }

        // Both primary and backup failed - return empty registry
        if path.exists() || backup_path.exists() {
            eprintln!(
                "Warning: Creating empty registry due to corruption. \
                 Old files preserved for manual recovery."
            );
        }

        Ok(Self::empty(&path))
    }

    /// Create empty registry
    pub fn empty(path: impl AsRef<Path>) -> Self {
        Self {
            features: Vec::new(),
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Save registry atomically with automatic backup
    ///
    /// Strategy:
    /// 1. Create backup of existing file (if any)
    /// 2. Write to temp file
    /// 3. Rename temp to final (atomic on most filesystems)
    pub fn save(&self) -> HarnessResult<()> {
        let backup_path = self.path.with_extension("json.backup");
        let temp_path = self.path.with_extension("json.tmp");

        // Create backup of existing file
        if self.path.exists() {
            if let Err(e) = std::fs::copy(&self.path, &backup_path) {
                eprintln!("Warning: Failed to create backup: {}", e);
                // Continue anyway - backup is best-effort
            }
        }

        // Write to temp file
        let content = serde_json::to_string_pretty(&self.features)?;
        std::fs::write(&temp_path, &content)?;

        // Atomic rename
        std::fs::rename(&temp_path, &self.path)?;

        Ok(())
    }

    /// Validate registry integrity
    ///
    /// Checks for:
    /// - Duplicate feature IDs
    /// - Invalid dependency references
    /// - Circular dependencies
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for duplicate IDs
        let mut seen_ids = std::collections::HashSet::new();
        for feature in &self.features {
            if !seen_ids.insert(&feature.id) {
                issues.push(format!("Duplicate feature ID: {}", feature.id));
            }
        }

        // Check for invalid dependency references
        let all_ids: std::collections::HashSet<&str> =
            self.features.iter().map(|f| f.id.as_str()).collect();
        for feature in &self.features {
            for dep in &feature.depends_on {
                if !all_ids.contains(dep.as_str()) {
                    issues.push(format!(
                        "Feature '{}' references non-existent dependency: {}",
                        feature.id, dep
                    ));
                }
            }
        }

        // Check for circular dependencies
        let cycles = self.detect_cycles();
        if !cycles.is_empty() {
            issues.push(format!("Circular dependencies detected: {:?}", cycles));
        }

        issues
    }

    /// Get feature summary
    pub fn summary(&self) -> FeatureSummary {
        FeatureSummary::from_features(&self.features)
    }

    /// Get all features
    pub fn features(&self) -> &[FeatureSpec] {
        &self.features
    }

    /// Get features by status
    pub fn passing(&self) -> Vec<&FeatureSpec> {
        self.features.iter().filter(|f| f.passes).collect()
    }

    /// Get failing features
    pub fn failing(&self) -> Vec<&FeatureSpec> {
        self.features
            .iter()
            .filter(|f| !f.passes && f.last_verified.is_some())
            .collect()
    }

    /// Get pending features (not yet verified)
    pub fn pending(&self) -> Vec<&FeatureSpec> {
        self.features
            .iter()
            .filter(|f| !f.passes && f.last_verified.is_none())
            .collect()
    }

    /// Get next incomplete feature (highest priority, respecting dependencies)
    ///
    /// Returns the highest priority feature that:
    /// 1. Is not yet passing
    /// 2. Has all its dependencies satisfied (all dependencies passing)
    pub fn next_incomplete(&self) -> Option<&FeatureSpec> {
        self.features
            .iter()
            .filter(|f| !f.passes && !self.is_blocked(&f.id))
            .min_by_key(|f| f.priority)
    }

    /// Check if a feature is blocked (has unsatisfied dependencies)
    pub fn is_blocked(&self, id: &str) -> bool {
        let Some(feature) = self.find(id) else {
            return false;
        };

        for dep_id in &feature.depends_on {
            match self.find(dep_id) {
                Some(dep) if !dep.passes => return true,
                None => return true, // Missing dependency is a blocker
                _ => {}
            }
        }
        false
    }

    /// Get all blocked features (have unsatisfied dependencies)
    pub fn blocked(&self) -> Vec<&FeatureSpec> {
        self.features
            .iter()
            .filter(|f| !f.passes && self.is_blocked(&f.id))
            .collect()
    }

    /// Get features that are ready to work on (not passing, not blocked)
    pub fn ready(&self) -> Vec<&FeatureSpec> {
        self.features
            .iter()
            .filter(|f| !f.passes && !self.is_blocked(&f.id))
            .collect()
    }

    /// Detect circular dependencies in the registry
    ///
    /// Returns a list of feature IDs involved in cycles, or empty if no cycles.
    pub fn detect_cycles(&self) -> Vec<String> {
        use std::collections::HashSet;

        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        fn dfs(
            node: &str,
            features: &[FeatureSpec],
            visited: &mut HashSet<String>,
            in_stack: &mut HashSet<String>,
            path: &mut Vec<String>,
            cycles: &mut Vec<String>,
        ) {
            let node_str = node.to_string();

            if in_stack.contains(&node_str) {
                // Found a cycle - record the cycle
                if let Some(start) = path.iter().position(|n| n == node) {
                    for n in &path[start..] {
                        if !cycles.contains(n) {
                            cycles.push(n.clone());
                        }
                    }
                }
                return;
            }

            if visited.contains(&node_str) {
                return;
            }

            visited.insert(node_str.clone());
            in_stack.insert(node_str.clone());
            path.push(node_str.clone());

            if let Some(feature) = features.iter().find(|f| f.id == node) {
                for dep in &feature.depends_on {
                    dfs(dep, features, visited, in_stack, path, cycles);
                }
            }

            path.pop();
            in_stack.remove(&node_str);
        }

        for feature in &self.features {
            if !visited.contains(&feature.id) {
                let mut path = Vec::new();
                dfs(
                    &feature.id,
                    &self.features,
                    &mut visited,
                    &mut in_stack,
                    &mut path,
                    &mut cycles,
                );
            }
        }

        cycles
    }

    /// Get dependency chain for a feature (topological order)
    ///
    /// Returns features that must be completed before this one, in order.
    pub fn dependency_chain(&self, id: &str) -> Vec<&FeatureSpec> {
        use std::collections::HashSet;

        let mut result = Vec::new();
        let mut visited = HashSet::new();

        fn collect_deps<'a>(
            registry: &'a FeatureRegistry,
            id: &str,
            visited: &mut HashSet<String>,
            result: &mut Vec<&'a FeatureSpec>,
        ) {
            if visited.contains(id) {
                return;
            }
            visited.insert(id.to_string());

            if let Some(feature) = registry.find(id) {
                for dep_id in &feature.depends_on {
                    collect_deps(registry, dep_id, visited, result);
                }
                result.push(feature);
            }
        }

        collect_deps(self, id, &mut visited, &mut result);

        // Remove the target feature itself from the chain
        result.pop();
        result
    }

    /// Find feature by ID
    pub fn find(&self, id: &str) -> Option<&FeatureSpec> {
        self.features.iter().find(|f| f.id == id)
    }

    /// Find feature by ID (mutable)
    pub fn find_mut(&mut self, id: &str) -> Option<&mut FeatureSpec> {
        self.features.iter_mut().find(|f| f.id == id)
    }

    /// Mark feature as passing
    pub fn mark_passing(&mut self, id: &str) -> HarnessResult<()> {
        let feature = self
            .find_mut(id)
            .ok_or_else(|| HarnessError::feature_not_found(id))?;
        feature.mark_passing();
        Ok(())
    }

    /// Mark feature as failing
    pub fn mark_failing(&mut self, id: &str, note: impl Into<String>) -> HarnessResult<()> {
        let feature = self
            .find_mut(id)
            .ok_or_else(|| HarnessError::feature_not_found(id))?;
        feature.mark_failing(note);
        Ok(())
    }

    /// Add a new feature
    pub fn add(&mut self, feature: FeatureSpec) {
        self.features.push(feature);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::types::FeatureCategory;
    use tempfile::tempdir;

    #[test]
    fn test_load_save_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        // Create and save
        let mut registry = FeatureRegistry::empty(&path);
        registry
            .add(FeatureSpec::new("f1", FeatureCategory::Functional, "Feature 1").with_priority(1));
        registry.add(FeatureSpec::new("f2", FeatureCategory::Api, "Feature 2").with_priority(2));
        registry.save().unwrap();

        // Load and verify
        let loaded = FeatureRegistry::load(&path).unwrap();
        assert_eq!(loaded.features().len(), 2);
        assert_eq!(loaded.find("f1").unwrap().priority, 1);
    }

    #[test]
    fn test_next_incomplete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);
        registry.add(
            FeatureSpec::new("low", FeatureCategory::Functional, "Low priority").with_priority(10),
        );
        registry.add(
            FeatureSpec::new("high", FeatureCategory::Functional, "High priority").with_priority(1),
        );

        let next = registry.next_incomplete().unwrap();
        assert_eq!(next.id, "high");
    }

    #[test]
    fn test_mark_passing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);
        registry.add(FeatureSpec::new("f1", FeatureCategory::Functional, "Test"));

        assert!(!registry.find("f1").unwrap().passes);
        registry.mark_passing("f1").unwrap();
        assert!(registry.find("f1").unwrap().passes);
    }

    #[test]
    fn test_is_blocked_with_dependencies() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);

        // Feature A has no dependencies
        registry.add(FeatureSpec::new(
            "a",
            FeatureCategory::Functional,
            "Feature A",
        ));

        // Feature B depends on A
        let mut b = FeatureSpec::new("b", FeatureCategory::Functional, "Feature B");
        b.depends_on = vec!["a".to_string()];
        registry.add(b);

        // B is blocked because A is not passing
        assert!(registry.is_blocked("b"));
        assert!(!registry.is_blocked("a"));

        // Mark A as passing
        registry.mark_passing("a").unwrap();

        // Now B is not blocked
        assert!(!registry.is_blocked("b"));
    }

    #[test]
    fn test_is_blocked_missing_dependency() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);

        // Feature A depends on non-existent feature
        let mut a = FeatureSpec::new("a", FeatureCategory::Functional, "Feature A");
        a.depends_on = vec!["nonexistent".to_string()];
        registry.add(a);

        // A is blocked due to missing dependency
        assert!(registry.is_blocked("a"));
    }

    #[test]
    fn test_blocked_and_ready() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);

        // Feature A: no deps
        registry
            .add(FeatureSpec::new("a", FeatureCategory::Functional, "Feature A").with_priority(1));

        // Feature B: depends on A
        let mut b = FeatureSpec::new("b", FeatureCategory::Functional, "Feature B");
        b.depends_on = vec!["a".to_string()];
        b.priority = 2;
        registry.add(b);

        // Feature C: depends on B
        let mut c = FeatureSpec::new("c", FeatureCategory::Functional, "Feature C");
        c.depends_on = vec!["b".to_string()];
        c.priority = 3;
        registry.add(c);

        // Initially: A is ready, B and C are blocked
        let ready = registry.ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "a");

        let blocked = registry.blocked();
        assert_eq!(blocked.len(), 2);

        // Mark A passing: now B is ready, C is still blocked
        registry.mark_passing("a").unwrap();
        let ready = registry.ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "b");

        let blocked = registry.blocked();
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].id, "c");
    }

    #[test]
    fn test_next_incomplete_respects_dependencies() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);

        // Feature A: priority 10, no deps
        registry
            .add(FeatureSpec::new("a", FeatureCategory::Functional, "Feature A").with_priority(10));

        // Feature B: priority 1 (highest), but depends on A
        let mut b = FeatureSpec::new("b", FeatureCategory::Functional, "Feature B");
        b.depends_on = vec!["a".to_string()];
        b.priority = 1;
        registry.add(b);

        // Despite B having higher priority, A should be returned first
        // because B is blocked
        let next = registry.next_incomplete().unwrap();
        assert_eq!(next.id, "a");

        // Mark A passing, now B should be next
        registry.mark_passing("a").unwrap();
        let next = registry.next_incomplete().unwrap();
        assert_eq!(next.id, "b");
    }

    #[test]
    fn test_detect_cycles_no_cycles() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);

        // Linear chain: A -> B -> C
        registry.add(FeatureSpec::new("a", FeatureCategory::Functional, "A"));

        let mut b = FeatureSpec::new("b", FeatureCategory::Functional, "B");
        b.depends_on = vec!["a".to_string()];
        registry.add(b);

        let mut c = FeatureSpec::new("c", FeatureCategory::Functional, "C");
        c.depends_on = vec!["b".to_string()];
        registry.add(c);

        let cycles = registry.detect_cycles();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_detect_cycles_with_cycle() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);

        // Cycle: A -> B -> C -> A
        let mut a = FeatureSpec::new("a", FeatureCategory::Functional, "A");
        a.depends_on = vec!["c".to_string()];
        registry.add(a);

        let mut b = FeatureSpec::new("b", FeatureCategory::Functional, "B");
        b.depends_on = vec!["a".to_string()];
        registry.add(b);

        let mut c = FeatureSpec::new("c", FeatureCategory::Functional, "C");
        c.depends_on = vec!["b".to_string()];
        registry.add(c);

        let cycles = registry.detect_cycles();
        assert!(!cycles.is_empty());
        // All three should be in the cycle
        assert!(
            cycles.contains(&"a".to_string())
                || cycles.contains(&"b".to_string())
                || cycles.contains(&"c".to_string())
        );
    }

    #[test]
    fn test_dependency_chain() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);

        // Chain: A <- B <- C (C depends on B, B depends on A)
        registry.add(FeatureSpec::new("a", FeatureCategory::Functional, "A"));

        let mut b = FeatureSpec::new("b", FeatureCategory::Functional, "B");
        b.depends_on = vec!["a".to_string()];
        registry.add(b);

        let mut c = FeatureSpec::new("c", FeatureCategory::Functional, "C");
        c.depends_on = vec!["b".to_string()];
        registry.add(c);

        let chain = registry.dependency_chain("c");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].id, "a");
        assert_eq!(chain[1].id, "b");
    }

    #[test]
    fn test_dependency_chain_empty_for_no_deps() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);
        registry.add(FeatureSpec::new("a", FeatureCategory::Functional, "A"));

        let chain = registry.dependency_chain("a");
        assert!(chain.is_empty());
    }

    #[test]
    fn test_save_creates_backup() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");
        let backup_path = dir.path().join("features.json.backup");

        // Create and save initial registry
        let mut registry = FeatureRegistry::empty(&path);
        registry.add(FeatureSpec::new(
            "f1",
            FeatureCategory::Functional,
            "Feature 1",
        ));
        registry.save().unwrap();

        // No backup for first save (no existing file)
        assert!(!backup_path.exists());

        // Modify and save again
        registry.add(FeatureSpec::new(
            "f2",
            FeatureCategory::Functional,
            "Feature 2",
        ));
        registry.save().unwrap();

        // Now backup should exist
        assert!(backup_path.exists());

        // Backup should contain only f1
        let backup = FeatureRegistry::load(&backup_path).unwrap();
        assert_eq!(backup.features().len(), 1);
        assert_eq!(backup.features()[0].id, "f1");
    }

    #[test]
    fn test_load_with_recovery_from_backup() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");
        let backup_path = dir.path().join("features.json.backup");

        // Create backup with valid content
        let mut backup_registry = FeatureRegistry::empty(&backup_path);
        backup_registry.add(FeatureSpec::new(
            "backup-feature",
            FeatureCategory::Functional,
            "Backup",
        ));
        let content = serde_json::to_string_pretty(backup_registry.features()).unwrap();
        std::fs::write(&backup_path, content).unwrap();

        // Write corrupted primary file
        std::fs::write(&path, "{ not valid json }").unwrap();

        // Load with recovery should use backup
        let registry = FeatureRegistry::load_with_recovery(&path).unwrap();
        assert_eq!(registry.features().len(), 1);
        assert_eq!(registry.features()[0].id, "backup-feature");
    }

    #[test]
    fn test_load_with_recovery_returns_empty_when_all_corrupted() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");
        let backup_path = dir.path().join("features.json.backup");

        // Write corrupted primary and backup
        std::fs::write(&path, "{ not valid }").unwrap();
        std::fs::write(&backup_path, "{ also not valid }").unwrap();

        // Should return empty registry
        let registry = FeatureRegistry::load_with_recovery(&path).unwrap();
        assert!(registry.features().is_empty());
    }

    #[test]
    fn test_validate_detects_issues() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);

        // Add duplicate IDs
        registry.add(FeatureSpec::new(
            "dup",
            FeatureCategory::Functional,
            "First",
        ));
        registry.add(FeatureSpec::new(
            "dup",
            FeatureCategory::Functional,
            "Second",
        ));

        // Add feature with missing dependency
        let mut missing_dep = FeatureSpec::new("orphan", FeatureCategory::Functional, "Orphan");
        missing_dep.depends_on = vec!["nonexistent".to_string()];
        registry.add(missing_dep);

        let issues = registry.validate();
        assert!(!issues.is_empty());

        // Should detect duplicate
        assert!(issues.iter().any(|i| i.contains("Duplicate")));

        // Should detect missing dependency
        assert!(issues.iter().any(|i| i.contains("nonexistent")));
    }

    #[test]
    fn test_validate_no_issues_for_valid_registry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("features.json");

        let mut registry = FeatureRegistry::empty(&path);
        registry.add(FeatureSpec::new("a", FeatureCategory::Functional, "A"));

        let mut b = FeatureSpec::new("b", FeatureCategory::Functional, "B");
        b.depends_on = vec!["a".to_string()];
        registry.add(b);

        let issues = registry.validate();
        assert!(issues.is_empty());
    }
}
