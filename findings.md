# Investigation Findings

## 1. investigation.txt Summary
The test output shows that `config::tests::test_default_config` is failing at line 1494 in `crates/swarm-agents/src/config.rs`.

**Failure Details:**
- Test: `config::tests::test_default_config`
- Location: `crates/swarm-agents/src/config.rs:1494:9`
- Error: `assertion failed: config.fast_endpoint.url.contains("vasp-03")`
- Result: 799 passed; 1 failed

## 2. Code Analysis (lines 1480-1510)
The failing test `test_default_config` at line 1494 asserts:
```rust
assert!(config.fast_endpoint.url.contains("vasp-03"));
```

The test also checks:
- Line 1495: `config.coder_endpoint.url.contains("vasp-01")`
- Line 1496: `config.reasoning_endpoint.url.contains("vasp-02")`
- Line 1497: `config.fast_endpoint.model == "OmniCoder-9B"`
- Line 1498: `config.coder_endpoint.model == "Qwen3.5-27B"`
- Line 1499: `config.reasoning_endpoint.model == "Qwen3.5-27B"`
- Line 1500: `config.fast_endpoint.api_key == "not-needed"`

The test first clears various environment variables (lines 1480-1491) before creating a default config.

## 3. Git Diff Analysis
The git diff `HEAD~1 -- crates/swarm-agents/src/config.rs` returns empty output, indicating no changes were made to this file in the most recent commit.

## Root Cause
The test expects `config.fast_endpoint.url` to contain "vasp-03" when using default configuration, but the actual URL does not contain this string. This suggests either:
1. The default configuration has been changed to use a different endpoint URL
2. The test expectation is outdated and needs to be updated to match the current default configuration
