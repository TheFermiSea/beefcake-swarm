# Model Performance

Current variant lineup and observed performance characteristics.
Updated from TensorZero Autopilot data and dogfood run outcomes.

## Active Models

| Model | Tier | Hardware | Throughput | Context |
|-------|------|----------|------------|---------|
| GLM-4.7-Flash (30B/3B MoE) | Scout/Fast | vasp-03 V100S | ~50 tok/s | 200K |
| Qwen3.5-27B (dense) | Coder | vasp-01 V100S | ~27 tok/s | 32K |
| Devstral-Small-2-24B (dense) | Reasoning | vasp-02 V100S | ~30 tok/s | 32K |
| SERA-14B (Qwen3 backbone) | SWE Specialist | vasp-03 V100S (shared) | TBD | 8K |

## Cloud Fallback Cascade

1. gpt-5.4-mini (primary)
2. gemini-3.1-pro-high (fallback-1)
3. claude-sonnet-4-6 (fallback-2)
4. gemini-3.1-flash-lite-preview (fallback-3)

## Observations

- **GLM-4.7-Flash**: SOTA tool-calling (tau2 84.7). Excellent for scout/reviewer/fixer roles where structured output matters.
- **Qwen3.5-27B**: Reliable code generation. Primary workhorse for single-file Rust edits.
- **Devstral-24B**: Strong on planning and multi-step reasoning. Used for architecture decisions and complex multi-file changes.
- **SERA-14B**: SWE-focused fine-tune on Qwen3 backbone. 8K context is VRAM-constrained; Qwen3.5-27B fallback configured for overflow.

## TZ Variant Rankings

_This section is updated automatically from TensorZero Autopilot insights._
