# CodeEvolve CF-LIBS Run 1 Report
**Date:** 2026-03-24 15:42 — 20:30 UTC (~5 hours)
**Status:** Running (Epoch 23 of 200, slow progress)

## Configuration
- Islands: 3, Ring topology, Migration interval: 20 epochs
- Models: Opus 4.6, Sonnet 4.6, Gemini 3.1 Pro, GPT-5.4, Qwen3-Coder-Next (27B), Qwen3.5-122B
- MAP-Elites: CVT with f1_score × precision features, 50 centroids
- Evaluation: 74 Aalto spectra, ALIAS-only (NO NNLS — basis library failed)
- Fitness: -(1 - F1), hard gates at recall<0.50 and precision<0.20

## Results

### Baseline
| Metric | Value |
|--------|-------|
| F1 | 0.487 |
| Precision | 0.396 |
| Recall | 0.635 |
| False Positives | 162 |
| Exact Match | 8.1% (6/74) |

### Best Solution Found (Island 2, Epoch 9)
| Metric | Value | Change |
|--------|-------|--------|
| F1 | **0.595** | **+22%** |
| Precision | **0.610** | **+54%** |
| Recall | 0.581 | -8% |
| False Positives | **62** | **-62%** |
| Exact Match | **20.3%** (15/74) | **+150%** |

### Pareto Front (Precision vs Recall tradeoffs)
| F1 | Precision | Recall | FP | Strategy |
|-----|-----------|--------|----|----------|
| 0.595 | 0.610 | 0.581 | 62 | Balanced |
| 0.595 | 0.655 | 0.545 | 48 | High precision |
| 0.588 | 0.630 | 0.551 | 54 | Balanced |
| 0.580 | 0.522 | 0.653 | 100 | High recall |
| 0.551 | 0.658 | 0.473 | 41 | Conservative |

### Per-Island Performance
| Island | Best F1 | Best Epoch | # Evals | Notes |
|--------|---------|------------|---------|-------|
| 0 | 0.506 | 13 | 12 | Stuck near baseline |
| 1 | 0.543 | 1 | 10 | Early win, never improved |
| 2 | **0.595** | 9 | 13 | Dominant — found multiple good solutions |

## Evaluation Statistics
- **Total evaluations:** 35 (12 + 10 + 13)
- **Syntax errors:** 3 (8.6%)
- **Connection failures:** 9 (LLM endpoint issues)
- **Rate limit failures:** 2 (GPT-5.4 cooldown)
- **Successful mutations:** 23 (65.7%)
- **Improving mutations:** ~10 (29% of total, 43% of successful)

## Model Performance
| Model | Calls | Notes |
|-------|-------|-------|
| Opus 4.6 | 21 | Reliable, expensive |
| GPT-5.4 | 20 | **2 cooldown failures (26h reset!), should remove** |
| Sonnet 4.6 | 19 | Reliable, good value |
| Gemini 3.1 Pro | 18 | Reliable |
| 122B (vasp-01/02) | 14 | Slow (~5 tok/s), reliable |
| 27B (vasp-03) | 13 | Fast, reliable |

## Issues Identified

### Critical
1. **NNLS data missing** — Basis library generation failed (missing xarray module). The combiner only has ALIAS features, halving available information.
2. **GPT-5.4 permanently rate-limited** — 26-hour cooldown. Wastes ~4 min per failed attempt.

### Performance
3. **Slow epoch rate** — ~12 min/epoch due to 122B model speed + barrier sync. 200 epochs = 40 hours.
4. **No improvement since Epoch 9** — 14 epochs of plateau. Search space may be too narrow with ALIAS-only.
5. **Prompt token growth** — 2K → 10K tokens as chat depth increases. No max_chat_depth set.
6. **Inference servers died mid-run** — 9 connection errors before manual restart.

### Minor
7. **Migration sent weak solution** — Island 0's F1=0.505 migrated to Island 2 (already at 0.595).
8. **Evaluation only has 74 spectra** — No ChemCam standards integrated.

## Fixes for Run 2
1. Install xarray, generate basis library, regenerate cache WITH NNLS data
2. Remove GPT-5.4, replace with gpt-5.2-codex or gpt-5.1
3. Set max_chat_depth: 3 in EVOLVE_CONFIG
4. Increase BUDGET_CONFIG.timeout_s to 180
5. Reduce num_islands to 2 (faster epochs with barrier sync)
6. Add ChemCam spectra to cache (if PDS parser ready)
7. Monitor inference server health (add health check before run)
