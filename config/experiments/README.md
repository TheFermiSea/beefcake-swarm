# Experiment Profiles

Each `.toml` file defines the `candidate_variants` for adaptive experiments.
The active profile is applied by `scripts/tz-apply-experiment.sh`, which
patches the generated `tensorzero.toml` and restarts the TZ gateway.

## Files

- `normal.toml` — default production routing
- `gemma-experiment.toml` — Gemma-4-31B-it time-sliced experiment on vasp-02

## Usage

```bash
./scripts/tz-apply-experiment.sh normal           # restore default
./scripts/tz-apply-experiment.sh gemma-experiment  # activate gemma
./scripts/tz-apply-experiment.sh --current         # show active profile
```
