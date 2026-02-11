# Beefcake Swarm

Autonomous coding swarm: Rig + Gastown + Beads on HPC cluster (3x V100S).

## Quick Start

```bash
cargo build --workspace       # Build all crates
cargo test -p coordination    # Run coordination tests
cargo run -p swarm-agents     # Run orchestrator (needs inference running)
```

## Inference Endpoints (must be running via SLURM)

| Tier | Endpoint | Model | Throughput |
|------|----------|-------|------------|
| Fast (14B) | http://vasp-02:8080 | strand-rust-coder-14b-q8_0 | ~53 tok/s |
| Reasoning (72B) | http://vasp-01:8081 | or1-behemoth-q4_k_m | ~13 tok/s |

Start inference:
```bash
ssh root@10.0.0.5 "sbatch /cluster/shared/scripts/llama-cpp/run-14b.slurm"
ssh root@10.0.0.5 "sbatch /cluster/shared/scripts/llama-cpp/run-72b-distributed.slurm"
```

## Architecture

Rig agents → Gastown worktrees → Beads tracking → SLURM dispatch

### Escalation Ladder
Implementer (14B) → Integrator (72B) → Cloud Council → Human

### Crate Structure
- `crates/swarm-agents`: Rig-based orchestrator (implementer + validator loop)
- `coordination/`: Deterministic quality gates, escalation state machine, ensemble voting

## External Tools (install separately)
- `br` (beads_rust): `cargo install --git https://github.com/Dicklesworthstone/beads_rust`
- `bv` (beads_viewer): `go install github.com/Dicklesworthstone/beads_viewer@latest`
- `gastown`: `go install github.com/steveyegge/gastown@latest`

## Cluster Access
- slurm-ctl: `ssh root@10.0.0.5`
- vasp-01: `ssh root@10.0.0.20` (72B head)
- vasp-02: `ssh root@10.0.0.21` (14B fast)
- vasp-03: `ssh root@10.0.0.22` (72B RPC worker)
- ai-proxy: `ssh root@100.105.113.58`

## SLURM Rules
**ALL computational tasks MUST go through SLURM.** Never run workloads directly on compute nodes.
