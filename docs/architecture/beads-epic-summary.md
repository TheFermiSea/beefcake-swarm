# Beads Epic Structure Summary

**Epic ID**: beefcake2-lhr0
**Title**: Distributed OR1-Behemoth 72B Inference Cluster - Production Deployment
**Priority**: P0 (Highest)
**Status**: OPEN

## Epic Overview

Deploy a production-grade distributed inference cluster running OR1-Behemoth 72B model across 3×Tesla V100S nodes using llama.cpp with RPC backend for pipeline-parallel execution.

### Critical Technical Decisions

1. **Model Quantization**: Q8_0 (77.3 GB) over Q4_K_M (47.4 GB)
   - Rationale: V100S lacks INT4 tensor cores → Q4 de-quantization bottleneck
   - Expected impact: +1.5-2× throughput improvement
   - Memory fit: 77.3 GB in 96 GB VRAM (6 GB/GPU headroom for KV cache)

2. **Service Management**: Systemd services over nohup scripts
   - Rationale: Rocky Linux 8 systemd-logind cleans up user scope on SSH disconnect
   - Impact: Reliable lifecycle, auto-restart on failure, survives disconnects

3. **Parallelism Configuration**: `--parallel 1` baseline
   - Rationale: Pipeline-parallel RPC with high parallelism increases bubble time
   - Impact: Stable baseline, reduced network contention, tunable later

## Task Breakdown

### Phase 1: Model Acquisition (2 tasks)

| Task ID | Title | Priority | Status |
|----------|-------|--------|
| beefcake2-7c3h | Switch from Q4_K_M to Q8_0 | P2 | READY |
| beefcake2-tgpq | Deploy llama.cpp Inference Engine | P1 | READY |

**Description**:
- Stop current Q4_K_M download on node1
- Download Q8_0 parts (38.7 GB + 38.6 GB)
- Combine into single GGUF file (77.3 GB)
- Verify file integrity

**Success Criteria**: Q8_0 model exists and is 77.3 GB

### Phase 2: Systemd Service Deployment (2 tasks)

| Task ID | Title | Priority | Status |
|----------|-------|--------|
| beefcake2-hkfl | RPC Worker Services (Nodes 2 & 3) | P2 | READY |
| beefcake2-vad2 | Head Node Service (Node 1) | P2 | READY |

**Dependencies**:
- beefcake2-0xyf (Launch RPC Workers) depends on beefcake2-hkfl
- beefcake2-r5ey (Launch Head Node) depends on beefcake2-vad2

**Description**:
- Create `/etc/systemd/system/llama-rpc.service` on nodes 2 & 3
- Create `/etc/systemd/system/llama-head.service` on node1
- Configure environment: `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES=0`
- Set `Restart=always`, `RestartSec=5` (workers) or `10` (head)
- Set `LimitNOFILE=65535`

**Success Criteria**: All systemd services created and enabled

### Phase 3: Service Launch & Verification (3 tasks)

| Task ID | Title | Priority | Status | Dependencies |
|----------|-------|--------|--------------|
| beefcake2-0xyf | Launch RPC Workers | P2 | READY | beefcake2-hkfl |
| beefcake2-r5ey | Launch Head Node | P2 | READY | beefcake2-7c3h, beefcake2-vad2, beefcake2-hkfl |
| beefcake2-eipz | Basic API Health Check | P2 | READY | beefcake2-0xyf, beefcake2-r5ey |

**Description**:
- Start RPC workers: `systemctl start llama-rpc`
- Verify port 50052 listening on both workers
- Start head node: `systemctl start llama-head` (after model download)
- Check service status: `systemctl status llama-head`
- Test API endpoint: `curl http://10.0.0.31:8000/v1/chat/completions`

**Success Criteria**: All services running and API responding

### Phase 4: Distributed Inference Verification (2 tasks)

| Task ID | Title | Priority | Status | Dependencies |
|----------|-------|--------|--------------|
| beefcake2-0qj9 | Verify Distributed GPU Usage | P2 | READY | beefcake2-eipz |
| beefcake2-0o50 | Performance Baseline Testing | P2 | READY | beefcake2-0qj9 |

**Description**:
- Monitor all 3 GPUs during inference (should all show >50% utilization)
- Verify memory usage: ~26 GB/GPU for Q8_0 model
- Run baseline throughput tests (short, long, code generation prompts)
- Measure tokens/second for each test
- Document baseline metrics

**Success Criteria**: All 3 GPUs active, throughput >10 tokens/s

### Phase 5: Performance Tuning (2 tasks)

| Task ID | Title | Priority | Status | Dependencies |
|----------|-------|--------|--------------|
| beefcake2-2io9 | Parallelism Tuning | P2 | READY | beefcake2-0o50 |
| beefcake2-4rca | Context Size Tuning (Optional) | P2 | READY | beefcake2-0o50 |

**Description**:
- Test `--parallel 3` vs `--parallel 1` configuration
- Measure throughput differences
- Analyze GPU utilization and network contention
- Select optimal configuration and update service
- Optional: Test `--ctx-size 8192` vs 4096

**Success Criteria**: Optimal parallelism and context size selected

### Phase 6: Monitoring & Observability (5 tasks)

| Task ID | Title | Priority | Status | Dependencies |
|----------|-------|--------|--------------|
| beefcake2-43pr | GPU Monitoring Dashboard | P2 | READY | beefcake2-r5ey, beefcake2-0xyf |
| beefcake2-yr00 | Service Monitoring Dashboard | P2 | READY | - |
| beefcake2-b8kj | Log Aggregation | P2 | READY | - |
| beefcake2-z7r5 | Performance Metrics Collection | P2 | READY | - |
| beefcake2-l7q2 | Documentation Updates | P2 | READY | beefcake2-0o50 |

**Description**:
- Create monitoring scripts: `/usr/local/bin/llama-cluster-monitor`
- Create service status scripts: `/usr/local/bin/llama-service-status`
- Configure log rotation in journald
- Create metrics collection scripts
- Update all documentation with monitoring procedures

**Success Criteria**: All monitoring systems operational, docs updated

### Phase 7: Technical Decision Documentation (3 tasks)

| Task ID | Title | Priority | Status | Dependencies |
|----------|-------|--------|--------------|
| beefcake2-x6t1 | Decision Log Creation | P2 | READY | - |
| beefcake2-jyru | Rationale Documentation | P2 | READY | - |
| beefcake2-l7q2 | Updates to Documentation | P2 | READY | beefcake2-0o50 |

**Description**:
- Create `/opt/docs/architecture-decisions.md`
- Document Q8_0 vs Q4_K_M decision with performance data
- Document systemd vs nohup decision
- Document all tradeoffs and alternatives
- Link decision log to main guides

**Success Criteria**: All architectural decisions documented

## Dependency Flow

```
Phase 1 (Model)
  beefcake2-7c3h (Switch to Q8_0)
    ↓
Phase 2 (Services)
  beefcake2-hkfl (RPC Workers) → beefcake2-0xyf (Launch RPC Workers)
  beefcake2-vad2 (Head Service) → beefcake2-r5ey (Launch Head)
    ↓
Phase 3 (Launch & Verify)
  beefcake2-0xyf (Launch Workers) ✓
  beefcake2-r5ey (Launch Head) → beefcake2-eipz (Health Check)
                                    ↓
Phase 4 (Verification)
  beefcake2-eipz (Health Check) → beefcake2-0qj9 (Verify GPU Usage)
                                        ↓
                                  beefcake2-0o50 (Baseline Testing)
                                        ↓
Phase 5 (Tuning)
  beefcake2-0o50 (Baseline) → beefcake2-2io9 (Parallelism Tuning)
                               → beefcake2-4rca (Context Tuning)
                               → beefcake2-l7q2 (Documentation Updates)
```

## Success Criteria for Epic

1. ✅ All 3 nodes have llama.cpp binaries deployed (v7760)
2. ✅ Q8_0 model downloaded (77.3 GB) and verified
3. ✅ RPC workers running as systemd services on nodes 2 & 3
4. ✅ Head server running as systemd service on node1
5. ✅ Distributed inference functional with all 3 GPUs active (>50% utilization)
6. ✅ Throughput >10 tokens/s (target: 15-20 tokens/s)
7. ✅ Services auto-restart on failure
8. ✅ Time-to-first-token <5 seconds
9. ✅ Monitoring dashboards operational
10. ✅ All decisions documented

## Performance Targets

| Metric | Target | Acceptable | Critical |
|--------|---------|------------|----------|
| Throughput | 15-20 tokens/s | >10 tokens/s | <8 tokens/s |
| Time to First Token | <2 seconds | <5 seconds | >10 seconds |
| GPU Utilization | 70-90% | >50% | <40% |
| Memory/GPU | 26-28 GB | <30 GB | >30 GB |
| Network (IB) | >50 Gbps | >30 Gbps | <20 Gbps |

## Next Steps

1. Start with **beefcake2-7c3h** (Switch from Q4_K_M to Q8_0)
2. Download Q8_0 model (77.3 GB, ~20-25 minutes)
3. Deploy systemd services (**beefcake2-hkfl**, **beefcake2-vad2**)
4. Launch and verify services (**beefcake2-0xyf**, **beefcake2-r5ey**, **beefcake2-eipz**)
5. Verify distributed GPU usage and baseline performance (**beefcake2-0qj9**, **beefcake2-0o50**)
6. Tune and monitor (**beefcake2-2io9**, **beefcake2-4rca**, Phase 6 tasks)

## Documentation References

- **Production Guide**: `distributed-llama-production-guide.md`
- **Deployment Summary**: `deployment-summary.md`
- **Original Progress**: `distributed-llama-progress.md`
- **Beads Epic**: `bd show beefcake2-lhr0`

## Commands

```bash
# Show epic details
bd show beefcake2-lhr0

# Show dependency tree
bd dep tree beefcake2-lhr0

# List ready tasks
bd ready

# Claim work
bd update <task-id> --status in_progress

# Complete task
bd close <task-id> --reason "Completed"

# View all tasks in epic
bd list --issue-type task --limit 0
```

---

**Created**: January 16, 2026
**Total Tasks**: 17 (excluding epic)
**Total Estimated Time**: ~8-12 hours (including download time)
