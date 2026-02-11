# Production Deployment Summary

**Date**: January 16, 2026
**Status**: ✅ Production guide created, ready for deployment

## What Changed

Based on independent analysis, created **production-ready deployment guide** (`distributed-llama-production-guide.md`) with critical corrections:

### 1. **Switch to Q8_0 (not Q4_K_M)**

**Why**: V100S lacks INT4 tensor cores → Q4 requires slow de-quantization on CUDA cores
- Q4_K_M: 47.4 GB, ~8-12 tokens/s (bottlenecked)
- Q8_0: 77.3 GB, ~15-20 tokens/s (near memory bandwidth limit)

**Memory Fit**: Q8_0 fits in 96GB VRAM (3×32GB) with ~6GB/GPU for KV cache
**Tradeoff**: +30GB model size for +1.5-2× throughput

### 2. **Systemd Services (not nohup)**

**Why**: Rocky Linux 8 systemd-logind cleans up user scope on SSH disconnect
- nohup processes get killed when you disconnect
- Systemd services survive disconnects and auto-restart

**Services Created**:
- `llama-rpc.service` → Nodes 2 & 3 (RPC workers)
- `llama-head.service` → Node 1 (inference server)

### 3. **Corrected Launch Parameters**

| Issue | Fix |
|--------|------|
| Duplicate `-ngl 999` | Removed duplicate flag |
| `--parallel 3` | Changed to `--parallel 1` (baseline) |
| No restart policy | Added `Restart=always` |
| Missing env vars | Explicit `LD_LIBRARY_PATH` and `CUDA_VISIBLE_DEVICES` |

### 4. **InfiniBand IPs for RPC**

Already using 10.100.0.x (100 Gb/s) - this was correct.

## Next Steps (6 tasks)

1. **Stop Q4 download** - Abort current 47.4 GB download
2. **Download Q8_0** - Get 77.3 GB model (20-25 min)
3. **Deploy RPC workers** - Systemd services on nodes 2 & 3
4. **Deploy head node** - Systemd service on node 1
5. **Launch server** - Start distributed inference
6. **Benchmark Q8_0** - Measure throughput (target: 15-20 tokens/s)

## Key Commands

**Stop Q4 & Download Q8_0**:
```bash
ssh root@100.127.208.104 root@10.0.0.31 'bash -s' <<'EOF'
pkill wget
rm -f /opt/models/OR1-Behemoth.Q4_K_M.gguf
cd /opt/models
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q8_0.gguf.part1of2"
wget -c "https://huggingface.co/mradermacher/OR1-Behemoth-GGUF/resolve/main/OR1-Behemoth.Q8_0.gguf.part2of2"
cat OR1-Behemoth.Q8_0.gguf.part1of2 OR1-Behemoth.Q8_0.gguf.part2of2 > OR1-Behemoth.Q8_0.gguf
rm -f OR1-Behemoth.Q8_0.gguf.part*
ls -lh OR1-Behemoth.Q8_0.gguf
EOF
```

**Deploy RPC Workers (Node 2 & 3)**:
```bash
# Node 2
ssh -o ConnectTimeout=10 root@100.127.30.114 root@10.0.0.32 'bash -s' < /path/to/rpc-service.sh

# Node 3
ssh -o ConnectTimeout=10 root@100.68.22.98 root@10.0.0.33 'bash -s' < /path/to/rpc-service.sh
```

**Launch Head (after download)**:
```bash
ssh -o ConnectTimeout=10 root@100.127.208.104 root@10.0.0.31 'bash -s' < /path/to/head-service.sh
```

## Files Created

1. **`distributed-llama-production-guide.md`** (NEW)
   - Complete production deployment guide
   - Systemd service files
   - All commands ready to copy-paste
   - Monitoring & troubleshooting sections

2. **`distributed-llama-progress.md`** (ORIGINAL)
   - Kept for reference
   - Contains Q4 discussion (now deprecated)

## Performance Expectations

| Metric | Q4_K_M (Old) | Q8_0 (New) |
|--------|----------------|---------------|
| Throughput | 8-12 tokens/s | 15-20 tokens/s |
| Memory/GPU | 16 GB | 26 GB |
| V100 Utilization | Bottlenecked | Near max |
| Initial Load | 4s | 6s |

**Network**: Same (InfiniBand @ 100 Gb/s)

## Success Criteria

- ✅ Q8_0 model downloaded (77.3 GB)
- ✅ All 3 systemd services running
- ✅ Throughput >10 tokens/s (target: 15-20)
- ✅ All 3 GPUs >50% utilization
- ✅ Services auto-restart on failure
- ✅ API responds within 5 seconds

---

**Next Action**: Run deployment commands from `distributed-llama-production-guide.md` Phase 1-3
