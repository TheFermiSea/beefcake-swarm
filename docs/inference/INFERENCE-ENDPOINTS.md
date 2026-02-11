# AI Inference Endpoints

Local LLM inference on the beefcake2 cluster via llama.cpp.

## Quick Start

```bash
# Fast tier (14B, single GPU) - default for most tasks
curl -sf https://slurm-ctl.tailc46cd0.ts.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "strand-rust-coder-14b-q8_0",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Reasoning tier (72B distributed) - complex architecture decisions
curl -sf https://slurm-ctl.tailc46cd0.ts.net/reasoning/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OR1-Behemoth",
    "messages": [{"role": "user", "content": "Design a distributed cache..."}],
    "max_tokens": 500
  }'
```

## Endpoints

| Tier | Tailscale URL | Port | GPUs | Use Case |
|------|---------------|------|------|----------|
| **Fast** | `https://slurm-ctl.tailc46cd0.ts.net/` | 8080 | 1 | Code completion, quick fixes |
| **Reasoning** | `https://slurm-ctl.tailc46cd0.ts.net/reasoning/` | 8081 | 3 | Complex reasoning, architecture |

### Internal Network (from cluster nodes)

```bash
# Fast tier
curl http://10.0.0.5:8080/v1/chat/completions ...

# Reasoning tier
curl http://10.0.0.5:8081/v1/chat/completions ...
```

## API Reference

OpenAI-compatible API. Full documentation: https://platform.openai.com/docs/api-reference/chat

### Chat Completions

```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "strand-rust-coder-14b-q8_0",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Your prompt here"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

### List Available Models

```bash
curl -sf https://slurm-ctl.tailc46cd0.ts.net/v1/models | jq
```

### Health Check

```bash
curl -sf https://slurm-ctl.tailc46cd0.ts.net/health | jq
```

## Model Tiers

### Fast Tier (Router Mode)

- **Models**: Any `.gguf` file in `/scratch/ai/models/`
- **Auto-load**: Models load on-demand based on `model` field in request
- **Capacity**: 2 models in VRAM simultaneously (LRU eviction)
- **Throughput**: 25-45 tok/s, <1s TTFT
- **Node**: vasp-02 (dedicated)
- **Always running**: Auto-starts when no VASP jobs in queue

**Available models** (check with `/v1/models`):
- `strand-rust-coder-14b-q8_0` - Rust/code focused

### Reasoning Tier (Distributed)

- **Model**: OR1-Behemoth 72B Q4_K_M (~45GB)
- **Architecture**: Layer parallelism across 3x V100S via RPC (vasp-01 head + vasp-02,03 workers)
- **Context**: 128K tokens (131072) with q4_0 quantized KV cache
- **Throughput**: ~22 tok/s prompt, ~13 tok/s generation
- **Nodes**: vasp-01 (head, 26 layers) + vasp-02,03 (RPC workers, 27 layers each)
- **Auto-start**: Daemon manages both tiers simultaneously

**Configuration** (optimized 2026-01-31):
- `--split-mode layer` + `--tensor-split 26,27,27` (bypasses head divisibility for 3 GPUs)
- NUMA binding for GPU/InfiniBand locality
- Matched batch sizes (512/512) for reduced network round trips
- Memory locking (--mlock) prevents model swap
- KV cache: q4_0 quantized (~23GB distributed across GPUs)
- Memory: 200GB SLURM limit per node

**Important**: OR1-Behemoth is a reasoning model that outputs chain-of-thought in `reasoning_content` field. Use `max_tokens: 500+` for complete responses.

```json
{
  "choices": [{
    "message": {
      "content": "",
      "reasoning_content": "Let me think about this..."
    }
  }]
}
```

**Note**: Both tiers can run simultaneously since they use different nodes.

```bash
# Start reasoning tier
ssh slurm-ctl 'sbatch /cluster/shared/scripts/llama-cpp/run-72b-distributed.slurm'

# Check status
ssh slurm-ctl 'squeue -u root --name=llama-72b'
```

## Tailscale Configuration

### Current Setup

Services are exposed via `tailscale serve` on slurm-ctl:

```
https://slurm-ctl.tailc46cd0.ts.net/
|-- /           proxy http://127.0.0.1:8080  (fast tier)
|-- /reasoning/ proxy http://127.0.0.1:8081  (reasoning tier)
```

### Enable Service Discovery

To see these endpoints in Tailscale admin console:

1. Go to https://login.tailscale.com/admin/services
2. Click "Discovered" tab
3. Enable "Endpoint collection" if not already enabled
4. Services will auto-appear when running

### Access Control

By default, all devices on your tailnet can access these endpoints. To restrict:

1. Edit ACLs at https://login.tailscale.com/admin/acls
2. Add rules for the slurm-ctl device or specific ports

Example ACL to restrict AI endpoints to specific users:
```json
{
  "grants": [{
    "src": ["group:ai-users"],
    "dst": ["slurm-ctl"],
    "app": {
      "tailscale.com/cap/services": [{
        "ports": [8080, 8081]
      }]
    }
  }]
}
```

## SLURM Integration

### Preemption

Both tiers run as preemptible jobs:
- **QoS**: `ai_opportunistic` (priority 100)
- **Preempted by**: `vasp_priority` (priority 1000)
- **Grace period**: 30 seconds (SIGTERM warning)
- **Behavior**: Jobs are **requeued**, not killed

When a VASP job needs GPUs:
1. AI job receives SIGTERM
2. Server shuts down gracefully (30s)
3. AI job goes back to queue
4. VASP runs
5. AI job auto-restarts when resources free

### Manual Job Management

```bash
# Check running AI jobs
ssh slurm-ctl 'squeue --name=llama-14b,llama-72b'

# Cancel AI job
ssh slurm-ctl 'scancel --name=llama-14b'

# Submit fast tier manually
ssh slurm-ctl 'sbatch /cluster/shared/scripts/llama-cpp/run-14b.slurm'

# Submit reasoning tier
ssh slurm-ctl 'sbatch /cluster/shared/scripts/llama-cpp/run-72b-distributed.slurm'
```

## Troubleshooting

### "Connection refused"

```bash
# Check if job is running
ssh slurm-ctl 'squeue --name=llama-14b'

# Check endpoint files
ssh slurm-ctl 'ls -la /cluster/shared/ai/endpoints/'

# Check nginx upstream
ssh slurm-ctl 'cat /etc/nginx/ai-inference/upstream-fast.conf'
```

### "Model not found"

```bash
# List available models
curl -sf https://slurm-ctl.tailc46cd0.ts.net/v1/models | jq -r '.data[].id'

# Check models directory
ssh slurm-ctl 'ssh vasp-02 "ls -lh /scratch/ai/models/*.gguf"'
```

### Slow responses

- Fast tier: Expected 25-45 tok/s
- Reasoning tier: Expected 5-12 tok/s (RPC overhead)
- Cold start: First request loads model (~10-30s)

### Job not starting

```bash
# Check SLURM queue
ssh slurm-ctl 'squeue -p gpu_ai'

# Check for VASP jobs (they have priority)
ssh slurm-ctl 'squeue -p gpu_vasp'

# Check job logs
ssh slurm-ctl 'ls -lt /scratch/ai/logs/ | head'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Tailnet                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ https://slurm-ctl.tailc46cd0.ts.net                 │   │
│  │   /           → nginx :8080 → fast tier             │   │
│  │   /reasoning/ → nginx :8081 → reasoning tier        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    slurm-ctl (10.0.0.5)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ tailscale    │  │    nginx     │  │ endpoint updater │  │
│  │ serve        │──│  :8080/:8081 │──│ (30s interval)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   vasp-01     │     │   vasp-02     │     │   vasp-03     │
│   V100S GPU   │     │   V100S GPU   │     │   V100S GPU   │
│               │     │               │     │               │
│ Fast: :8080   │     │ Fast: :8080   │     │ Fast: :8080   │
│ (router mode) │     │ (router mode) │     │ (router mode) │
│               │     │               │     │               │
│ 72B HEAD :8081│────►│ 72B RPC worker│◄────│ 72B RPC worker│
│  26 layers    │     │  27 layers    │     │  27 layers    │
│               │     │    :50052     │     │    :50052     │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `/cluster/shared/scripts/llama-cpp/run-14b.slurm` | Fast tier SLURM job |
| `/cluster/shared/scripts/llama-cpp/run-72b-distributed.slurm` | Reasoning tier job |
| `/cluster/shared/ai/endpoints/*.json` | Active endpoint registry |
| `/etc/nginx/sites-available/ai-inference` | Nginx proxy config |
| `/usr/local/bin/ai-inference-upstream.sh` | Dynamic upstream updater |

## RPC Distributed Inference - Lessons Learned (2026-01-31)

This section documents findings from extensive stress testing and debugging of llama.cpp RPC distributed inference.

### Hardware Configuration

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | V100S 32GB | Per node |
| System RAM | 384GB | Per node |
| CPU | 40 cores | Per node |
| Network | InfiniBand HDR100 | RPC uses TCP over IPoIB |

### Issues Discovered

#### 1. SLURM Memory Limit Too Low (CRITICAL)

**Symptom:** RPC worker killed by SIGKILL at 25-40K tokens
```
*** STEP CANCELLED DUE to SIGNAL Killed ***
recv failed (bytes_recv=0, size_to_recv=8)
Remote RPC server crashed or returned malformed response
```

**Root Cause:** Original `--mem=64G` was insufficient:
- Model weights: ~45GB (loaded into page cache via prefetch)
- KV cache: ~11.5GB
- Activation buffers: Variable, spikes during batch processing
- Total > 64GB cgroup limit → SLURM SIGKILL

**Fix:** Increased to `--mem=200G`

#### 2. GPU VRAM Limits Computation Buffers

**Symptom:** Crashes at 40K+ tokens even with 200G RAM limit

**Root Cause:** V100S 32GB VRAM per GPU
- Model shard: ~22GB
- KV cache shard: ~6GB
- Remaining: ~4GB for computation buffers
- Large batches during 40K+ token prefill exceed this

**GPU Memory Profile (vasp-03 RPC worker):**
```
Baseline: 22574 MiB (model + KV)
Peak:     28708 MiB (during prefill)
Limit:    32768 MiB
```

**Mitigation:** Reduced batch size from 1024 to 512, practical limit ~32K tokens

#### 3. llama.cpp RPC is "Proof of Concept"

Per llama.cpp documentation, RPC is:
> "proof-of-concept… fragile"

Issues encountered:
- No timeout/buffer tuning options
- TCP-only (no native InfiniBand RDMA)
- Hard abort on any socket failure
- No graceful degradation

#### 4. Health Monitor TCP Probe False Positives (CRITICAL)

**Symptom:** 72B job dies after ~6 minutes with "Worker DEAD" messages
```
WARNING: Worker vasp-02:50052 is now DEAD
CRITICAL: Only 0 workers alive (minimum: 2)
Initiating graceful shutdown due to insufficient workers
```

**Root Cause:** llama.cpp RPC uses persistent connections. Once the head node connects,
the RPC workers hold that connection and stop accepting new TCP connections on the listening port.
The health monitor's TCP probe (`echo >/dev/tcp/$node/$RPC_PORT`) returns failure even though
the workers are actually running and processing requests correctly.

**Fix:** Changed health monitor to use process-based checking (verify srun PIDs are alive)
instead of TCP port probing. Added separate `check_worker_port()` for startup validation
where TCP probe is appropriate.

### Optimizations Applied

```bash
# SLURM job configuration
#SBATCH --mem=200G              # Was 64G - caused SIGKILL

# Server arguments
--batch-size 512                # Balanced for network efficiency
--ubatch-size 512               # Match batch size for layer split
--cache-type-k q4_0             # Quantized KV cache (required for 128K)
--cache-type-v q4_0
--mlock                         # Lock model in memory, prevent swap
--parallel 1                    # Single slot for max context
--timeout 600                   # Increased for large prompts
# NOTE: DO NOT use --flash-attn on V100 (causes crash)
```

### Tested Performance Limits (Updated 2026-01-31)

**Layer Split Mode (3 GPUs, 96GB total VRAM):**

| Context Size | Result | Prompt tok/s | Gen tok/s |
|--------------|--------|--------------|-----------|
| 64K tokens | ✅ Stable | 22.5 | 12.7 |
| 128K tokens | ✅ Stable | 22.5 | 12.7 |

**Key Breakthrough:** `--split-mode layer` distributes whole layers (26,27,27) instead of tensor slices, bypassing the head divisibility requirement (64 heads ÷ 3 = invalid for standard TP).

**Optimizations Applied:**
- NUMA binding on head node for GPU/InfiniBand affinity
- Matched batch sizes (512/512) reduces network round trips by 50%
- Memory locking (--mlock) prevents model swap
- q4_0 KV cache quantization enables 128K in 96GB total VRAM

**Do NOT use on V100:**
- `--flash-attn` - Causes server crash on SM70 (no FA2 support, SDPA fallback broken)

### Debugging Commands

```bash
# Check SLURM accounting for kill reason
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,MaxVMSize,ReqMem -P

# Monitor GPU memory during inference
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 2

# Check kernel OOM messages
dmesg -T | grep -iE "oom|kill|xid"

# Enable RPC debug (set in environment before starting)
export GGML_RPC_DEBUG=1

# Check RPC worker logs
tail -f /cluster/shared/ai/logs/<JOBID>/rpc-vasp-03.log
```

### Key Takeaways

1. **SLURM memory limits matter** - Set `--mem` high enough for model + KV + buffers
2. **Layer split unlocks 3-GPU** - `--split-mode layer` bypasses head divisibility requirement
3. **NUMA binding critical** - Bind to GPU-local NUMA node for optimal IPoIB/GPU affinity
4. **128K context achieved** - 3x V100S (96GB) with q4_0 KV cache supports full 128K tokens
5. **Performance** - 22.5 tok/s prompt, 12.7 tok/s generation on distributed 72B model
6. **Health monitoring** - Use process-based checks (PIDs), not TCP probes (llama.cpp RPC uses persistent connections)

## See Also

- [VASP User Guide](../hpc/VASP-USER-GUIDE.md) - VASP has GPU priority
- [SLURM Configuration](../infrastructure/SLURM-CONFIGURATION.md) - Queue management
- [Tailscale Services](https://tailscale.com/kb/1100/services) - Service discovery
- [llama.cpp RPC README](https://github.com/ggerganov/llama.cpp/blob/master/tools/rpc/README.md) - Official RPC docs
- [vLLM Documentation](https://docs.vllm.ai/) - Production inference server
