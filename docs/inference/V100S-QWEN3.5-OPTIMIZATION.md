# Qwen 3.5 V100S (Volta) Optimization Guide (2026 Edition)

This guide provides a technical roadmap for developers to implement, configure, and benchmark native acceleration methods for Qwen 3.5 on NVIDIA V100S (Volta) GPUs using `llama.cpp`.

## Phase 1: Native Flash Attention (MANDATORY)
Unlike the standard Python `flash-attn` library, `llama.cpp` (GGML) includes custom CUDA kernels specifically for the Volta (SM70) architecture. This is the single most important flag for V100S performance.

**Implementation:**
Always include the `-fa` (or `--flash-attn`) flag.
```bash
--flash-attn
```

**Impact:**
*   **Memory:** Reduces VRAM usage for the KV cache by up to 50% for long contexts.
*   **Throughput:** Increases token generation speed by 20-40% on V100S.
*   **Enabler:** Native Flash Attention is a prerequisite for high-performance KV cache quantization.

## Phase 2: KV Cache Quantization (Optimized)
With `-fa` enabled, you can reclaim significant VRAM to support 32k+ contexts on the V100S (32GB). For Qwen 3.5, **`q8_0` is the gold standard** for maintaining reasoning integrity while reducing memory footprint.

**Implementation:**
```bash
--cache-type-k q8_0 --cache-type-v q8_0
```

**Why:** Using `q8_0` on V100S provides a significant memory saving with near-zero perplexity loss. Avoid `q4_0` for reasoning tasks unless VRAM is critically exhausted.

## Phase 3: High-Throughput Batch Tuning
Because Flash Attention handles large matrix multiplications efficiently on Volta's Tensor Cores, you should use **larger** batch sizes than previously recommended to maximize GPU utilization.

**Settings:**
*   **Physical Batch (`-b`):** `2048` or `4096`.
*   **Logical Batch (`-ub`):** Match your physical batch size.

```bash
-b 2048 -ub 2048
```

## Phase 4: Speculative Decoding (Optional)
For Qwen 3.5-27B, you can achieve "instant" response speeds by using a 1.5B draft model.

**Execution Command:**
```bash
./llama-cli \
    -m models/qwen3.5-27b-instruct-q8_0.gguf \
    -md models/qwen3.5-1.5b-instruct-q8_0.gguf \
    -fa -ngl 99 -ngld 99 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --draft 8
```

## Phase 5: Verification Protocol
Run `llama-bench` to verify that the Flash Attention kernels are active and providing the expected speedup.

**Benchmark Command:**
```bash
./llama-bench \
    -m models/qwen3.5-27b-instruct-q8_0.gguf \
    -p 512,2048,4096 \
    -n 128 \
    -fa 1 \
    --cache-type-k q8_0 --cache-type-v q8_0
```

**Success Metrics (V100S):**
*   **Prompt Processing (PP):** Should exceed 150 t/s.
*   **Token Generation (TG):** Should exceed 25 t/s (without speculative decoding).

## Troubleshooting Checklist:
*   **Flash Attention Warning:** If the logs show `flash_attn kernel not found`, verify your build was compiled with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`.
*   **OOM on 32k Context:** Even with `-fa`, 32k context + Q8_0 model + Q8_0 KV cache can push 32GB limits. If you OOM, drop KV cache to `q4_0` using `--cache-type-k q4_0 --cache-type-v q4_0`.
