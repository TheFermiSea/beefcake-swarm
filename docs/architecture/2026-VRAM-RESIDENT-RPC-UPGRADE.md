# 2026 Model Ensemble Upgrade - VRAM-Resident Distributed Strategy (RPC)

**Date**: March 9, 2026
**Status**: DEPLOYED / ACTIVE
**Epic**: `beefcake-wy9r`

## Overview

Based on deep analysis of the V100S (32GB) cluster's memory bandwidth bottlenecks, we have transitioned to a **VRAM-Resident Distributed** architecture. This strategy eliminates the CPU MoE offloading bottleneck of the 397B model by keeping active weights and large context (KV cache) entirely within GPU memory paths.

## The Active Ensemble Stack

| Tier | Model | Hardware | Context | Key Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Scout / Reviewer** | **Qwen3.5-27B-Distilled** | `vasp-03` | **192K** | 100% VRAM-resident. Blazing fast prefill for codebase context. |
| **Integrator (RPC)** | **Qwen3.5-122B-A10B-MoE** | `vasp-01` + `vasp-02` | **128K** | Layer-distributed across 2 nodes (64GB VRAM). High integration throughput. |
| **The Council** | **Claude 4.6 Opus** | Cloud | 200K | Deepest reasoning fallback for complex refactors. |

## Rationale for Strategy Shift

### 1. Eliminating the CPU Bottleneck
The previous Qwen3.5-397B model offloaded ~200GB of weights to CPU RAM. Reading these weights through the 150GB/s CPU bus resulted in 10-15 minute turn latencies for large contexts. 
- The **27B model** fits in 16GB VRAM, leaving 16GB for a **192K KV Cache** on a single node. 
- The **122B model** is split via **llama.cpp RPC** across two nodes, keeping ~90% of active parameters in the 900GB/s VRAM bandwidth path.

### 2. Prompt Caching
All servers have `--prompt-cache` enabled. This allows the swarm to "remember" the codebase state between iterations, dropping turn latency from minutes to milliseconds after the initial prefill.

## Node Assignment (RPC Topology)

| Node | Model Role | Primary Model | Hardware Strategy |
| :--- | :--- | :--- | :--- |
| **vasp-03** | Scout / Reviewer | Qwen3.5-27B-Distilled | Full GPU Offload (Massive Context) |
| **vasp-01** | Integrator (Head) | Qwen3.5-122B-A10B | Head Node (26 layers + RPC client) |
| **vasp-02** | Integrator (Worker) | Qwen3.5-122B-A10B | RPC Worker (27 layers) |

## Active Infrastructure Scripts
- `inference/slurm/run-27b-256k.slurm`: Single-node 192K context server.
- `inference/slurm/run-122b-rpc.slurm`: Dual-node distributed MoE server.

---
**Updated**: March 9, 2026
**Reference**: `PROMPT_VERSION 7.0.0`
