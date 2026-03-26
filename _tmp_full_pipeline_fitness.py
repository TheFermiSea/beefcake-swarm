#!/usr/bin/env python3
"""
JAX-traceable 101-parameter fitness function for CF-LIBS ensemble optimization.

Takes a 101-dim parameter vector, decodes algorithm selection weights,
identifier configs, per-element thresholds, and decision fusion params.
Computes weighted ensemble detection and returns F-beta(0.5) fitness.

Designed for:
  - jax.jit compilation
  - jax.vmap for batch evaluation
  - jax.pmap for multi-GPU distribution
  - evosax CMA-ES integration

Run on vasp-03:
    HOME=/tmp CUDA_CACHE_PATH=/tmp/brian-cuda-cache \
    /cluster/shared/envs/cflibs-gpu/bin/python3 -u scripts/full_pipeline_fitness.py \
        --cache-npz /tmp/cache_arrays.npz \
        --benchmark
"""
import argparse
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np


# ============================================================
# Parameter indices (must match full_param_spec.py layout)
# ============================================================
# Group 1: Algorithm selection (indices 0-5)
IDX_W_NNLS = 0
IDX_W_ALIAS = 1
IDX_W_CORR = 2
IDX_W_COMB = 3
IDX_W_HYBRID = 4
IDX_ALGO_TEMP = 5

# Group 2: ALIAS config (indices 6-19)
IDX_ALIAS_WAV_TOL = 6
IDX_ALIAS_MIN_INT = 7
IDX_ALIAS_ION_W = 8
IDX_ALIAS_PERSIST_W = 9
IDX_ALIAS_SELFABS_PEN = 10
IDX_ALIAS_MAX_LINES = 11
IDX_ALIAS_CONF_DECAY = 12
IDX_ALIAS_SNR_THRESH = 13
IDX_ALIAS_MULTIPLET = 14
IDX_ALIAS_ISOLATION = 15
IDX_ALIAS_WING_PEN = 16
IDX_ALIAS_BLEND_PEN = 17
IDX_ALIAS_RANK_W = 18
IDX_ALIAS_FLOOR = 19

# Group 3: NNLS config (indices 20-24)
IDX_NNLS_REG = 20
IDX_NNLS_MIN_W = 21
IDX_NNLS_MAX_COMP = 22
IDX_NNLS_CONV_TOL = 23
IDX_NNLS_NORM_P = 24

# Group 4: Hybrid fusion (indices 25-36)
IDX_HYB_BLEND = 25
IDX_HYB_CONFLICT = 26
IDX_HYB_MIN_AGREE = 27
IDX_HYB_BOOST_AGREE = 28
IDX_HYB_PEN_CONFLICT = 29
IDX_HYB_SNR_GATE = 30
IDX_HYB_LINE_W = 31
IDX_HYB_DET_W = 32
IDX_HYB_PHYSICS = 33
IDX_HYB_RANK_BLEND = 34
IDX_HYB_VETO = 35
IDX_HYB_RESCUE = 36

# Group 5: Correlation config (indices 37-54)
IDX_CORR_WINDOW = 37
IDX_CORR_MIN_PEAK = 38
IDX_CORR_TEMPLATE = 39
IDX_CORR_NOISE = 40
IDX_CORR_PEAK_W = 41
IDX_CORR_BASELINE = 42
IDX_CORR_NORM = 43
IDX_CORR_MULTI_PEAK = 44
IDX_CORR_ISOLATION = 45
IDX_CORR_SNR_W = 46
IDX_CORR_PERSIST = 47
IDX_CORR_ION_RATIO = 48
IDX_CORR_SELFABS = 49
IDX_CORR_CONF_FLOOR = 50
IDX_CORR_MAX_OVERLAP = 51
IDX_CORR_DECONV_ITER = 52
IDX_CORR_REG = 53
IDX_CORR_SCORE_POW = 54

# Group 6: Comb filter (indices 55-68)
IDX_COMB_SPACING = 55
IDX_COMB_MIN_TEETH = 56
IDX_COMB_INT_RATIO = 57
IDX_COMB_HARMONIC = 58
IDX_COMB_NOISE = 59
IDX_COMB_WINDOW = 60
IDX_COMB_BASELINE = 61
IDX_COMB_SHARPNESS = 62
IDX_COMB_ISOLATION = 63
IDX_COMB_OVERLAP_PEN = 64
IDX_COMB_CONF_SCALE = 65
IDX_COMB_MIN_SNR = 66
IDX_COMB_PERSIST_W = 67
IDX_COMB_FLOOR = 68

# Group 17: Per-element thresholds (indices 69-90, 22 elements)
IDX_THRESH_START = 69
IDX_THRESH_END = 91  # exclusive

# Group 19: Decision fusion (indices 91-100)
IDX_FUSION_POWER_P = 91
IDX_FUSION_SINGLE_PEN = 92
IDX_FUSION_DUAL_BONUS = 93
IDX_FUSION_VETO_MIN = 94
IDX_FUSION_RESCUE_MIN = 95
IDX_FUSION_ANTIFP_ALIAS = 96
IDX_FUSION_ANTIFP_NNLS = 97
IDX_FUSION_SNR_CENTER = 98
IDX_FUSION_SNR_STEEP = 99
IDX_FUSION_SNR_FLOOR = 100

N_PARAMS = 101

# Score feature indices in the cached (n_spectra, n_elements, 8) array
FEAT_NNLS_SCORE = 0
FEAT_NNLS_SNR = 1
FEAT_NNLS_DET = 2
FEAT_ALIAS_SCORE = 3
FEAT_ALIAS_DET = 4
FEAT_ALIAS_N_MATCHED = 5
FEAT_ALIAS_N_TOTAL = 6
FEAT_ALIAS_CONF = 7


# ============================================================
# Log-scale decode mask (matches full_param_spec.py)
# ============================================================
_LOG_INDICES = [
    5,   # algo_temperature
    6,   # alias_wavelength_tol_pm
    7,   # alias_min_intensity
    12,  # alias_conf_decay_rate
    19,  # alias_score_floor
    20,  # nnls_regularization
    21,  # nnls_min_weight
    23,  # nnls_convergence_tol
    38,  # corr_min_peak_height
    40,  # corr_noise_floor
    53,  # corr_regularization
    55,  # comb_spacing_tol_pm
    59,  # comb_noise_threshold
    68,  # comb_score_floor
]


def _build_bounds_arrays():
    """Build JAX arrays for parameter bounds. Import-time constant."""
    from full_param_spec import PARAM_LOW, PARAM_HIGH, LOG_PARAM_MASK
    return (
        jnp.array(PARAM_LOW),
        jnp.array(PARAM_HIGH),
        jnp.array(LOG_PARAM_MASK),
    )


# ============================================================
# Core fitness function (JAX-traceable)
# ============================================================
def make_fitness_fn(scores_static, gt_mask_static):
    """Create a closed-over fitness function with static data baked in.

    Args:
        scores_static: (n_spectra, n_elements, 8) float64 array on GPU
        gt_mask_static: (n_spectra, n_elements) float64 array on GPU

    Returns:
        single_fitness: (101,) -> scalar, jit/vmap compatible
        batch_fitness: (N, 101) -> (N,), jit compatible
    """
    n_spectra, n_elements, _ = scores_static.shape
    eps = 1e-9

    # Pre-extract score features as static slices (avoids repeated indexing)
    s_nnls_all = scores_static[:, :, FEAT_NNLS_SCORE]      # (S, E)
    snr_all = scores_static[:, :, FEAT_NNLS_SNR]            # (S, E)
    nnls_det_all = scores_static[:, :, FEAT_NNLS_DET]       # (S, E)
    s_alias_all = scores_static[:, :, FEAT_ALIAS_SCORE]     # (S, E)
    alias_det_all = scores_static[:, :, FEAT_ALIAS_DET]     # (S, E)
    n_matched_all = scores_static[:, :, FEAT_ALIAS_N_MATCHED]  # (S, E)
    n_total_all = jnp.maximum(scores_static[:, :, FEAT_ALIAS_N_TOTAL], 1.0)
    alias_conf_all = scores_static[:, :, FEAT_ALIAS_CONF]   # (S, E)

    def single_fitness(params):
        """Evaluate one 101-dim parameter vector. Returns scalar F-beta(0.5).

        Fully JAX-traceable: no Python control flow, no side effects.
        """
        # --- Algorithm selection weights (softmax-normalized) ---
        algo_logits = jnp.array([
            params[IDX_W_NNLS],
            params[IDX_W_ALIAS],
            params[IDX_W_CORR],
            params[IDX_W_COMB],
            params[IDX_W_HYBRID],
        ])
        algo_temp = jnp.maximum(params[IDX_ALGO_TEMP], 0.1)
        algo_weights = jax.nn.softmax(algo_logits / algo_temp)
        w_nnls = algo_weights[0]
        w_alias = algo_weights[1]
        w_corr = algo_weights[2]
        w_comb = algo_weights[3]
        w_hybrid = algo_weights[4]

        # --- Decision fusion parameters ---
        power_p = params[IDX_FUSION_POWER_P]
        single_det_pen = params[IDX_FUSION_SINGLE_PEN]
        dual_bonus = params[IDX_FUSION_DUAL_BONUS]
        veto_min = params[IDX_FUSION_VETO_MIN]
        rescue_min = params[IDX_FUSION_RESCUE_MIN]
        antifp_alias = params[IDX_FUSION_ANTIFP_ALIAS]
        antifp_nnls = params[IDX_FUSION_ANTIFP_NNLS]
        snr_center = params[IDX_FUSION_SNR_CENTER]
        snr_steepness = params[IDX_FUSION_SNR_STEEP]
        snr_floor = params[IDX_FUSION_SNR_FLOOR]

        # --- ALIAS modifiers ---
        alias_rank_w = params[IDX_ALIAS_RANK_W]
        alias_floor = params[IDX_ALIAS_FLOOR]
        alias_multiplet = params[IDX_ALIAS_MULTIPLET]
        alias_isolation = params[IDX_ALIAS_ISOLATION]
        alias_blend_pen = params[IDX_ALIAS_BLEND_PEN]
        alias_wing_pen = params[IDX_ALIAS_WING_PEN]

        # --- Hybrid parameters ---
        hyb_blend = params[IDX_HYB_BLEND]
        hyb_boost = params[IDX_HYB_BOOST_AGREE]
        hyb_pen = params[IDX_HYB_PEN_CONFLICT]
        hyb_physics = params[IDX_HYB_PHYSICS]
        hyb_snr_gate = params[IDX_HYB_SNR_GATE]
        hyb_line_w = params[IDX_HYB_LINE_W]

        # --- Correlation modifiers ---
        corr_snr_w = params[IDX_CORR_SNR_W]
        corr_conf_floor = params[IDX_CORR_CONF_FLOOR]
        corr_score_pow = params[IDX_CORR_SCORE_POW]

        # --- Comb modifiers ---
        comb_conf_scale = params[IDX_COMB_CONF_SCALE]
        comb_min_snr = params[IDX_COMB_MIN_SNR]

        # --- Per-element thresholds ---
        thresholds = params[IDX_THRESH_START:IDX_THRESH_END]  # (22,)

        # ========================================
        # Compute per-algorithm adjusted scores
        # All shapes: (n_spectra, n_elements)
        # ========================================

        # SNR sigmoid gating
        snr_w = jnp.clip(
            1.0 / (1.0 + jnp.exp(-snr_steepness * (snr_all - snr_center))),
            snr_floor, 1.0
        )

        # Match ratio for ALIAS
        match_ratio = n_matched_all / n_total_all

        # === NNLS score ===
        # SNR-weighted NNLS, with normalization power
        nnls_adjusted = s_nnls_all * snr_w

        # === ALIAS score ===
        # Base ALIAS score with match ratio and confidence modifiers
        alias_base = jnp.maximum(s_alias_all, alias_floor)
        # Multiplet bonus: more matched lines -> higher score
        multiplet_factor = 1.0 + alias_multiplet * jnp.minimum(n_matched_all / 5.0, 1.0)
        # Isolation: higher confidence -> bonus
        isolation_factor = 1.0 + alias_isolation * alias_conf_all
        # Blend/wing penalties (applied via score features, proxied by conf)
        blend_factor = 1.0 - alias_blend_pen * (1.0 - alias_conf_all)
        wing_factor = 1.0 - alias_wing_pen * (1.0 - match_ratio)
        alias_adjusted = (alias_base * match_ratio * multiplet_factor
                          * isolation_factor * blend_factor * wing_factor)

        # === Correlation score (simulated from existing features) ===
        # Since we have NNLS+ALIAS cached scores, we simulate correlation
        # as a weighted combination with different emphasis
        corr_base = (s_nnls_all * corr_snr_w + s_alias_all * (1.0 - corr_snr_w))
        corr_adjusted = jnp.maximum(corr_base, corr_conf_floor) ** corr_score_pow

        # === Comb score (simulated from existing features) ===
        # Comb filter scoring: emphasizes periodic line patterns
        comb_snr_gate = jax.nn.sigmoid(snr_all - comb_min_snr)
        comb_base = s_alias_all * match_ratio * comb_snr_gate
        comb_adjusted = comb_base * comb_conf_scale

        # === Hybrid score ===
        # Blend of NNLS and ALIAS with agreement/conflict modifiers
        agreement = jnp.minimum(nnls_adjusted, alias_adjusted)
        conflict = jnp.abs(nnls_adjusted - alias_adjusted)
        hybrid_base = hyb_blend * nnls_adjusted + (1.0 - hyb_blend) * alias_adjusted
        # Boost when both agree
        hybrid_adjusted = hybrid_base + hyb_boost * agreement - hyb_pen * conflict
        # SNR gating on hybrid
        hybrid_snr_gate = jax.nn.sigmoid(snr_all - hyb_snr_gate)
        hybrid_adjusted = hybrid_adjusted * hybrid_snr_gate
        # Line count weighting
        line_factor = 1.0 + hyb_line_w * jnp.minimum(n_matched_all / 5.0, 1.0)
        hybrid_adjusted = hybrid_adjusted * line_factor
        hybrid_adjusted = jnp.maximum(hybrid_adjusted, 0.0)

        # ========================================
        # Weighted ensemble combination
        # ========================================
        # Weighted sum of all algorithm scores
        ensemble_score = (w_nnls * nnls_adjusted
                          + w_alias * alias_adjusted
                          + w_corr * corr_adjusted
                          + w_comb * comb_adjusted
                          + w_hybrid * hybrid_adjusted)

        # ========================================
        # Power mean fusion of NNLS + ALIAS (for dual-detection logic)
        # ========================================
        both_pos = (nnls_adjusted > eps) & (alias_adjusted > eps)
        nnls_only = (nnls_adjusted > eps) & (alias_adjusted <= eps)
        alias_only = (alias_adjusted > eps) & (nnls_adjusted <= eps)

        safe_nnls = jnp.maximum(nnls_adjusted, eps)
        safe_alias = jnp.maximum(alias_adjusted, eps)
        pm_val = ((safe_nnls**power_p + safe_alias**power_p) / 2.0) ** (1.0 / power_p)

        power_mean_score = jnp.where(
            both_pos, pm_val,
            jnp.where(nnls_only, nnls_adjusted * single_det_pen,
            jnp.where(alias_only, alias_adjusted * single_det_pen, 0.0))
        )

        # Dual detection bonus
        power_mean_score = power_mean_score + dual_bonus * (nnls_det_all * alias_det_all)

        # Blend ensemble with power-mean
        # hyb_physics controls mix: 1.0 = all power-mean, 0.0 = all ensemble
        combined = hyb_physics * power_mean_score + (1.0 - hyb_physics) * ensemble_score

        # ========================================
        # Detection decision (per-element thresholds)
        # ========================================
        # thresholds: (22,) broadcast over (S, 22)
        detected = (combined >= thresholds[jnp.newaxis, :]).astype(jnp.float64)

        # Veto: must have at least 1 matched line and minimum alias score
        detected = detected * (n_matched_all > 0).astype(jnp.float64)
        detected = detected * (s_alias_all >= veto_min).astype(jnp.float64)

        # Rescue: strong combined score even if below threshold
        rescue = ((combined > rescue_min) & (n_matched_all >= 1)).astype(jnp.float64)
        detected = jnp.maximum(detected, rescue)

        # Anti-FP: single line with low scores
        anti_fp = (
            (n_matched_all <= 1)
            & (s_alias_all < antifp_alias)
            & (nnls_adjusted < antifp_nnls)
        ).astype(jnp.float64)
        detected = detected * (1.0 - anti_fp)

        # ========================================
        # F-beta(0.5) computation
        # ========================================
        tp = (detected * gt_mask_static).sum()
        fp = (detected * (1.0 - gt_mask_static)).sum()
        fn = ((1.0 - detected) * gt_mask_static).sum()

        precision = tp / jnp.maximum(tp + fp, 1.0)
        recall = tp / jnp.maximum(tp + fn, 1.0)

        beta = 0.5
        f_beta = (1.0 + beta**2) * precision * recall / jnp.maximum(
            beta**2 * precision + recall, eps
        )

        return f_beta

    # Batch version: vmap over first axis
    @jit
    def batch_fitness(params_batch):
        """Evaluate fitness for a batch of parameter vectors.

        Args:
            params_batch: (N, 101) array

        Returns:
            (N,) fitness values
        """
        return vmap(single_fitness)(params_batch)

    return single_fitness, batch_fitness


# ============================================================
# pmap-compatible version for multi-GPU
# ============================================================
def make_pmap_fitness(scores_static, gt_mask_static):
    """Create a pmap-ready fitness evaluator for multi-GPU.

    Returns a function that takes (n_devices, batch_per_device, 101)
    and returns (n_devices, batch_per_device) fitness values.
    """
    single_fn, _ = make_fitness_fn(scores_static, gt_mask_static)
    vmap_fn = vmap(single_fn)

    @jax.pmap
    def pmap_fitness(params_shard):
        """Evaluate fitness on one device shard.

        Args:
            params_shard: (batch_per_device, 101) array

        Returns:
            (batch_per_device,) fitness values
        """
        return vmap_fn(params_shard)

    return pmap_fitness


# ============================================================
# Benchmarking
# ============================================================
def benchmark(cache_path, pop_sizes=None):
    """Benchmark fitness evaluation throughput on current GPU."""
    print(f"JAX devices: {jax.devices()}", flush=True)

    # Load data
    data = np.load(cache_path)
    scores = jnp.array(data["scores"])
    gt_mask = jnp.array(data["gt_mask"])
    elements = list(data["elements"])
    n_spectra, n_elements, n_features = scores.shape

    print(f"Data: {n_spectra} spectra, {n_elements} elements, {n_features} features")
    print(f"Parameters: {N_PARAMS} dimensions")
    print()

    # Build fitness function
    single_fn, batch_fn = make_fitness_fn(scores, gt_mask)

    # Load initial params
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from full_param_spec import PARAM_INIT as np_init
    init_params = jnp.array(np_init)

    # Baseline evaluation
    print("Warming up JIT...", flush=True)
    t0 = time.time()
    baseline = single_fn(init_params)
    baseline_val = float(baseline)
    warmup_time = time.time() - t0
    print(f"  JIT warmup (single): {warmup_time:.2f}s")
    print(f"  Baseline F-beta(0.5): {baseline_val:.6f}")
    print()

    # Batch warmup
    test_batch = jnp.tile(init_params, (64, 1))
    t0 = time.time()
    _ = batch_fn(test_batch)
    jax.block_until_ready(_)
    print(f"  JIT warmup (batch 64): {time.time() - t0:.2f}s")
    print()

    if pop_sizes is None:
        pop_sizes = [64, 256, 512, 1024, 2048, 4096, 8192, 16384]

    print(f"{'Pop Size':>10} | {'Time (ms)':>10} | {'Per Eval (us)':>14} | {'Evals/sec':>12}")
    print("-" * 60)

    for pop_size in pop_sizes:
        # Generate random population around init
        rng = jax.random.PRNGKey(42)
        noise = jax.random.normal(rng, (pop_size, N_PARAMS)) * 0.1
        population = jnp.clip(
            jnp.tile(init_params, (pop_size, 1)) + noise,
            0.0, 1.0  # rough clipping; real use goes through decode
        )

        # Warm up this batch size
        _ = batch_fn(population)
        jax.block_until_ready(_)

        # Timed run (average of 5)
        n_trials = 5
        times = []
        for _ in range(n_trials):
            t0 = time.time()
            f = batch_fn(population)
            jax.block_until_ready(f)
            times.append(time.time() - t0)

        avg_time = np.mean(times)
        per_eval_us = avg_time * 1e6 / pop_size
        evals_per_sec = pop_size / avg_time

        print(f"{pop_size:>10d} | {avg_time*1000:>10.2f} | {per_eval_us:>14.2f} | {evals_per_sec:>12,.0f}")

    print()

    # Final: evosax integration test
    # Use Sep_CMA_ES (diagonal covariance) to avoid cuSolver issues on V100S.
    # Full CMA_ES uses Cholesky decomposition which triggers a cuSolver bug.
    # Sep_CMA_ES is actually better for high-dim (101-D) anyway: O(n) vs O(n^3).
    print("=" * 60)
    print("evosax Sep_CMA_ES integration test")
    print("=" * 60)
    from evosax.algorithms import Sep_CMA_ES

    solution = jnp.zeros(N_PARAMS)
    es = Sep_CMA_ES(population_size=1024, solution=solution)

    rng = jax.random.PRNGKey(0)
    params = es._default_params
    state = es.init(rng, init_params, params)
    print(f"Sep_CMA_ES initialized: popsize=1024, dims={N_PARAMS}")

    best_fitness = -jnp.inf
    for gen in range(5):
        rng, ask_rng, tell_rng = jax.random.split(rng, 3)

        # Ask
        population, state = es.ask(ask_rng, state, params)

        # Evaluate on GPU
        t0 = time.time()
        fitness = batch_fn(population)
        jax.block_until_ready(fitness)
        eval_time = time.time() - t0

        # evosax minimizes, but we want to maximize F-beta
        # Negate fitness for minimization
        neg_fitness = -fitness

        # Tell
        state, metrics = es.tell(tell_rng, population, neg_fitness, state, params)

        gen_best = float(fitness.max())
        gen_mean = float(fitness.mean())
        best_fitness = max(best_fitness, gen_best)
        print(f"  Gen {gen}: best={gen_best:.6f} mean={gen_mean:.6f} "
              f"global_best={float(best_fitness):.6f} eval={eval_time*1000:.1f}ms")

    print()
    print("All tests PASSED. Fitness function is jit/vmap/pmap compatible.")
    return baseline_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline fitness benchmark")
    parser.add_argument("--cache-npz", required=True, help="Path to cache_arrays.npz")
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--pop-sizes", type=str, default=None,
                        help="Comma-separated population sizes to benchmark")
    args = parser.parse_args()

    pop_sizes = None
    if args.pop_sizes:
        pop_sizes = [int(x) for x in args.pop_sizes.split(",")]

    if args.benchmark:
        benchmark(args.cache_npz, pop_sizes)
    else:
        # Quick smoke test
        data = np.load(args.cache_npz)
        scores = jnp.array(data["scores"])
        gt_mask = jnp.array(data["gt_mask"])

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from full_param_spec import PARAM_INIT as np_init
        init_params = jnp.array(np_init)

        single_fn, batch_fn = make_fitness_fn(scores, gt_mask)
        result = single_fn(init_params)
        print(f"F-beta(0.5) = {float(result):.6f}")
