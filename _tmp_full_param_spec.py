"""
Expanded parameter specification for CF-LIBS ensemble optimization.

Phase 1: ~100 parameters covering:
  Group 1: Algorithm selection weights (6 params)
  Group 2: ALIAS identifier config (14 params)
  Group 3: NNLS identifier config (5 params)
  Group 4: Hybrid fusion config (12 params)
  Group 5: Correlation identifier config (18 params)
  Group 6: Comb filter config (14 params)
  Group 17: Per-element thresholds (22 params)
  Group 19: Decision fusion (10 params)

Total: 101 parameters

Each parameter: (name, initial_value, lower_bound, upper_bound, group, encoding)
Encoding types:
  "linear"  - direct [low, high] mapping
  "log"     - log-scale (for parameters spanning orders of magnitude)
  "sigmoid" - sigmoid mapping to (0, 1) for weights
"""
import numpy as np

# ============================================================
# Parameter definitions by group
# ============================================================

# Group 1: Algorithm selection weights
# These control how much each identifier contributes to the ensemble.
# Softmax-normalized at decode time, so raw values are unconstrained logits.
GROUP_1_ALGO_SELECTION = [
    ("w_nnls",           1.0,  -3.0,  3.0,  1, "linear"),
    ("w_alias",          1.0,  -3.0,  3.0,  1, "linear"),
    ("w_correlation",    0.5,  -3.0,  3.0,  1, "linear"),
    ("w_comb",           0.3,  -3.0,  3.0,  1, "linear"),
    ("w_hybrid",         0.8,  -3.0,  3.0,  1, "linear"),
    ("algo_temperature", 1.0,   0.1,  5.0,  1, "log"),
]

# Group 2: ALIAS identifier configuration
GROUP_2_ALIAS = [
    ("alias_wavelength_tol_pm",   10.0,    1.0,   50.0,  2, "log"),
    ("alias_min_intensity",        0.01,   0.001,   0.5,  2, "log"),
    ("alias_ionization_weight",    0.5,    0.1,    2.0,   2, "linear"),
    ("alias_persistence_weight",   0.3,    0.0,    1.0,   2, "linear"),
    ("alias_self_abs_penalty",     0.2,    0.0,    1.0,   2, "linear"),
    ("alias_max_lines_per_elem",  15.0,    3.0,   50.0,   2, "linear"),
    ("alias_conf_decay_rate",      0.1,    0.01,   1.0,   2, "log"),
    ("alias_snr_threshold",        2.0,    0.5,   10.0,   2, "linear"),
    ("alias_multiplet_bonus",      0.15,   0.0,    0.5,   2, "linear"),
    ("alias_isolation_bonus",      0.1,    0.0,    0.5,   2, "linear"),
    ("alias_wing_penalty",         0.05,   0.0,    0.3,   2, "linear"),
    ("alias_blend_penalty",        0.1,    0.0,    0.5,   2, "linear"),
    ("alias_rank_weight",          0.4,    0.0,    1.0,   2, "linear"),
    ("alias_score_floor",          0.01,   0.001,  0.1,   2, "log"),
]

# Group 3: NNLS identifier configuration
GROUP_3_NNLS = [
    ("nnls_regularization",    0.01,  0.0001,  1.0,   3, "log"),
    ("nnls_min_weight",        0.001, 0.0001,  0.1,   3, "log"),
    ("nnls_max_components",   20.0,   5.0,    50.0,   3, "linear"),
    ("nnls_convergence_tol",   1e-4,  1e-6,   1e-2,   3, "log"),
    ("nnls_normalization_p",   1.0,   0.5,    2.0,    3, "linear"),
]

# Group 4: Hybrid fusion configuration
GROUP_4_HYBRID = [
    ("hybrid_nnls_alias_blend",     0.5,   0.0,  1.0,  4, "linear"),
    ("hybrid_conflict_resolution",  0.5,   0.0,  1.0,  4, "linear"),
    ("hybrid_min_agreement",        0.3,   0.1,  0.9,  4, "linear"),
    ("hybrid_boost_agreement",      0.15,  0.0,  0.5,  4, "linear"),
    ("hybrid_penalty_conflict",     0.2,   0.0,  0.5,  4, "linear"),
    ("hybrid_snr_gate",             2.0,   0.5,  8.0,  4, "linear"),
    ("hybrid_line_count_weight",    0.3,   0.0,  1.0,  4, "linear"),
    ("hybrid_det_count_weight",     0.4,   0.0,  1.0,  4, "linear"),
    ("hybrid_physics_prior",        0.5,   0.0,  1.0,  4, "linear"),
    ("hybrid_rank_blend",           0.3,   0.0,  1.0,  4, "linear"),
    ("hybrid_veto_threshold",       0.05,  0.0,  0.3,  4, "linear"),
    ("hybrid_rescue_threshold",     0.7,   0.3,  1.0,  4, "linear"),
]

# Group 5: Correlation identifier configuration
GROUP_5_CORRELATION = [
    ("corr_window_size",        5.0,   1.0,  20.0,   5, "linear"),
    ("corr_min_peak_height",    0.05,  0.01,  0.5,   5, "log"),
    ("corr_template_threshold", 0.6,   0.2,   0.95,  5, "linear"),
    ("corr_noise_floor",        0.02,  0.001, 0.2,   5, "log"),
    ("corr_peak_width_factor",  1.5,   0.5,   5.0,   5, "linear"),
    ("corr_baseline_order",     2.0,   0.0,   5.0,   5, "linear"),
    ("corr_normalization",      1.0,   0.0,   2.0,   5, "linear"),
    ("corr_multi_peak_bonus",   0.1,   0.0,   0.5,   5, "linear"),
    ("corr_isolation_weight",   0.2,   0.0,   1.0,   5, "linear"),
    ("corr_snr_weight",         0.5,   0.0,   1.0,   5, "linear"),
    ("corr_persistence_check",  0.3,   0.0,   1.0,   5, "linear"),
    ("corr_ionization_ratio",   0.4,   0.0,   1.0,   5, "linear"),
    ("corr_self_abs_check",     0.2,   0.0,   1.0,   5, "linear"),
    ("corr_confidence_floor",   0.1,   0.0,   0.5,   5, "linear"),
    ("corr_max_overlap",        0.3,   0.05,  0.8,   5, "linear"),
    ("corr_deconv_iterations",  5.0,   1.0,  20.0,   5, "linear"),
    ("corr_regularization",     0.01,  0.001, 0.5,   5, "log"),
    ("corr_score_power",        1.0,   0.5,   3.0,   5, "linear"),
]

# Group 6: Comb filter configuration
GROUP_6_COMB = [
    ("comb_spacing_tol_pm",     5.0,   0.5,  20.0,   6, "log"),
    ("comb_min_teeth",          3.0,   2.0,  10.0,   6, "linear"),
    ("comb_intensity_ratio",    0.3,   0.05,  0.9,   6, "linear"),
    ("comb_harmonic_weight",    0.2,   0.0,   1.0,   6, "linear"),
    ("comb_noise_threshold",    0.05,  0.01,  0.3,   6, "log"),
    ("comb_window_width",      10.0,   2.0,  50.0,   6, "linear"),
    ("comb_baseline_subtract",  0.5,   0.0,   1.0,   6, "linear"),
    ("comb_peak_sharpness",     1.5,   0.5,   5.0,   6, "linear"),
    ("comb_isolation_bonus",    0.1,   0.0,   0.5,   6, "linear"),
    ("comb_overlap_penalty",    0.15,  0.0,   0.5,   6, "linear"),
    ("comb_confidence_scale",   1.0,   0.3,   3.0,   6, "linear"),
    ("comb_min_snr",            1.5,   0.5,   8.0,   6, "linear"),
    ("comb_persistence_weight", 0.3,   0.0,   1.0,   6, "linear"),
    ("comb_score_floor",        0.02,  0.001, 0.2,   6, "log"),
]

# Group 17: Per-element detection thresholds (one per element in the dataset)
_ELEMENTS = [
    "Fe", "Ca", "Mg", "Si", "Al", "Ti", "Na", "K", "Mn", "Cr", "Ni",
    "Cu", "Co", "V", "Li", "Sr", "Ba", "Zn", "Pb", "Mo", "Zr", "Sn",
]
# Initial thresholds: majors lower, traces higher
_MAJOR_ELEMENTS = {"Fe", "Ca", "Mg", "Si", "Al", "Ti", "Na", "K"}
GROUP_17_THRESHOLDS = [
    (f"thresh_{el}", 0.30 if el in _MAJOR_ELEMENTS else 0.42, 0.05, 0.95, 17, "linear")
    for el in _ELEMENTS
]

# Group 19: Decision fusion parameters
GROUP_19_FUSION = [
    ("fusion_power_mean_p",         -0.5,  -3.0,  2.0,  19, "linear"),
    ("fusion_single_det_penalty",    0.35,  0.0,   1.0,  19, "linear"),
    ("fusion_dual_detect_bonus",     0.08,  0.0,   0.3,  19, "linear"),
    ("fusion_veto_min_score",        0.03,  0.0,   0.3,  19, "linear"),
    ("fusion_rescue_combined_min",   0.6,   0.2,   0.95, 19, "linear"),
    ("fusion_antifp_alias_max",      0.2,   0.05,  0.5,  19, "linear"),
    ("fusion_antifp_nnls_max",       0.3,   0.05,  0.6,  19, "linear"),
    ("fusion_snr_center",            2.5,   0.5,  10.0,  19, "linear"),
    ("fusion_snr_steepness",         1.5,   0.1,   5.0,  19, "linear"),
    ("fusion_snr_floor",             0.05,  0.0,   0.5,  19, "linear"),
]


# ============================================================
# Combine all groups
# ============================================================
ALL_GROUPS = [
    ("Algorithm Selection",  GROUP_1_ALGO_SELECTION),
    ("ALIAS Config",         GROUP_2_ALIAS),
    ("NNLS Config",          GROUP_3_NNLS),
    ("Hybrid Fusion",        GROUP_4_HYBRID),
    ("Correlation Config",   GROUP_5_CORRELATION),
    ("Comb Filter",          GROUP_6_COMB),
    ("Per-Element Thresh",   GROUP_17_THRESHOLDS),
    ("Decision Fusion",      GROUP_19_FUSION),
]

# Flat list of all parameters
FULL_PARAM_SPEC = []
for _group_name, _group_params in ALL_GROUPS:
    FULL_PARAM_SPEC.extend(_group_params)

N_PARAMS = len(FULL_PARAM_SPEC)
PARAM_NAMES   = [p[0] for p in FULL_PARAM_SPEC]
PARAM_INIT    = np.array([p[1] for p in FULL_PARAM_SPEC], dtype=np.float64)
PARAM_LOW     = np.array([p[2] for p in FULL_PARAM_SPEC], dtype=np.float64)
PARAM_HIGH    = np.array([p[3] for p in FULL_PARAM_SPEC], dtype=np.float64)
PARAM_GROUPS  = np.array([p[4] for p in FULL_PARAM_SPEC], dtype=np.int32)
PARAM_ENCODING = [p[5] for p in FULL_PARAM_SPEC]

# Build index lookups
PARAM_INDEX = {name: i for i, name in enumerate(PARAM_NAMES)}
GROUP_SLICES = {}
for group_name, group_params in ALL_GROUPS:
    names = [p[0] for p in group_params]
    start = PARAM_INDEX[names[0]]
    end = PARAM_INDEX[names[-1]] + 1
    GROUP_SLICES[group_name] = slice(start, end)

# Element name -> threshold parameter index
ELEMENT_THRESH_INDEX = {
    el: PARAM_INDEX[f"thresh_{el}"] for el in _ELEMENTS
}

# Identify log-scale parameters for encode/decode
LOG_PARAM_MASK = np.array([enc == "log" for enc in PARAM_ENCODING], dtype=bool)


# ============================================================
# Encode / Decode functions
# ============================================================
def encode_to_unit(raw_params):
    """Map raw parameter values to [0, 1] unit hypercube.

    Linear params: (x - low) / (high - low)
    Log params: (log(x) - log(low)) / (log(high) - log(low))
    """
    unit = np.zeros_like(raw_params)
    for i in range(N_PARAMS):
        lo, hi = PARAM_LOW[i], PARAM_HIGH[i]
        if LOG_PARAM_MASK[i]:
            unit[i] = (np.log(raw_params[i]) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            unit[i] = (raw_params[i] - lo) / (hi - lo)
    return np.clip(unit, 0.0, 1.0)


def decode_from_unit(unit_params):
    """Map [0, 1] unit hypercube back to raw parameter values.

    Linear params: low + x * (high - low)
    Log params: exp(log(low) + x * (log(high) - log(low)))
    """
    raw = np.zeros_like(unit_params)
    for i in range(N_PARAMS):
        lo, hi = PARAM_LOW[i], PARAM_HIGH[i]
        u = np.clip(unit_params[i], 0.0, 1.0)
        if LOG_PARAM_MASK[i]:
            raw[i] = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
        else:
            raw[i] = lo + u * (hi - lo)
    return raw


# ============================================================
# JAX-compatible vectorized encode/decode
# ============================================================
def make_jax_codec():
    """Return JAX-jittable encode/decode functions.

    These operate on arrays and are vmap/jit compatible.
    Returns (jax_encode, jax_decode) functions.
    """
    import jax.numpy as jnp

    low = jnp.array(PARAM_LOW)
    high = jnp.array(PARAM_HIGH)
    log_mask = jnp.array(LOG_PARAM_MASK)
    log_low = jnp.log(jnp.where(log_mask, jnp.maximum(low, 1e-10), 1.0))
    log_high = jnp.log(jnp.where(log_mask, jnp.maximum(high, 1e-10), 1.0))
    log_range = log_high - log_low
    lin_range = high - low

    def jax_encode(raw):
        """Raw params -> [0, 1] unit cube. JIT-compatible."""
        log_unit = (jnp.log(jnp.maximum(raw, 1e-10)) - log_low) / jnp.maximum(log_range, 1e-10)
        lin_unit = (raw - low) / jnp.maximum(lin_range, 1e-10)
        unit = jnp.where(log_mask, log_unit, lin_unit)
        return jnp.clip(unit, 0.0, 1.0)

    def jax_decode(unit):
        """[0, 1] unit cube -> raw params. JIT-compatible."""
        u = jnp.clip(unit, 0.0, 1.0)
        log_raw = jnp.exp(log_low + u * log_range)
        lin_raw = low + u * lin_range
        return jnp.where(log_mask, log_raw, lin_raw)

    return jax_encode, jax_decode


# ============================================================
# Summary
# ============================================================
def print_summary():
    """Print parameter specification summary."""
    print(f"Total parameters: {N_PARAMS}")
    print(f"Log-scale parameters: {LOG_PARAM_MASK.sum()}")
    print()
    for group_name, group_params in ALL_GROUPS:
        sl = GROUP_SLICES[group_name]
        n = sl.stop - sl.start
        log_count = LOG_PARAM_MASK[sl].sum()
        print(f"  {group_name}: {n} params (indices {sl.start}-{sl.stop - 1})"
              f"{f', {log_count} log-scale' if log_count else ''}")
    print()
    print("Element threshold indices:")
    for el, idx in ELEMENT_THRESH_INDEX.items():
        print(f"  {el}: param[{idx}] = {PARAM_INIT[idx]:.2f} "
              f"[{PARAM_LOW[idx]:.2f}, {PARAM_HIGH[idx]:.2f}]")


if __name__ == "__main__":
    print_summary()

    # Verify encode/decode round-trip
    unit = encode_to_unit(PARAM_INIT)
    decoded = decode_from_unit(unit)
    max_err = np.max(np.abs(decoded - PARAM_INIT))
    print(f"\nEncode/decode round-trip max error: {max_err:.2e}")
    assert max_err < 1e-10, f"Round-trip error too large: {max_err}"
    print("Round-trip verification PASSED")
