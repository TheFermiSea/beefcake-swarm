#!/usr/bin/env bash
# post-merge-benchmark.sh — Physics benchmark gate for CF-LIBS changes.
#
# Classifies whether a merge touched physics-critical code, then runs
# GPU-accelerated benchmarks on vasp-03 to validate correctness.
#
# Usage:
#   ./scripts/post-merge-benchmark.sh classify <repo-dir> [base-ref]
#   ./scripts/post-merge-benchmark.sh run <repo-dir> <tier> [issue-id]
#   ./scripts/post-merge-benchmark.sh setup
#
# Tiers:
#   light    Quick unified benchmark (~1-2 min GPU, post-merge)
#   heavy    Comprehensive benchmark (~10-30 min GPU, high-risk changes)
#   nightly  Full regression suite (scheduled timer)
#
# Environment:
#   BENCH_HOST          SSH target for benchmark node (default: root@10.0.0.22)
#   BENCH_NFS_PATH      Benchmark repo on NFS (default: /cluster/shared/cf-libs-bench/repo)
#   BENCH_DRAIN         Always drain inference before benchmark (default: 0)
#   BENCH_SKIP_SYNC     Skip code sync, useful for debugging (default: 0)
#   BENCH_TIMEOUT       Benchmark timeout in seconds (default: 1800 = 30 min)
#   BENCH_VENV          Python venv path on bench host (default: .venv)
#
set -euo pipefail

# --- Configuration ---
BENCH_HOST="${BENCH_HOST:-root@10.0.0.22}"
BENCH_NFS_PATH="${BENCH_NFS_PATH:-/scratch/cf-libs-bench/repo}"
BENCH_DRAIN="${BENCH_DRAIN:-0}"
BENCH_SKIP_SYNC="${BENCH_SKIP_SYNC:-0}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
BENCH_VENV="${BENCH_VENV:-.venv}"

log() { echo "[benchmark $(date '+%H:%M:%S')] $*"; }

# ── Physics-touching path patterns ──
# Fail-closed: if uncertain, benchmark.
#
# "heavy" paths warrant the comprehensive benchmark (manifold, basis, benchmark infra).
# "physics" paths get the quick benchmark (atomic data, plasma, radiation, etc.).
# Everything else is "skip" (docs, tests, cosmetic).

is_heavy_path() {
  local file="$1"
  case "$file" in
    cflibs/manifold/*) return 0 ;;
    cflibs/benchmark/*) return 0 ;;
    pyproject.toml) return 0 ;;
    datagen_v2.py) return 0 ;;
  esac
  return 1
}

is_physics_path() {
  local file="$1"
  case "$file" in
    cflibs/atomic/*) return 0 ;;
    cflibs/benchmark/*) return 0 ;;
    cflibs/core/constants.py) return 0 ;;
    cflibs/core/platform_config.py) return 0 ;;
    cflibs/inversion/*) return 0 ;;
    cflibs/manifold/*) return 0 ;;
    cflibs/plasma/*) return 0 ;;
    cflibs/radiation/*) return 0 ;;
    cflibs/validation/*) return 0 ;;
    datagen_v2.py) return 0 ;;
    pyproject.toml) return 0 ;;
    scripts/run_*benchmark*) return 0 ;;
  esac
  return 1
}

is_skip_path() {
  local file="$1"
  case "$file" in
    docs/*|*.md|.github/*|.swarm/*|.gpd/*) return 0 ;;
  esac
  return 1
}

# ── classify ──
# Determines whether a merge needs benchmarking and at what tier.
# Outputs: "skip", "light", or "heavy"
do_classify() {
  local repo_dir="$1"
  local base_ref="${2:-HEAD~1}"

  local changed_files
  changed_files=$(git -C "$repo_dir" diff --name-only "${base_ref}..HEAD" 2>/dev/null || true)

  if [[ -z "$changed_files" ]]; then
    echo "skip"
    return
  fi

  local dominated_by_skip=true
  local has_physics=false
  local has_heavy=false

  while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    if is_heavy_path "$file"; then
      has_heavy=true
      dominated_by_skip=false
    elif is_physics_path "$file"; then
      has_physics=true
      dominated_by_skip=false
    elif ! is_skip_path "$file"; then
      dominated_by_skip=false
      # Unknown path — check if it's a Python runtime file.
      # Conservative: non-skip Python files are assumed physics-touching.
      if [[ "$file" == *.py ]] && [[ "$file" != tests/* ]]; then
        # Check if changes are more than just annotations/docstrings/comments
        local diff_content
        diff_content=$(git -C "$repo_dir" diff "${base_ref}..HEAD" -- "$file" 2>/dev/null || true)
        if echo "$diff_content" | grep -qE '^\+[^+#"'"'"'].*[=():]'; then
          has_physics=true
        fi
      fi
    fi
  done <<< "$changed_files"

  if $has_heavy; then
    echo "heavy"
  elif $has_physics; then
    echo "light"
  elif $dominated_by_skip; then
    echo "skip"
  else
    # Fail-closed: unknown files → benchmark
    echo "light"
  fi
}

# ── sync ──
# Rsync code from the local repo to NFS on the benchmark host.
# Excludes .venv, data, ASD_da, output (persistent on NFS).
do_sync() {
  local repo_dir="$1"

  if [[ "$BENCH_SKIP_SYNC" == "1" ]]; then
    log "Skipping sync (BENCH_SKIP_SYNC=1)"
    return 0
  fi

  log "Syncing $repo_dir → $BENCH_HOST:$BENCH_NFS_PATH"
  rsync -az --delete \
    --exclude='.venv' \
    --exclude='data/' \
    --exclude='ASD_da/' \
    --exclude='output/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.egg-info/' \
    "$repo_dir/" "$BENCH_HOST:$BENCH_NFS_PATH/"
}

# ── drain / restore inference ──
drain_inference() {
  log "Draining inference on $BENCH_HOST..."
  ssh "$BENCH_HOST" "pkill -TERM -f 'llama-server-mmq' 2>/dev/null || true"
  # Wait for process to exit (up to 30s)
  local waited=0
  while ssh "$BENCH_HOST" "pgrep -f llama-server-mmq >/dev/null 2>&1"; do
    sleep 2
    waited=$((waited + 2))
    if [[ $waited -ge 30 ]]; then
      log "WARNING: Inference did not stop after 30s, force killing"
      ssh "$BENCH_HOST" "pkill -9 -f 'llama-server-mmq' 2>/dev/null || true"
      sleep 2
      break
    fi
  done
  log "Inference drained"
}

restore_inference() {
  log "Restoring inference on $BENCH_HOST..."
  # Use llm-stack if available, otherwise fall back to known start script
  if ssh "$BENCH_HOST" "test -x /usr/local/sbin/llm-stack" 2>/dev/null; then
    ssh "$BENCH_HOST" "/usr/local/sbin/llm-stack start"
  elif ssh "$BENCH_HOST" "test -f /tmp/start-qwen35-mmq.sh" 2>/dev/null; then
    ssh "$BENCH_HOST" "bash /tmp/start-qwen35-mmq.sh"
  else
    log "WARNING: No inference start script found — manual restart required"
    return 1
  fi
  log "Inference restore initiated"
}

# ── run_benchmark_on_host ──
# SSH to benchmark host and execute the benchmark.
# Returns: 0 on success, 1 on benchmark failure, 2 on OOM
run_benchmark_on_host() {
  local tier="$1"
  local git_sha="$2"
  local shared_gpu="${3:-1}"  # 1 = try shared GPU, 0 = exclusive

  local jax_env="JAX_PLATFORMS=cuda"
  if [[ "$shared_gpu" == "1" ]]; then
    jax_env="$jax_env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5"
  fi

  local bench_cmd output_dir
  output_dir="output/benchmark_gate/${git_sha}"

  case "$tier" in
    light)
      bench_cmd="$jax_env ${BENCH_VENV}/bin/python scripts/run_unified_benchmark.py \
        --quick --max-outer-folds 1 \
        --sections all \
        --id-workflows alias comb \
        --composition-workflows iterative \
        --db-path ASD_da/libs_production.db \
        --data-dir data \
        --output-dir $output_dir"
      ;;
    heavy|nightly)
      bench_cmd="$jax_env ${BENCH_VENV}/bin/python scripts/run_comprehensive_benchmark.py \
        --db ASD_da/libs_production.db \
        --n-compositions 50 \
        --output-dir $output_dir"
      ;;
    *)
      log "ERROR: Unknown tier: $tier"
      return 1
      ;;
  esac

  log "Running $tier benchmark on $BENCH_HOST (shared_gpu=$shared_gpu)..."
  local exit_code=0
  local bench_output
  bench_output=$(ssh -o ConnectTimeout=10 "$BENCH_HOST" \
    "cd $BENCH_NFS_PATH && HOME=/tmp CUDA_CACHE_PATH=/tmp/cuda-cache \
     timeout $BENCH_TIMEOUT $bench_cmd" 2>&1) || exit_code=$?

  # Detect OOM
  if echo "$bench_output" | grep -qiE 'out.of.memory|OOM|RESOURCE_EXHAUSTED|XLA_PYTHON_CLIENT'; then
    log "GPU OOM detected"
    echo "$bench_output" | tail -5
    return 2
  fi

  if [[ $exit_code -ne 0 ]]; then
    log "Benchmark failed (exit=$exit_code)"
    echo "$bench_output" | tail -10
    return 1
  fi

  # Verify results exist
  local results_exist
  results_exist=$(ssh "$BENCH_HOST" "ls $BENCH_NFS_PATH/$output_dir/*.json 2>/dev/null | wc -l" || echo "0")
  if [[ "$results_exist" == "0" ]]; then
    log "WARNING: Benchmark exited 0 but no result JSON found"
    echo "$bench_output" | tail -10
    return 1
  fi

  log "Benchmark passed ($results_exist result files)"
  return 0
}

# ── run ──
# Main benchmark runner with shared-GPU-then-drain-on-OOM logic.
do_run() {
  local repo_dir="$1"
  local tier="$2"
  local issue_id="${3:-unknown}"

  local git_sha
  git_sha=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo "unknown")

  log "=== Benchmark gate: tier=$tier issue=$issue_id sha=$git_sha ==="

  # Step 1: Sync code
  do_sync "$repo_dir"

  # Step 2: Run benchmark
  local drained=false
  local exit_code=0

  if [[ "$BENCH_DRAIN" == "1" ]] || [[ "$tier" == "heavy" ]] || [[ "$tier" == "nightly" ]]; then
    # Heavy/nightly: always drain for full GPU access
    drain_inference
    drained=true
    run_benchmark_on_host "$tier" "$git_sha" 0 || exit_code=$?
  else
    # Light: try shared GPU first
    run_benchmark_on_host "$tier" "$git_sha" 1 || exit_code=$?

    if [[ $exit_code -eq 2 ]]; then
      # OOM — drain and retry with full GPU
      log "Retrying with full GPU (draining inference)..."
      drain_inference
      drained=true
      exit_code=0
      run_benchmark_on_host "$tier" "$git_sha" 0 || exit_code=$?
    fi
  fi

  # Step 3: Restore inference if we drained
  if $drained; then
    restore_inference || log "WARNING: Failed to restart inference"
  fi

  if [[ $exit_code -eq 0 ]]; then
    log "=== Benchmark PASSED (tier=$tier) ==="
  else
    log "=== Benchmark FAILED (tier=$tier, exit=$exit_code) ==="
  fi

  return $exit_code
}

# ── setup ──
# One-time environment setup on the benchmark host.
do_setup() {
  local repo_dir="${1:-}"

  log "Setting up benchmark environment on $BENCH_HOST..."

  # Create NFS directory
  ssh "$BENCH_HOST" "mkdir -p $BENCH_NFS_PATH"

  # Sync full repo including data
  if [[ -n "$repo_dir" ]]; then
    log "Syncing full repo (including data)..."
    rsync -az --delete \
      --exclude='.venv' \
      --exclude='output/' \
      --exclude='__pycache__/' \
      --exclude='*.egg-info/' \
      "$repo_dir/" "$BENCH_HOST:$BENCH_NFS_PATH/"
  fi

  # Create venv with CUDA JAX
  log "Creating Python venv with JAX CUDA support..."
  ssh "$BENCH_HOST" "cd $BENCH_NFS_PATH && \
    python3.11 -m venv ${BENCH_VENV} && \
    source ${BENCH_VENV}/bin/activate && \
    pip install --upgrade pip && \
    pip install -e '.[jax-cuda,dev]'"

  log "Setup complete. Test with: $0 run <repo-dir> light"
}

# ── Main dispatch ──
ACTION="${1:-}"

case "$ACTION" in
  classify)
    REPO_DIR="${2:?Usage: $0 classify <repo-dir> [base-ref]}"
    BASE_REF="${3:-HEAD~1}"
    do_classify "$REPO_DIR" "$BASE_REF"
    ;;
  run)
    REPO_DIR="${2:?Usage: $0 run <repo-dir> <tier> [issue-id]}"
    TIER="${3:?Usage: $0 run <repo-dir> <tier> [issue-id]}"
    ISSUE_ID="${4:-unknown}"
    do_run "$REPO_DIR" "$TIER" "$ISSUE_ID"
    ;;
  restore)
    # Restore inference after a benchmark or crash (used by systemd ExecStopPost)
    restore_inference
    ;;
  setup)
    REPO_DIR="${2:-}"
    do_setup "$REPO_DIR"
    ;;
  *)
    echo "Usage: $0 {classify|run|setup|restore} [args...]"
    echo ""
    echo "Commands:"
    echo "  classify <repo-dir> [base-ref]   Classify changes (outputs: skip|light|heavy)"
    echo "  run <repo-dir> <tier> [issue-id] Run benchmark (tiers: light|heavy|nightly)"
    echo "  restore                          Restore inference on benchmark host"
    echo "  setup [repo-dir]                 One-time env setup on benchmark host"
    exit 1
    ;;
esac
