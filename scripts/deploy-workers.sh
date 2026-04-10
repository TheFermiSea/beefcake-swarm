#!/usr/bin/env bash
# deploy-workers.sh — Bootstrap autonomous swarm workers on vasp compute nodes.
#
# Installs Rust toolchain, bd, cargo-nextest, clones the repo locally,
# builds swarm-agents, and deploys the systemd service. Idempotent.
#
# Usage:
#   bash scripts/deploy-workers.sh                    # Deploy to all 3 nodes
#   bash scripts/deploy-workers.sh vasp-01            # Deploy to one node
#   bash scripts/deploy-workers.sh --build-only       # Just rebuild binary on all nodes
#
# Prerequisites:
#   - SSH root access to vasp-{01,02,03} from the current host
#   - /cluster/shared/code/beefcake-swarm exists (NFS, used as clone source)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Node inventory: hostname → IP
declare -A NODES=(
    [vasp-01]=10.0.0.20
    [vasp-02]=10.0.0.21
    [vasp-03]=10.0.0.22
)

# Cloud proxy on ai-proxy (Tailnet IP)
CLOUD_PROXY_URL="http://100.105.113.58:8317/v1"
TZ_URL="http://100.105.113.58:3000"
TZ_PG_URL="postgres://tensorzero:tensorzero@100.105.113.58:5433/tensorzero"
DOLT_REMOTE="http://100.105.113.58:8001/beads"

# Parse args
BUILD_ONLY=false
TARGET_NODES=()
for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=true ;;
        vasp-*) TARGET_NODES+=("$arg") ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done
[[ ${#TARGET_NODES[@]} -eq 0 ]] && TARGET_NODES=("vasp-01" "vasp-02" "vasp-03")

log() { echo "[deploy] $(date +%H:%M:%S) $*"; }

deploy_node() {
    local name="$1"
    local ip="${NODES[$name]}"
    local ssh="ssh -o StrictHostKeyChecking=accept-new root@$ip"

    log "=== Deploying to $name ($ip) ==="

    if $BUILD_ONLY; then
        log "[$name] Build-only mode — rebuilding swarm-agents"
        $ssh 'source "$HOME/.cargo/env" 2>/dev/null
        cd /root/code/beefcake-swarm
        git pull --rebase
        cargo build --release -p swarm-agents 2>&1 | tail -3'
        log "[$name] Build complete"
        return
    fi

    # ── 1. Rust toolchain ─────────────────────────────────────────────
    log "[$name] Checking Rust toolchain..."
    $ssh 'command -v rustc >/dev/null 2>&1 || {
        echo "Installing rustup..."
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
        source "$HOME/.cargo/env"
    }
    source "$HOME/.cargo/env"
    rustup update stable 2>/dev/null || true
    rustup component add clippy rustfmt 2>/dev/null || true
    echo "rustc: $(rustc --version)"
    '

    # ── 2. cargo-nextest ──────────────────────────────────────────────
    log "[$name] Checking cargo-nextest..."
    $ssh 'source "$HOME/.cargo/env"
    command -v cargo-nextest >/dev/null 2>&1 || {
        echo "Installing cargo-nextest..."
        curl -LsSf https://get.nexte.st/latest/linux | tar zxf - -C "$HOME/.cargo/bin"
    }
    cargo nextest --version 2>/dev/null || echo "nextest not found"
    '

    # ── 3. bd (beads) ────────────────────────────────────────────────
    log "[$name] Checking bd..."
    $ssh 'command -v bd >/dev/null 2>&1 || {
        echo "Installing bd..."
        curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash
    }
    bd --version 2>/dev/null || command -v bd >/dev/null 2>&1 && echo "bd installed" || echo "bd install failed"
    '

    # ── 4. sccache ───────────────────────────────────────────────────
    log "[$name] Checking sccache..."
    $ssh 'source "$HOME/.cargo/env"
    command -v sccache >/dev/null 2>&1 || {
        echo "Installing sccache..."
        cargo install sccache --locked 2>/dev/null || echo "sccache install failed (non-critical)"
    }
    '

    # ── 5. Clone repo locally ────────────────────────────────────────
    log "[$name] Setting up local repo clone..."
    $ssh 'mkdir -p /root/code
    # Allow NFS-owned repos
    git config --global --add safe.directory "*" 2>/dev/null || true
    if [[ ! -d /root/code/beefcake-swarm/.git ]]; then
        echo "Cloning from NFS..."
        git clone --local /cluster/shared/code/beefcake-swarm /root/code/beefcake-swarm
        cd /root/code/beefcake-swarm
        # Point origin at GitHub for future fetches
        git remote set-url origin https://github.com/TheFermiSea/beefcake-swarm.git 2>/dev/null || true
    else
        echo "Updating existing clone..."
        cd /root/code/beefcake-swarm
        git fetch origin 2>/dev/null || git fetch /cluster/shared/code/beefcake-swarm
        git reset --hard origin/main 2>/dev/null || git reset --hard FETCH_HEAD
    fi
    git log --oneline -1
    '

    # ── 6. Build swarm-agents ────────────────────────────────────────
    log "[$name] Building swarm-agents (release)..."
    $ssh 'source "$HOME/.cargo/env"
    cd /root/code/beefcake-swarm
    # Use sccache if available
    export RUSTC_WRAPPER=$(command -v sccache 2>/dev/null || echo "")
    cargo build --release -p swarm-agents 2>&1 | tail -5
    ls -lh target/release/swarm-agents
    '

    # ── 7. Configure beads Dolt remote ───────────────────────────────
    log "[$name] Configuring beads..."
    $ssh "cd /root/code/beefcake-swarm
    # Initialize beads if needed
    [[ -d .beads ]] || bd init 2>/dev/null || true
    # Set Dolt remote to ai-proxy
    bd dolt remote set-url origin '$DOLT_REMOTE' 2>/dev/null || true
    bd dolt pull 2>/dev/null || true
    "

    # ── 8. Deploy swarm-worker service ───────────────────────────────
    log "[$name] Deploying swarm-worker service..."
    # Copy the worker runner script
    scp -o StrictHostKeyChecking=accept-new \
        "$REPO_ROOT/scripts/swarm-worker.sh" \
        "root@$ip:/root/code/beefcake-swarm/scripts/swarm-worker.sh"
    $ssh 'chmod +x /root/code/beefcake-swarm/scripts/swarm-worker.sh'

    # Deploy systemd unit
    cat <<UNIT | $ssh 'cat > /etc/systemd/system/swarm-worker.service'
[Unit]
Description=Beefcake Swarm Worker (%H)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/code/beefcake-swarm
ExecStart=/root/code/beefcake-swarm/scripts/swarm-worker.sh
Restart=on-failure
RestartSec=60
Environment=HOME=/root
Environment=PATH=/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin
Environment=SWARM_CLOUD_URL=${CLOUD_PROXY_URL}
Environment=SWARM_CLOUD_API_KEY=rust-daq-proxy-key
Environment=SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0
Environment=SWARM_TENSORZERO_URL=${TZ_URL}
Environment=SWARM_TENSORZERO_PG_URL=${TZ_PG_URL}
Environment=RUST_LOG=info,hyper=info,reqwest=info,h2=info,rustls=info,tower=info

# Resource limits — let it use the full machine
LimitNOFILE=65536
Nice=10

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=swarm-worker

[Install]
WantedBy=multi-user.target
UNIT

    $ssh 'systemctl daemon-reload
    systemctl enable swarm-worker.service
    echo "Service installed and enabled"
    '

    log "[$name] Deploy complete!"
}

# ── Main ─────────────────────────────────────────────────────────────

for name in "${TARGET_NODES[@]}"; do
    if [[ -z "${NODES[$name]+x}" ]]; then
        echo "Unknown node: $name (expected: vasp-01, vasp-02, vasp-03)"
        exit 1
    fi
    deploy_node "$name"
    echo ""
done

log "=== All nodes deployed. Start with: ==="
for name in "${TARGET_NODES[@]}"; do
    echo "  ssh root@${NODES[$name]} 'systemctl start swarm-worker'"
done
echo ""
log "Monitor with:"
echo "  ssh root@10.0.0.20 'journalctl -u swarm-worker -f'"
