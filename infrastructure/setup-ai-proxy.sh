#!/bin/bash
###############################################################################
# AI Proxy Setup Script — Phase 0 of Autonomous Swarm Deployment
#
# Sets up the ai-proxy LXC container (CT 800 on pve3) with:
# - SSH key-based access
# - Base toolchain (Node.js 22, Rust, git)
# - Coding agent CLIs (Claude Code, Codex, Gemini CLI)
# - Beads issue tracker
# - CLIProxyPlus endpoint verification
#
# Usage:
#   # Step 0a: Inject SSH key (run from local machine)
#   ./setup-ai-proxy.sh inject-ssh-key
#
#   # Step 0b-0e: Full setup (run ON ai-proxy after SSH is working)
#   ssh root@100.105.113.58 < scripts/setup-ai-proxy.sh setup
#
# Or run individual steps:
#   ./setup-ai-proxy.sh toolchain     # Install base toolchain
#   ./setup-ai-proxy.sh agents        # Install coding agents
#   ./setup-ai-proxy.sh verify        # Verify end-to-end
#
# Prerequisites:
#   - CT 800 running on pve3 (100.68.22.98)
#   - Tailscale active: ai-proxy at 100.105.113.58
#   - CLIProxyPlus running on port 8317
###############################################################################

set -euo pipefail

# Configuration
AI_PROXY_TAILSCALE="100.105.113.58"
PVE3_TAILSCALE="100.68.22.98"
CT_ID="800"
SSH_PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILUd66NSLZYwtxk1sxFbyJSpvRACtbRyMTFHXke49myY squires.b@gmail.com"
CLI_PROXY_PORT="8317"
CLI_PROXY_KEY="rust-daq-proxy-key"

# Local model endpoints (used in configure_environment)
export LOCAL_FAST="http://slurm-ctl.tailc46cd0.ts.net:8080/v1/chat/completions"
export LOCAL_REASONING="http://slurm-ctl.tailc46cd0.ts.net:8081/v1/chat/completions"

log() {
    echo "[$(date -Iseconds)] $*"
}

###############################################################################
# Step 0a: Inject SSH Key (run from LOCAL machine, NOT on ai-proxy)
###############################################################################
inject_ssh_key() {
    log "Injecting SSH key into CT $CT_ID via pve3..."

    ssh root@"$PVE3_TAILSCALE" "pct exec $CT_ID -- bash -c 'mkdir -p /root/.ssh && echo \"$SSH_PUBKEY\" >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && echo SSH key injected'"

    log "Verifying direct SSH access..."
    if ssh -o ConnectTimeout=10 root@"$AI_PROXY_TAILSCALE" "hostname" 2>/dev/null; then
        log "SUCCESS: Direct SSH to ai-proxy works"
    else
        log "WARNING: Direct SSH failed. Check Tailscale connectivity."
        exit 1
    fi
}

###############################################################################
# Step 0b: Install Base Toolchain
###############################################################################
install_toolchain() {
    log "Installing base toolchain..."

    apt-get update -qq
    apt-get install -y -qq curl git build-essential jq tmux

    # Node.js 22 LTS
    if ! command -v node &>/dev/null; then
        log "Installing Node.js 22 LTS..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
        apt-get install -y -qq nodejs
    else
        log "Node.js already installed: $(node --version)"
    fi

    # Rust toolchain
    if ! command -v rustc &>/dev/null; then
        log "Installing Rust toolchain..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    else
        log "Rust already installed: $(rustc --version)"
    fi

    # Verify
    log "Toolchain versions:"
    node --version
    npm --version
    rustc --version
    cargo --version
    git --version
}

###############################################################################
# Step 0c: Deploy Coding Agents
###############################################################################
install_agents() {
    log "Installing coding agents..."

    # Claude Code (headless mode)
    if ! command -v claude &>/dev/null; then
        log "Installing Claude Code..."
        npm install -g @anthropic-ai/claude-code
    else
        log "Claude Code already installed: $(claude --version 2>/dev/null || echo 'installed')"
    fi

    # Codex CLI (OpenAI)
    if ! command -v codex &>/dev/null; then
        log "Installing Codex CLI..."
        npm install -g @openai/codex 2>/dev/null || log "Codex not available via npm, skip"
    fi

    # Create data directory
    mkdir -p /data/projects

    log "Agent installation complete"
}

###############################################################################
# Step 0d: Configure Environment
###############################################################################
configure_environment() {
    log "Configuring environment..."

    # Create HPC environment profile
    cat > /etc/profile.d/ai-swarm.sh << 'ENVEOF'
# AI Swarm Environment
export CLAUDE_MODEL="claude-opus-4-6"

# Local model endpoints (llama.cpp on HPC cluster)
export LOCAL_FAST_ENDPOINT="http://slurm-ctl.tailc46cd0.ts.net:8080/v1/chat/completions"
export LOCAL_REASONING_ENDPOINT="http://slurm-ctl.tailc46cd0.ts.net:8081/v1/chat/completions"

# CLIProxyPlus (all cloud providers)
export CLI_PROXY_ENDPOINT="http://localhost:8317/v1/chat/completions"
export CLI_PROXY_KEY="rust-daq-proxy-key"

# Cargo/Rust
if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
fi

# Path
export PATH="/data/projects/beefcake2/tools/rust-cluster-mcp/target/release:$PATH"
ENVEOF

    chmod +x /etc/profile.d/ai-swarm.sh
    # shellcheck source=/dev/null
    source /etc/profile.d/ai-swarm.sh

    log "Environment configured"
}

###############################################################################
# Step 0e: Verify End-to-End
###############################################################################
verify() {
    log "Running end-to-end verification..."

    local failures=0

    # Check CLIProxyPlus
    log "Checking CLIProxyPlus on port $CLI_PROXY_PORT..."
    if curl -sf -H "Authorization: Bearer $CLI_PROXY_KEY" "http://localhost:$CLI_PROXY_PORT/v1/models" >/dev/null 2>&1; then
        log "  CLIProxyPlus: OK"
    else
        log "  CLIProxyPlus: FAIL (port $CLI_PROXY_PORT not responding)"
        ((failures++)) || true
    fi

    # Check local fast tier
    log "Checking local fast tier..."
    if curl -sf --max-time 5 "http://slurm-ctl.tailc46cd0.ts.net:8080/health" >/dev/null 2>&1; then
        log "  Fast tier (14B): OK"
    else
        log "  Fast tier (14B): UNAVAILABLE (may not be running)"
    fi

    # Check local reasoning tier
    log "Checking local reasoning tier..."
    if curl -sf --max-time 5 "http://slurm-ctl.tailc46cd0.ts.net:8081/health" >/dev/null 2>&1; then
        log "  Reasoning tier (72B): OK"
    else
        log "  Reasoning tier (72B): UNAVAILABLE (may not be running)"
    fi

    # Check toolchain
    log "Checking toolchain..."
    for cmd in node npm git cargo rustc; do
        if command -v "$cmd" &>/dev/null; then
            log "  $cmd: OK ($($cmd --version 2>&1 | head -1))"
        else
            log "  $cmd: MISSING"
            ((failures++)) || true
        fi
    done

    # Check agents
    log "Checking coding agents..."
    if command -v claude &>/dev/null; then
        log "  claude: OK"
    else
        log "  claude: NOT INSTALLED"
        ((failures++)) || true
    fi

    if [[ $failures -eq 0 ]]; then
        log "VERIFICATION PASSED — all checks OK"
    else
        log "VERIFICATION INCOMPLETE — $failures issue(s) found"
    fi
}

###############################################################################
# Full Setup (Steps 0b through 0e)
###############################################################################
full_setup() {
    install_toolchain
    install_agents
    configure_environment
    verify
}

###############################################################################
# Entry Point
###############################################################################
case "${1:-help}" in
    inject-ssh-key)
        inject_ssh_key
        ;;
    toolchain)
        install_toolchain
        ;;
    agents)
        install_agents
        ;;
    configure)
        configure_environment
        ;;
    verify)
        verify
        ;;
    setup)
        full_setup
        ;;
    help|*)
        echo "Usage: $0 {inject-ssh-key|toolchain|agents|configure|verify|setup}"
        echo ""
        echo "  inject-ssh-key  Inject SSH key into CT 800 (run from local machine)"
        echo "  toolchain       Install Node.js, Rust, git, build-essential"
        echo "  agents          Install Claude Code, Codex, Gemini CLI"
        echo "  configure       Set up environment variables"
        echo "  verify          Run end-to-end verification"
        echo "  setup           Full setup (Steps 0b-0e, run ON ai-proxy)"
        ;;
esac
