#!/bin/bash
###############################################################################
# AI Proxy Setup Script — Phase 0 of Autonomous Swarm Deployment
#
# Sets up the ai-proxy LXC container (CT 800 on pve3) with:
# - SSH key-based access
# - Base toolchain (Node.js 22, Rust, git)
# - Coding agent CLIs (Claude Code, Codex, OpenCode, Factory Droid, Crush)
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

    # OpenCode
    if ! command -v opencode &>/dev/null; then
        log "Installing OpenCode..."
        curl -fsSL https://opencode.ai/install | bash
    else
        log "OpenCode already installed: $(opencode --version 2>/dev/null || echo 'installed')"
    fi

    # Factory Droid
    if ! command -v droid &>/dev/null; then
        log "Installing Factory Droid..."
        curl -fsSL https://app.factory.ai/cli | sh
    else
        log "Factory Droid already installed: $(droid --version 2>/dev/null || echo 'installed')"
    fi

    # Crush (Charmbracelet)
    if ! command -v crush &>/dev/null; then
        log "Installing Crush..."
        npm install -g @charmland/crush
    else
        log "Crush already installed: $(crush --version 2>/dev/null || echo 'installed')"
    fi

    # Create data directory
    mkdir -p /data/projects

    log "Agent installation complete"
}

###############################################################################
# Step 0c2: Configure Agent CLIs (OpenCode, Factory Droid, Crush)
###############################################################################
configure_agents() {
    log "Configuring coding agent CLIs..."

    # --- OpenCode ---
    # Uses config.json with "provider" key, @ai-sdk/openai-compatible npm adapter
    local opencode_dir="$HOME/.config/opencode"
    mkdir -p "$opencode_dir"
    cat > "$opencode_dir/config.json" << 'OCEOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "beefcake-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Beefcake Proxy (Cloud)",
      "options": {
        "baseURL": "http://localhost:8317/v1",
        "apiKey": "rust-daq-proxy-key"
      },
      "models": {
        "claude-opus-4-6": {
          "name": "Claude Opus 4.6",
          "limit": { "context": 200000, "output": 16384 }
        },
        "claude-opus-4-6-thinking": {
          "name": "Claude Opus 4.6 Thinking",
          "limit": { "context": 200000, "output": 32768 }
        },
        "claude-sonnet-4-5-20250929": {
          "name": "Claude Sonnet 4.5",
          "limit": { "context": 200000, "output": 16384 }
        },
        "gpt-5.2-codex": {
          "name": "GPT-5.2 Codex",
          "limit": { "context": 128000, "output": 16384 }
        },
        "gemini-3-pro-preview": {
          "name": "Gemini 3 Pro",
          "limit": { "context": 2000000, "output": 65536 }
        }
      }
    },
    "beefcake-72b": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Beefcake 72B (Local)",
      "options": {
        "baseURL": "http://slurm-ctl.tailc46cd0.ts.net:8081/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "or1-behemoth-q4_k_m": {
          "name": "OR1 Behemoth 72B",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    },
    "beefcake-14b": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Beefcake 14B (Local)",
      "options": {
        "baseURL": "http://slurm-ctl.tailc46cd0.ts.net:8080/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "strand-rust-coder-14b-q8_0": {
          "name": "Strand Rust Coder 14B",
          "limit": { "context": 16384, "output": 4096 }
        }
      }
    }
  }
}
OCEOF
    log "  OpenCode config written to $opencode_dir/config.json"

    # --- Factory Droid ---
    # Uses settings.json (NOT config.json) with customModels array
    local factory_dir="$HOME/.factory"
    mkdir -p "$factory_dir"
    cat > "$factory_dir/settings.json" << 'FDEOF'
{
  "logoAnimation": "off",
  "customModels": [
    {
      "model": "claude-opus-4-6",
      "displayName": "Claude Opus 4.6 (proxy)",
      "baseUrl": "http://localhost:8317/v1",
      "apiKey": "rust-daq-proxy-key",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 16384
    },
    {
      "model": "claude-opus-4-6-thinking",
      "displayName": "Claude Opus 4.6 Thinking (proxy)",
      "baseUrl": "http://localhost:8317/v1",
      "apiKey": "rust-daq-proxy-key",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 32768
    },
    {
      "model": "claude-sonnet-4-5-20250929",
      "displayName": "Claude Sonnet 4.5 (proxy)",
      "baseUrl": "http://localhost:8317/v1",
      "apiKey": "rust-daq-proxy-key",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 16384
    },
    {
      "model": "gpt-5.2-codex",
      "displayName": "GPT-5.2 Codex (proxy)",
      "baseUrl": "http://localhost:8317/v1",
      "apiKey": "rust-daq-proxy-key",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 16384
    },
    {
      "model": "gemini-3-pro-preview",
      "displayName": "Gemini 3 Pro (proxy)",
      "baseUrl": "http://localhost:8317/v1",
      "apiKey": "rust-daq-proxy-key",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 65536
    },
    {
      "model": "or1-behemoth-q4_k_m",
      "displayName": "Beefcake 72B (local)",
      "baseUrl": "http://slurm-ctl.tailc46cd0.ts.net:8081/v1",
      "apiKey": "not-needed",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 8192
    },
    {
      "model": "strand-rust-coder-14b-q8_0",
      "displayName": "Beefcake 14B (local)",
      "baseUrl": "http://slurm-ctl.tailc46cd0.ts.net:8080/v1",
      "apiKey": "not-needed",
      "provider": "generic-chat-completion-api",
      "maxOutputTokens": 4096
    }
  ]
}
FDEOF
    log "  Factory Droid settings written to $factory_dir/settings.json"

    # --- Crush (Charmbracelet) ---
    # Crush uses environment variables for provider auto-detection.
    # OPENAI_API_KEY/BASE_URL and ANTHROPIC_API_KEY/BASE_URL are set in
    # /etc/profile.d/ai-swarm.sh (see configure_environment).
    log "  Crush uses env vars from /etc/profile.d/ai-swarm.sh"

    log "Agent CLI configuration complete"
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

# CLIProxyPlus (all cloud providers via single endpoint)
export CLI_PROXY_ENDPOINT="http://localhost:8317/v1/chat/completions"
export CLI_PROXY_KEY="rust-daq-proxy-key"

# Crush + other OpenAI-compat tools — uses env vars for provider detection
export OPENAI_API_KEY="rust-daq-proxy-key"
export OPENAI_BASE_URL="http://localhost:8317/v1"
export ANTHROPIC_API_KEY="rust-daq-proxy-key"
export ANTHROPIC_BASE_URL="http://localhost:8317/v1"

# Cargo/Rust
if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
fi

# Path — include droid, opencode
export PATH="/root/.local/bin:/root/.opencode/bin:$PATH"
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
    for agent in claude codex opencode droid crush; do
        if command -v "$agent" &>/dev/null; then
            log "  $agent: OK"
        else
            log "  $agent: NOT INSTALLED"
            ((failures++)) || true
        fi
    done

    # Check agent configs
    log "Checking agent configurations..."
    for cfg in "$HOME/.config/opencode/config.json" "$HOME/.factory/settings.json"; do
        if [[ -f "$cfg" ]]; then
            log "  $cfg: OK"
        else
            log "  $cfg: MISSING (run configure_agents)"
            ((failures++)) || true
        fi
    done

    # Check Crush env vars
    if [[ -n "${OPENAI_API_KEY:-}" ]] && [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        log "  Crush env vars: OK"
    else
        log "  Crush env vars: MISSING (source /etc/profile.d/ai-swarm.sh)"
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
    configure_agents
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
    configure-agents)
        configure_agents
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
        echo "Usage: $0 {inject-ssh-key|toolchain|agents|configure-agents|configure|verify|setup}"
        echo ""
        echo "  inject-ssh-key    Inject SSH key into CT 800 (run from local machine)"
        echo "  toolchain         Install Node.js, Rust, git, build-essential"
        echo "  agents            Install Claude Code, Codex, OpenCode, Droid, Crush"
        echo "  configure-agents  Write config files for OpenCode, Droid, Crush"
        echo "  configure         Set up environment variables"
        echo "  verify            Run end-to-end verification"
        echo "  setup             Full setup (Steps 0b-0e, run ON ai-proxy)"
        ;;
esac
