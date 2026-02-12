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
#
# All 30 proxy models organized by provider:
#   Antigravity (12): Gemini-hosted Claude, Gemini 3 previews, GPT-OSS, Tab models
#   Anthropic (8): Direct subscription Claude models
#   Codex/OpenAI (9): GPT-5.x series
#   Google (1): Gemini 2.5 Pro
#   Local (3): OR1 72B, Strand 14B, Qwen3-Coder-Next 80B MoE (via SLURM)
###############################################################################
configure_agents() {
    log "Configuring coding agent CLIs..."

    # --- OpenCode ---
    # Uses config.json with "provider" key, @ai-sdk/openai-compatible npm adapter
    # Models grouped by provider — each provider shows as a header in the UI
    local opencode_dir="$HOME/.config/opencode"
    mkdir -p "$opencode_dir"
    cat > "$opencode_dir/config.json" << 'OCEOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "antigravity": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Antigravity (Cloud)",
      "options": {
        "baseURL": "http://localhost:8317/v1",
        "apiKey": "rust-daq-proxy-key"
      },
      "models": {
        "claude-opus-4-6-thinking": {
          "name": "Claude Opus 4.6 Thinking",
          "limit": { "context": 200000, "output": 32768 }
        },
        "gemini-claude-opus-4-5-thinking": {
          "name": "Gemini Claude Opus 4.5 Thinking",
          "limit": { "context": 200000, "output": 32768 }
        },
        "gemini-claude-sonnet-4-5-thinking": {
          "name": "Gemini Claude Sonnet 4.5 Thinking",
          "limit": { "context": 200000, "output": 32768 }
        },
        "gemini-claude-sonnet-4-5": {
          "name": "Gemini Claude Sonnet 4.5",
          "limit": { "context": 200000, "output": 16384 }
        },
        "gemini-3-pro-preview": {
          "name": "Gemini 3 Pro Preview",
          "limit": { "context": 2000000, "output": 65536 }
        },
        "gemini-3-pro-image-preview": {
          "name": "Gemini 3 Pro Image Preview",
          "limit": { "context": 2000000, "output": 65536 }
        },
        "gemini-3-flash-preview": {
          "name": "Gemini 3 Flash Preview",
          "limit": { "context": 1000000, "output": 65536 }
        },
        "gemini-2.5-flash": {
          "name": "Gemini 2.5 Flash",
          "limit": { "context": 1000000, "output": 65536 }
        },
        "gemini-2.5-flash-lite": {
          "name": "Gemini 2.5 Flash Lite",
          "limit": { "context": 1000000, "output": 8192 }
        },
        "gpt-oss-120b-medium": {
          "name": "GPT-OSS 120B Medium",
          "limit": { "context": 128000, "output": 16384 }
        },
        "tab_flash_lite_preview": {
          "name": "Tab Flash Lite Preview",
          "limit": { "context": 1000000, "output": 8192 }
        },
        "tab_jump_flash_lite_preview": {
          "name": "Tab Jump Flash Lite Preview",
          "limit": { "context": 1000000, "output": 8192 }
        }
      }
    },
    "anthropic-sub": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Anthropic (Subscription)",
      "options": {
        "baseURL": "http://localhost:8317/v1",
        "apiKey": "rust-daq-proxy-key"
      },
      "models": {
        "claude-opus-4-5-20251101": {
          "name": "Claude Opus 4.5",
          "limit": { "context": 200000, "output": 32768 }
        },
        "claude-opus-4-1-20250805": {
          "name": "Claude Opus 4.1",
          "limit": { "context": 200000, "output": 16384 }
        },
        "claude-opus-4-20250514": {
          "name": "Claude Opus 4",
          "limit": { "context": 200000, "output": 16384 }
        },
        "claude-sonnet-4-5-20250929": {
          "name": "Claude Sonnet 4.5",
          "limit": { "context": 200000, "output": 16384 }
        },
        "claude-sonnet-4-20250514": {
          "name": "Claude Sonnet 4",
          "limit": { "context": 200000, "output": 16384 }
        },
        "claude-3-7-sonnet-20250219": {
          "name": "Claude 3.7 Sonnet",
          "limit": { "context": 200000, "output": 16384 }
        },
        "claude-haiku-4-5-20251001": {
          "name": "Claude Haiku 4.5",
          "limit": { "context": 200000, "output": 8192 }
        },
        "claude-3-5-haiku-20241022": {
          "name": "Claude 3.5 Haiku",
          "limit": { "context": 200000, "output": 8192 }
        }
      }
    },
    "codex-sub": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Codex (Subscription)",
      "options": {
        "baseURL": "http://localhost:8317/v1",
        "apiKey": "rust-daq-proxy-key"
      },
      "models": {
        "gpt-5.2-codex": {
          "name": "GPT-5.2 Codex",
          "limit": { "context": 128000, "output": 16384 }
        },
        "gpt-5.2": {
          "name": "GPT-5.2",
          "limit": { "context": 128000, "output": 16384 }
        },
        "gpt-5.1-codex-max": {
          "name": "GPT-5.1 Codex Max",
          "limit": { "context": 128000, "output": 32768 }
        },
        "gpt-5.1-codex": {
          "name": "GPT-5.1 Codex",
          "limit": { "context": 128000, "output": 16384 }
        },
        "gpt-5.1-codex-mini": {
          "name": "GPT-5.1 Codex Mini",
          "limit": { "context": 128000, "output": 16384 }
        },
        "gpt-5.1": {
          "name": "GPT-5.1",
          "limit": { "context": 128000, "output": 16384 }
        },
        "gpt-5-codex": {
          "name": "GPT-5 Codex",
          "limit": { "context": 128000, "output": 16384 }
        },
        "gpt-5-codex-mini": {
          "name": "GPT-5 Codex Mini",
          "limit": { "context": 128000, "output": 16384 }
        },
        "gpt-5": {
          "name": "GPT-5",
          "limit": { "context": 128000, "output": 16384 }
        }
      }
    },
    "google": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Google",
      "options": {
        "baseURL": "http://localhost:8317/v1",
        "apiKey": "rust-daq-proxy-key"
      },
      "models": {
        "gemini-2.5-pro": {
          "name": "Gemini 2.5 Pro",
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
    "beefcake-coder": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Beefcake Coder (Local)",
      "options": {
        "baseURL": "http://slurm-ctl.tailc46cd0.ts.net:8080/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "strand-rust-coder-14b-q8_0": {
          "name": "Strand Rust Coder 14B",
          "limit": { "context": 16384, "output": 4096 }
        },
        "Qwen3-Coder-Next-UD-Q4_K_XL.gguf": {
          "name": "Qwen3 Coder Next 80B MoE",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
OCEOF
    log "  OpenCode config written to $opencode_dir/config.json"

    # --- Factory Droid ---
    # Uses settings.json (NOT config.json) with customModels array
    # Display names use [Provider] prefix for identification
    local factory_dir="$HOME/.factory"
    mkdir -p "$factory_dir"
    cat > "$factory_dir/settings.json" << 'FDEOF'
{
  "logoAnimation": "off",
  "customModels": [
    { "model": "claude-opus-4-6-thinking", "displayName": "[Antigravity] Claude Opus 4.6 Thinking", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 32768 },
    { "model": "gemini-claude-opus-4-5-thinking", "displayName": "[Antigravity] Gemini-Claude Opus 4.5 Thinking", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 32768 },
    { "model": "gemini-claude-sonnet-4-5-thinking", "displayName": "[Antigravity] Gemini-Claude Sonnet 4.5 Thinking", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gemini-claude-sonnet-4-5", "displayName": "[Antigravity] Gemini-Claude Sonnet 4.5", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gemini-3-pro-preview", "displayName": "[Antigravity] Gemini 3 Pro Preview", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 65536 },
    { "model": "gemini-3-pro-image-preview", "displayName": "[Antigravity] Gemini 3 Pro Image Preview", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 32768 },
    { "model": "gemini-3-flash-preview", "displayName": "[Antigravity] Gemini 3 Flash Preview", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 32768 },
    { "model": "gemini-2.5-flash", "displayName": "[Antigravity] Gemini 2.5 Flash", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gemini-2.5-flash-lite", "displayName": "[Antigravity] Gemini 2.5 Flash Lite", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 },
    { "model": "gpt-oss-120b-medium", "displayName": "[Antigravity] GPT-OSS 120B Medium", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "tab_flash_lite_preview", "displayName": "[Antigravity] Tab Flash Lite Preview", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 },
    { "model": "tab_jump_flash_lite_preview", "displayName": "[Antigravity] Tab Jump Flash Lite Preview", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 },
    { "model": "claude-opus-4-5-20251101", "displayName": "[Anthropic Sub] Claude Opus 4.5", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "claude-opus-4-1-20250805", "displayName": "[Anthropic Sub] Claude Opus 4.1", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "claude-opus-4-20250514", "displayName": "[Anthropic Sub] Claude Opus 4", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "claude-sonnet-4-5-20250929", "displayName": "[Anthropic Sub] Claude Sonnet 4.5", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "claude-sonnet-4-20250514", "displayName": "[Anthropic Sub] Claude Sonnet 4", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "claude-haiku-4-5-20251001", "displayName": "[Anthropic Sub] Claude Haiku 4.5", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 },
    { "model": "claude-3-7-sonnet-20250219", "displayName": "[Anthropic Sub] Claude 3.7 Sonnet", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 },
    { "model": "claude-3-5-haiku-20241022", "displayName": "[Anthropic Sub] Claude 3.5 Haiku", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 4096 },
    { "model": "gpt-5.2-codex", "displayName": "[Codex Sub] GPT-5.2 Codex", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gpt-5.2", "displayName": "[Codex Sub] GPT-5.2", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gpt-5.1-codex-max", "displayName": "[Codex Sub] GPT-5.1 Codex Max", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gpt-5.1-codex", "displayName": "[Codex Sub] GPT-5.1 Codex", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gpt-5.1-codex-mini", "displayName": "[Codex Sub] GPT-5.1 Codex Mini", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 },
    { "model": "gpt-5.1", "displayName": "[Codex Sub] GPT-5.1", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gpt-5-codex", "displayName": "[Codex Sub] GPT-5 Codex", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gpt-5-codex-mini", "displayName": "[Codex Sub] GPT-5 Codex Mini", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 },
    { "model": "gpt-5", "displayName": "[Codex Sub] GPT-5", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 16384 },
    { "model": "gemini-2.5-pro", "displayName": "[Google] Gemini 2.5 Pro", "baseUrl": "http://localhost:8317/v1", "apiKey": "rust-daq-proxy-key", "provider": "generic-chat-completion-api", "maxOutputTokens": 65536 },
    { "model": "or1-behemoth-q4_k_m", "displayName": "[Local] Beefcake 72B", "baseUrl": "http://slurm-ctl.tailc46cd0.ts.net:8081/v1", "apiKey": "not-needed", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 },
    { "model": "strand-rust-coder-14b-q8_0", "displayName": "[Local] Beefcake 14B", "baseUrl": "http://slurm-ctl.tailc46cd0.ts.net:8080/v1", "apiKey": "not-needed", "provider": "generic-chat-completion-api", "maxOutputTokens": 4096 },
    { "model": "Qwen3-Coder-Next-UD-Q4_K_XL.gguf", "displayName": "[Local] Qwen3 Coder Next 80B", "baseUrl": "http://slurm-ctl.tailc46cd0.ts.net:8080/v1", "apiKey": "not-needed", "provider": "generic-chat-completion-api", "maxOutputTokens": 8192 }
  ]
}
FDEOF
    log "  Factory Droid settings written to $factory_dir/settings.json"

    # --- Crush (Charmbracelet) ---
    # Config file: ~/.config/crush/crush.json (NOT config.json)
    # Uses "openai" type for OpenAI-compat endpoints, grouped by provider
    # Also auto-discovers via OPENAI/ANTHROPIC env vars (see configure_environment)
    local crush_dir="$HOME/.config/crush"
    mkdir -p "$crush_dir"
    cat > "$crush_dir/crush.json" << 'CREOF'
{
  "default_provider": "antigravity",
  "providers": {
    "antigravity": {
      "name": "Antigravity (Cloud)",
      "base_url": "http://localhost:8317/v1/",
      "type": "openai",
      "api_key": "rust-daq-proxy-key",
      "models": [
        { "name": "[Antigravity] Claude Opus 4.6 Thinking", "id": "claude-opus-4-6-thinking", "context_window": 200000, "default_max_tokens": 32768 },
        { "name": "[Antigravity] Gemini Claude Opus 4.5 Thinking", "id": "gemini-claude-opus-4-5-thinking", "context_window": 200000, "default_max_tokens": 32768 },
        { "name": "[Antigravity] Gemini Claude Sonnet 4.5 Thinking", "id": "gemini-claude-sonnet-4-5-thinking", "context_window": 200000, "default_max_tokens": 32768 },
        { "name": "[Antigravity] Gemini Claude Sonnet 4.5", "id": "gemini-claude-sonnet-4-5", "context_window": 200000, "default_max_tokens": 16384 },
        { "name": "[Antigravity] Gemini 3 Pro Preview", "id": "gemini-3-pro-preview", "context_window": 2000000, "default_max_tokens": 65536 },
        { "name": "[Antigravity] Gemini 3 Pro Image Preview", "id": "gemini-3-pro-image-preview", "context_window": 2000000, "default_max_tokens": 65536 },
        { "name": "[Antigravity] Gemini 3 Flash Preview", "id": "gemini-3-flash-preview", "context_window": 1000000, "default_max_tokens": 65536 },
        { "name": "[Antigravity] Gemini 2.5 Flash", "id": "gemini-2.5-flash", "context_window": 1000000, "default_max_tokens": 65536 },
        { "name": "[Antigravity] Gemini 2.5 Flash Lite", "id": "gemini-2.5-flash-lite", "context_window": 1000000, "default_max_tokens": 8192 },
        { "name": "[Antigravity] GPT-OSS 120B Medium", "id": "gpt-oss-120b-medium", "context_window": 128000, "default_max_tokens": 16384 },
        { "name": "[Antigravity] Tab Flash Lite Preview", "id": "tab_flash_lite_preview", "context_window": 1000000, "default_max_tokens": 8192 },
        { "name": "[Antigravity] Tab Jump Flash Lite Preview", "id": "tab_jump_flash_lite_preview", "context_window": 1000000, "default_max_tokens": 8192 }
      ]
    },
    "anthropic-sub": {
      "name": "Anthropic (Subscription)",
      "base_url": "http://localhost:8317/v1/",
      "type": "openai",
      "api_key": "rust-daq-proxy-key",
      "models": [
        { "name": "[Anthropic] Claude Opus 4.5", "id": "claude-opus-4-5-20251101", "context_window": 200000, "default_max_tokens": 32768 },
        { "name": "[Anthropic] Claude Opus 4.1", "id": "claude-opus-4-1-20250805", "context_window": 200000, "default_max_tokens": 16384 },
        { "name": "[Anthropic] Claude Opus 4", "id": "claude-opus-4-20250514", "context_window": 200000, "default_max_tokens": 16384 },
        { "name": "[Anthropic] Claude Sonnet 4.5", "id": "claude-sonnet-4-5-20250929", "context_window": 200000, "default_max_tokens": 16384 },
        { "name": "[Anthropic] Claude Sonnet 4", "id": "claude-sonnet-4-20250514", "context_window": 200000, "default_max_tokens": 16384 },
        { "name": "[Anthropic] Claude 3.7 Sonnet", "id": "claude-3-7-sonnet-20250219", "context_window": 200000, "default_max_tokens": 16384 },
        { "name": "[Anthropic] Claude Haiku 4.5", "id": "claude-haiku-4-5-20251001", "context_window": 200000, "default_max_tokens": 8192 },
        { "name": "[Anthropic] Claude 3.5 Haiku", "id": "claude-3-5-haiku-20241022", "context_window": 200000, "default_max_tokens": 8192 }
      ]
    },
    "codex-sub": {
      "name": "Codex (Subscription)",
      "base_url": "http://localhost:8317/v1/",
      "type": "openai",
      "api_key": "rust-daq-proxy-key",
      "models": [
        { "name": "[Codex] GPT-5.2 Codex", "id": "gpt-5.2-codex", "context_window": 128000, "default_max_tokens": 16384 },
        { "name": "[Codex] GPT-5.2", "id": "gpt-5.2", "context_window": 128000, "default_max_tokens": 16384 },
        { "name": "[Codex] GPT-5.1 Codex Max", "id": "gpt-5.1-codex-max", "context_window": 128000, "default_max_tokens": 32768 },
        { "name": "[Codex] GPT-5.1 Codex", "id": "gpt-5.1-codex", "context_window": 128000, "default_max_tokens": 16384 },
        { "name": "[Codex] GPT-5.1 Codex Mini", "id": "gpt-5.1-codex-mini", "context_window": 128000, "default_max_tokens": 16384 },
        { "name": "[Codex] GPT-5.1", "id": "gpt-5.1", "context_window": 128000, "default_max_tokens": 16384 },
        { "name": "[Codex] GPT-5 Codex", "id": "gpt-5-codex", "context_window": 128000, "default_max_tokens": 16384 },
        { "name": "[Codex] GPT-5 Codex Mini", "id": "gpt-5-codex-mini", "context_window": 128000, "default_max_tokens": 16384 },
        { "name": "[Codex] GPT-5", "id": "gpt-5", "context_window": 128000, "default_max_tokens": 16384 }
      ]
    },
    "google": {
      "name": "Google",
      "base_url": "http://localhost:8317/v1/",
      "type": "openai",
      "api_key": "rust-daq-proxy-key",
      "models": [
        { "name": "[Google] Gemini 2.5 Pro", "id": "gemini-2.5-pro", "context_window": 2000000, "default_max_tokens": 65536 }
      ]
    },
    "beefcake-72b": {
      "name": "Beefcake 72B (Local)",
      "base_url": "http://slurm-ctl.tailc46cd0.ts.net:8081/v1/",
      "type": "openai",
      "api_key": "not-needed",
      "models": [
        { "name": "[Local] OR1 Behemoth 72B", "id": "or1-behemoth-q4_k_m", "context_window": 32768, "default_max_tokens": 8192 }
      ]
    },
    "beefcake-coder": {
      "name": "Beefcake Coder (Local)",
      "base_url": "http://slurm-ctl.tailc46cd0.ts.net:8080/v1/",
      "type": "openai",
      "api_key": "not-needed",
      "models": [
        { "name": "[Local] Strand Rust Coder 14B", "id": "strand-rust-coder-14b-q8_0", "context_window": 16384, "default_max_tokens": 4096 },
        { "name": "[Local] Qwen3 Coder Next 80B", "id": "Qwen3-Coder-Next-UD-Q4_K_XL.gguf", "context_window": 32768, "default_max_tokens": 8192 }
      ]
    }
  },
  "models": {
    "large": {
      "model": "claude-opus-4-5-20251101",
      "provider": "anthropic-sub"
    },
    "small": {
      "model": "claude-sonnet-4-5-20250929",
      "provider": "anthropic-sub"
    }
  }
}
CREOF
    log "  Crush config written to $crush_dir/crush.json"

    # --- Claude Code ---
    # Uses ~/.claude/settings.json with env overrides
    # Proxy supports Anthropic Messages API (/v1/messages, /v1/messages/count_tokens)
    # so Claude Code can use ALL proxy models (Anthropic, Antigravity, Codex, Google)
    # Switch models at runtime: /model <name> or claude --model <name>
    local claude_dir="$HOME/.claude"
    mkdir -p "$claude_dir"
    cat > "$claude_dir/settings.json" << 'CLEOF'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8317",
    "ANTHROPIC_AUTH_TOKEN": "rust-daq-proxy-key",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-5-20251101",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-5-20250929",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5-20251001",
    "DISABLE_PROMPT_CACHING": "1"
  },
  "model": "sonnet"
}
CLEOF
    log "  Claude Code settings written to $claude_dir/settings.json"
    log "  Available models via /model: any of the 30 proxy models"
    log "  Aliases: opus → Opus 4.5, sonnet → Sonnet 4.5, haiku → Haiku 4.5"
    log "  Antigravity: claude --model gemini-3-pro-preview"
    log "  Codex: claude --model gpt-5.2-codex"

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
export ANTHROPIC_BASE_URL="http://localhost:8317"

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
    for cfg in "$HOME/.claude/settings.json" "$HOME/.config/opencode/config.json" "$HOME/.factory/settings.json" "$HOME/.config/crush/crush.json"; do
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
