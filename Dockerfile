# Multi-stage Dockerfile for beefcake-swarm orchestrator
# Build: docker build -t beefcake-swarm .
# Run:   docker compose up swarm-agents

# --- Builder stage ---
FROM rust:1.83-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev libclang-dev clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY coordination/ coordination/
COPY crates/ crates/

# Build release binary
RUN cargo build --release -p swarm-agents

# --- Runtime stage ---
# Use rust:1.83-slim as base — includes rustfmt + clippy needed by verifier gates.
# Avoids curl|sh security risk and reduces image layers.
FROM rust:1.83-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN rustup component add rustfmt clippy

COPY --from=builder /build/target/release/swarm-agents /usr/local/bin/swarm-agents

# Default working directory for target repos
WORKDIR /workspace

ENTRYPOINT ["swarm-agents"]
