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
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install cargo fmt + clippy for verifier gates
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.83.0 --component rustfmt clippy
ENV PATH="/root/.cargo/bin:${PATH}"

COPY --from=builder /build/target/release/swarm-agents /usr/local/bin/swarm-agents

# Default working directory for target repos
WORKDIR /workspace

ENTRYPOINT ["swarm-agents"]
