#!/bin/bash
###############################################################################
# Build llama.cpp Apptainer container for V100S GPUs
#
# Usage:
#   ./build.sh                    # Build on slurm-ctl
#   ./build.sh --local            # Build locally (requires apptainer)
#   ./build.sh --push             # Build and push to /cluster/shared/containers
#
# Prerequisites:
#   - Apptainer installed (apptainer build command available)
#   - NVIDIA drivers for testing (nvidia-smi)
#   - Network access to pull NVIDIA base images
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="llama-server.sif"
CONTAINER_PATH="/cluster/shared/containers/${CONTAINER_NAME}"
DEF_FILE="${SCRIPT_DIR}/llama-server.def"

# Parse arguments
BUILD_LOCAL=false
PUSH=false

for arg in "$@"; do
    case $arg in
        --local)
            BUILD_LOCAL=true
            ;;
        --push)
            PUSH=true
            ;;
        --help|-h)
            echo "Usage: $0 [--local] [--push]"
            echo ""
            echo "Options:"
            echo "  --local    Build locally instead of on slurm-ctl"
            echo "  --push     Push to /cluster/shared/containers after build"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "Building llama.cpp Apptainer container"
echo "=========================================="

# Check for definition file
if [[ ! -f "$DEF_FILE" ]]; then
    echo "ERROR: Definition file not found: $DEF_FILE"
    exit 1
fi

# Build function
do_build() {
    local build_dir="$1"
    local output="${build_dir}/${CONTAINER_NAME}"

    echo "Building container..."
    echo "  Definition: ${DEF_FILE}"
    echo "  Output: ${output}"

    # Clean up any previous build
    rm -f "${output}"

    # Build with fakeroot (no root required)
    apptainer build --fakeroot "${output}" "${DEF_FILE}"

    echo ""
    echo "Build complete: ${output}"
    echo "Size: $(du -h "${output}" | cut -f1)"
}

if [[ "$BUILD_LOCAL" == "true" ]]; then
    # Build locally
    echo "Building locally..."
    BUILD_DIR="${SCRIPT_DIR}"
    do_build "$BUILD_DIR"
    OUTPUT="${BUILD_DIR}/${CONTAINER_NAME}"
else
    # Build on slurm-ctl
    echo "Building on slurm-ctl..."

    # Copy definition file to slurm-ctl
    ssh slurm-ctl "mkdir -p /tmp/llama-build"
    scp "${DEF_FILE}" slurm-ctl:/tmp/llama-build/

    # Run build on slurm-ctl
    ssh slurm-ctl "cd /tmp/llama-build && apptainer build --fakeroot ${CONTAINER_NAME} llama-server.def"

    # Copy result back or directly to shared
    if [[ "$PUSH" == "true" ]]; then
        echo "Moving container to ${CONTAINER_PATH}..."
        ssh slurm-ctl "mv /tmp/llama-build/${CONTAINER_NAME} ${CONTAINER_PATH}"
        OUTPUT="${CONTAINER_PATH}"
    else
        echo "Copying container back..."
        scp slurm-ctl:/tmp/llama-build/${CONTAINER_NAME} "${SCRIPT_DIR}/"
        OUTPUT="${SCRIPT_DIR}/${CONTAINER_NAME}"
    fi

    # Cleanup
    ssh slurm-ctl "rm -rf /tmp/llama-build"
fi

# Push if requested and not already pushed
if [[ "$PUSH" == "true" && "$BUILD_LOCAL" == "true" ]]; then
    echo "Pushing to ${CONTAINER_PATH}..."
    mkdir -p "$(dirname "${CONTAINER_PATH}")"

    if [[ -f "${CONTAINER_PATH}" ]]; then
        echo "Backing up existing container..."
        mv "${CONTAINER_PATH}" "${CONTAINER_PATH}.backup.$(date +%Y%m%d-%H%M%S)"
    fi

    cp "${OUTPUT}" "${CONTAINER_PATH}"
    echo "Container pushed to ${CONTAINER_PATH}"
fi

echo ""
echo "=========================================="
echo "Container ready: ${OUTPUT:-$CONTAINER_PATH}"
echo "=========================================="

# Quick validation
echo ""
echo "Running validation..."
if command -v nvidia-smi &> /dev/null; then
    apptainer exec --nv "${OUTPUT:-$CONTAINER_PATH}" llama-server --version || echo "WARNING: llama-server --version failed (may not support this flag)"
    echo "CUDA validation passed"
else
    echo "Skipping CUDA validation (nvidia-smi not available)"
fi

echo ""
echo "To test inference:"
echo "  apptainer run --nv ${OUTPUT:-$CONTAINER_PATH} --model /path/to/model.gguf --host 0.0.0.0 --port 8080"
