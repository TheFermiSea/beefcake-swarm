#!/usr/bin/env bash
# convert-lora-to-gguf.sh — Convert a HuggingFace PEFT LoRA adapter to GGUF format.
#
# Requires: llama.cpp source with convert_lora_to_gguf.py (on vasp-03 at /scratch/build)
#
# Usage:
#   bash convert-lora-to-gguf.sh --adapter-dir /scratch/ai/adapters/rust-coder-v1 \
#                                 --output /scratch/ai/adapters/rust-coder-v1.gguf
#
set -euo pipefail

ADAPTER_DIR=""
OUTPUT=""
CONVERT_SCRIPT="/scratch/build/llama.cpp-src/convert_lora_to_gguf.py"
CONVERT_NODE="root@10.0.0.22"  # vasp-03 has the llama.cpp source

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-dir)  ADAPTER_DIR="$2"; shift 2 ;;
    --output)       OUTPUT="$2"; shift 2 ;;
    --script)       CONVERT_SCRIPT="$2"; shift 2 ;;
    --node)         CONVERT_NODE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ADAPTER_DIR" ]]; then
  echo "ERROR: --adapter-dir is required" >&2
  exit 1
fi

# Default output name from adapter dir
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${ADAPTER_DIR}.gguf"
fi

log() { echo "[convert-lora] $(date +%H:%M:%S) $*"; }

# The adapter dir is on NFS (/scratch), accessible from all nodes.
# The conversion script is on vasp-03, so run there.

log "Verifying adapter files at ${ADAPTER_DIR}..."
ssh "$CONVERT_NODE" "ls ${ADAPTER_DIR}/adapter_config.json ${ADAPTER_DIR}/adapter_model.safetensors 2>/dev/null" || {
  echo "ERROR: Missing adapter_config.json or adapter_model.safetensors in ${ADAPTER_DIR}" >&2
  exit 1
}

log "Converting to GGUF format..."
ssh "$CONVERT_NODE" "cd /scratch/build/llama.cpp-src && python ${CONVERT_SCRIPT} ${ADAPTER_DIR} --outfile ${OUTPUT} --outtype f16" 2>&1

if ssh "$CONVERT_NODE" "test -f '${OUTPUT}'"; then
  SIZE=$(ssh "$CONVERT_NODE" "du -h '${OUTPUT}' | cut -f1")
  log "Success! Adapter GGUF: ${OUTPUT} (${SIZE})"
  log "Deploy with: bash scripts/deploy-lora.sh --adapter ${OUTPUT} --node vasp-03"
else
  echo "ERROR: Conversion failed — output file not created" >&2
  exit 1
fi
