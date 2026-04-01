#!/usr/bin/env bash
# train-lora.sh — QLoRA fine-tuning for local swarm models.
#
# Uses unsloth for 4x faster QLoRA training on V100S 32GB.
# Reads chat-style JSONL produced by extract-training-data.py.
# Outputs safetensors adapter + GGUF LoRA for llama.cpp --lora flag.
#
# Usage:
#   ssh root@10.0.0.20 "bash /path/to/train-lora.sh --data training_data.jsonl --name agentic-coding-v1"
#
# Options:
#   --data <path>       Training data JSONL file (required)
#   --name <string>     Adapter name (required, e.g., "agentic-coding-v1")
#   --config <path>     TOML config override (default: config/lora-training.toml)
#   --epochs <int>      Override training epochs
#   --lr <float>        Override learning rate
#   --rank <int>        Override LoRA rank
#   --max-seq <int>     Override max sequence length
#   --dry-run           Print config and exit without training
#
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_FILE=""
ADAPTER_NAME=""
CONFIG_FILE="${REPO_ROOT}/config/lora-training.toml"
ADAPTER_DIR="/scratch/ai/adapters"
DRY_RUN=false

# Overridable hyperparameters (empty = use config/defaults)
OVERRIDE_EPOCHS=""
OVERRIDE_LR=""
OVERRIDE_RANK=""
OVERRIDE_MAX_SEQ=""

# ── CLI parsing ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)       DATA_FILE="$2"; shift 2 ;;
    --name)       ADAPTER_NAME="$2"; shift 2 ;;
    --config)     CONFIG_FILE="$2"; shift 2 ;;
    --epochs)     OVERRIDE_EPOCHS="$2"; shift 2 ;;
    --lr)         OVERRIDE_LR="$2"; shift 2 ;;
    --rank)       OVERRIDE_RANK="$2"; shift 2 ;;
    --max-seq)    OVERRIDE_MAX_SEQ="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=true; shift ;;
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

if [[ -z "$DATA_FILE" ]]; then
  echo "ERROR: --data is required" >&2
  exit 1
fi
if [[ -z "$ADAPTER_NAME" ]]; then
  echo "ERROR: --name is required" >&2
  exit 1
fi

# ── Validation ───────────────────────────────────────────────────────────────

log() { echo "[train-lora] $(date +%H:%M:%S) $*"; }

if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: Training data not found: $DATA_FILE" >&2
  exit 1
fi

SAMPLE_COUNT=$(wc -l < "$DATA_FILE")
log "Training data: $DATA_FILE ($SAMPLE_COUNT samples)"

if [[ "$SAMPLE_COUNT" -lt 10 ]]; then
  echo "ERROR: Too few training samples ($SAMPLE_COUNT). Need at least 10." >&2
  exit 1
fi

# Verify GPU is available
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Run this on a GPU node (vasp-01/02/03)." >&2
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
log "GPU: $GPU_NAME (${GPU_MEM} MiB)"

# ── Output directory ─────────────────────────────────────────────────────────

OUTPUT_DIR="${ADAPTER_DIR}/${ADAPTER_NAME}"
mkdir -p "$OUTPUT_DIR"
log "Output directory: $OUTPUT_DIR"

# ── Install dependencies ─────────────────────────────────────────────────────

log "Checking Python environment..."

# Prefer the system python3; create a venv if unsloth is not already installed.
VENV_DIR="/scratch/ai/venvs/lora-training"
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtualenv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install unsloth + deps if missing.  pip install is idempotent.
if ! python3 -c "import unsloth" &>/dev/null; then
  log "Installing unsloth and dependencies..."
  pip install --quiet --upgrade pip
  pip install --quiet "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
  pip install --quiet trl datasets toml
fi

# ── Generate training script ─────────────────────────────────────────────────

TRAIN_SCRIPT=$(mktemp /tmp/train_lora_XXXX.py)
trap 'rm -f "$TRAIN_SCRIPT"' EXIT

cat > "$TRAIN_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""QLoRA fine-tuning with unsloth — launched by train-lora.sh."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import toml
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


def load_config(config_path: str) -> dict:
    """Load TOML config with sensible defaults."""
    defaults = {
        "base_model": {
            "source": "unsloth/Devstral-Small-2-24B-Instruct",
        },
        "qlora": {
            "rank": 32,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
        "training": {
            "epochs": 2,
            "learning_rate": 2e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 4096,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "lr_scheduler": "cosine",
            "logging_steps": 10,
            "save_steps": 100,
            "seed": 42,
        },
    }
    if os.path.exists(config_path):
        cfg = toml.load(config_path)
        # Merge: config values override defaults
        for section in defaults:
            if section in cfg:
                defaults[section].update(cfg[section])
    return defaults


def load_training_data(data_path: str) -> Dataset:
    """Load JSONL with chat messages format.

    Expected format per line:
      {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
       "metadata": {"episode_id": "...", "model": "...", "iterations": N}}
    """
    records = []
    with open(data_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {line_num}: {e}", file=sys.stderr)
                continue
            if "messages" not in record:
                print(f"WARNING: Skipping line {line_num}: no 'messages' field", file=sys.stderr)
                continue
            records.append(record)

    if not records:
        print("ERROR: No valid training records found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} training examples")
    return Dataset.from_list(records)


def format_chat(example: dict, tokenizer) -> dict:
    """Apply the model's chat template to the messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with unsloth")
    parser.add_argument("--data", required=True, help="Path to training JSONL")
    parser.add_argument("--output-dir", required=True, help="Adapter output directory")
    parser.add_argument("--config", default="", help="TOML config path")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max-seq", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.epochs > 0:
        cfg["training"]["epochs"] = args.epochs
    if args.lr > 0:
        cfg["training"]["learning_rate"] = args.lr
    if args.rank > 0:
        cfg["qlora"]["rank"] = args.rank
    if args.max_seq > 0:
        cfg["training"]["max_seq_length"] = args.max_seq

    model_source = cfg["base_model"]["source"]
    max_seq_length = cfg["training"]["max_seq_length"]
    lora_rank = cfg["qlora"]["rank"]
    lora_alpha = cfg["qlora"]["alpha"]
    lora_dropout = cfg["qlora"]["dropout"]
    target_modules = cfg["qlora"]["target_modules"]

    print(f"{'='*60}")
    print(f"QLoRA Fine-Tuning Configuration")
    print(f"{'='*60}")
    print(f"  Base model:       {model_source}")
    print(f"  LoRA rank:        {lora_rank}")
    print(f"  LoRA alpha:       {lora_alpha}")
    print(f"  LoRA dropout:     {lora_dropout}")
    print(f"  Target modules:   {target_modules}")
    print(f"  Max seq length:   {max_seq_length}")
    print(f"  Epochs:           {cfg['training']['epochs']}")
    print(f"  Batch size:       {cfg['training']['batch_size']}")
    print(f"  Grad accum:       {cfg['training']['gradient_accumulation_steps']}")
    print(f"  Learning rate:    {cfg['training']['learning_rate']}")
    print(f"  Warmup ratio:     {cfg['training']['warmup_ratio']}")
    print(f"  Weight decay:     {cfg['training']['weight_decay']}")
    print(f"  LR scheduler:     {cfg['training']['lr_scheduler']}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Training data:    {args.data}")
    print(f"  GPU:              {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[dry-run] Would train with the above config. Exiting.")
        sys.exit(0)

    # ── Load model with unsloth 4-bit quantization ──────────────────────

    print(f"\nLoading {model_source} with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect (bf16 via software emulation on V100)
        load_in_4bit=True,
    )

    # ── Apply LoRA adapters ─────────────────────────────────────────────

    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=cfg["training"]["seed"],
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # ── Load and format training data ───────────────────────────────────

    dataset = load_training_data(args.data)
    dataset = dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=dataset.column_names,
    )
    print(f"Formatted {len(dataset)} examples")

    # ── Training ────────────────────────────────────────────────────────

    effective_batch = (
        cfg["training"]["batch_size"]
        * cfg["training"]["gradient_accumulation_steps"]
    )
    print(f"\nStarting training (effective batch size: {effective_batch})...")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=3,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        seed=cfg["training"]["seed"],
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,  # pack short examples together for efficiency
    )

    train_result = trainer.train()

    # ── Save results ────────────────────────────────────────────────────

    print(f"\nTraining complete. Metrics:")
    for key, value in sorted(train_result.metrics.items()):
        print(f"  {key}: {value}")

    # Save adapter in safetensors format (HuggingFace compatible)
    safetensors_dir = os.path.join(args.output_dir, "safetensors")
    print(f"\nSaving safetensors adapter to {safetensors_dir} ...")
    model.save_pretrained(safetensors_dir)
    tokenizer.save_pretrained(safetensors_dir)

    # Save training metadata
    metadata = {
        "base_model": model_source,
        "adapter_name": os.path.basename(args.output_dir),
        "training_samples": len(dataset),
        "training_data": os.path.basename(args.data),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["learning_rate"],
        "metrics": train_result.metrics,
    }
    metadata_path = os.path.join(args.output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {metadata_path}")

    print(f"\nAdapter saved. Next step: convert to GGUF LoRA format:")
    print(f"  See train-lora.sh post-training GGUF conversion output below.")


if __name__ == "__main__":
    main()
PYTHON_EOF

# ── Launch training ──────────────────────────────────────────────────────────

PYTHON_ARGS=(
  --data "$DATA_FILE"
  --output-dir "$OUTPUT_DIR"
  --config "$CONFIG_FILE"
)
[[ -n "$OVERRIDE_EPOCHS" ]] && PYTHON_ARGS+=(--epochs "$OVERRIDE_EPOCHS")
[[ -n "$OVERRIDE_LR" ]]     && PYTHON_ARGS+=(--lr "$OVERRIDE_LR")
[[ -n "$OVERRIDE_RANK" ]]   && PYTHON_ARGS+=(--rank "$OVERRIDE_RANK")
[[ -n "$OVERRIDE_MAX_SEQ" ]] && PYTHON_ARGS+=(--max-seq "$OVERRIDE_MAX_SEQ")
$DRY_RUN && PYTHON_ARGS+=(--dry-run)

log "Launching training..."
python3 "$TRAIN_SCRIPT" "${PYTHON_ARGS[@]}"

if $DRY_RUN; then
  exit 0
fi

# ── GGUF LoRA conversion ────────────────────────────────────────────────────

SAFETENSORS_DIR="${OUTPUT_DIR}/safetensors"
GGUF_OUTPUT="${OUTPUT_DIR}/${ADAPTER_NAME}.gguf"

# Read base model source from the metadata saved by the training script.
METADATA_FILE="${OUTPUT_DIR}/training_metadata.json"
BASE_MODEL_SOURCE=$(python3 -c "import json; print(json.load(open('${METADATA_FILE}'))['base_model'])" 2>/dev/null \
  || python3 -c "import toml; c=toml.load('${CONFIG_FILE}'); print(c['base_model']['source'])" 2>/dev/null \
  || echo "unsloth/Devstral-Small-2-24B-Instruct")

log "Converting adapter to GGUF LoRA format (base: ${BASE_MODEL_SOURCE})..."

# Try llama-export-lora first (llama.cpp native tool), fall back to Python converter.
if command -v llama-export-lora &>/dev/null; then
  log "Using llama-export-lora for GGUF conversion"
  llama-export-lora \
    --model-base "${SAFETENSORS_DIR}" \
    --lora "${SAFETENSORS_DIR}" \
    --output "${GGUF_OUTPUT}"
elif [[ -f "/cluster/shared/llama-cpp/convert_lora_to_gguf.py" ]]; then
  log "Using convert_lora_to_gguf.py for GGUF conversion"
  python3 /cluster/shared/llama-cpp/convert_lora_to_gguf.py \
    --base "${BASE_MODEL_SOURCE}" \
    "${SAFETENSORS_DIR}" \
    --outfile "${GGUF_OUTPUT}"
else
  # Try finding it in the llama.cpp source tree
  CONVERTER=$(find /cluster/shared/llama-cpp -name "convert_lora_to_gguf.py" 2>/dev/null | head -1)
  if [[ -n "$CONVERTER" ]]; then
    log "Using $CONVERTER for GGUF conversion"
    python3 "$CONVERTER" \
      --base "${BASE_MODEL_SOURCE}" \
      "${SAFETENSORS_DIR}" \
      --outfile "${GGUF_OUTPUT}"
  else
    log "WARNING: No GGUF converter found. Safetensors saved at:"
    log "  ${SAFETENSORS_DIR}"
    log "Convert manually with: python3 convert_lora_to_gguf.py --base <model> ${SAFETENSORS_DIR} --outfile ${GGUF_OUTPUT}"
    exit 0
  fi
fi

if [[ -f "$GGUF_OUTPUT" ]]; then
  GGUF_SIZE=$(du -h "$GGUF_OUTPUT" | cut -f1)
  log "GGUF LoRA adapter created: $GGUF_OUTPUT ($GGUF_SIZE)"
  log ""
  log "Deploy with:"
  log "  bash scripts/deploy-lora.sh --adapter $GGUF_OUTPUT --node vasp-03 --scale 1.0"
else
  log "ERROR: GGUF conversion produced no output file" >&2
  exit 1
fi
