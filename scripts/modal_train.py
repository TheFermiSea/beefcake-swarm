#!/usr/bin/env python3
"""Modal-based QLoRA training for beefcake-swarm models.

Runs QLoRA fine-tuning on Modal's cloud GPUs (A100/H100) without blocking
local inference on the V100S cluster. Uploads training data, trains adapter,
downloads the result.

Usage:
    # Train SERA-14B adapter from local JSONL
    modal run scripts/modal_train.py --base allenai/SERA-14B \
        --data /tmp/combined-training-data.jsonl \
        --output /scratch/ai/adapters/sera-rust-v1

    # Train Qwen3.5-27B with custom rank
    modal run scripts/modal_train.py --base Qwen/Qwen3.5-27B \
        --data /tmp/trajectories.jsonl --rank 64 --epochs 3

    # DPO training with preference pairs
    modal run scripts/modal_train.py --base allenai/SERA-14B \
        --data /tmp/preference-pairs.jsonl --method dpo

    # Dry run (estimate cost/time without training)
    modal run scripts/modal_train.py --base allenai/SERA-14B \
        --data /tmp/training-data.jsonl --dry-run

Prerequisites:
    pip install modal
    modal setup  # one-time auth
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------

CUDA_VERSION = "12.4.0"
PYTHON_VERSION = "3.11"

training_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.46.0",
        "peft>=0.14.0",
        "trl>=0.12.0",
        "bitsandbytes>=0.44.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "flash-attn>=2.6.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("beefcake-lora-training", image=training_image)

# Persistent volume for caching base models across runs
model_cache = modal.Volume.from_name("beefcake-model-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

DEFAULTS = {
    "rank": 32,
    "alpha": 16,
    "dropout": 0.05,
    "epochs": 2,
    "batch_size": 4,
    "grad_accum": 4,
    "lr": 2e-4,
    "max_seq_length": 4096,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


# ---------------------------------------------------------------------------
# Core training function (runs on Modal GPU)
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=7200,  # 2 hour max
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface", required=False)],
)
def train_lora_sft(
    base_model: str,
    training_jsonl: bytes,
    config: dict,
) -> dict:
    """Run QLoRA SFT training on a Modal GPU. Returns adapter files as dict."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    rank = config.get("rank", DEFAULTS["rank"])
    alpha = config.get("alpha", DEFAULTS["alpha"])
    dropout = config.get("dropout", DEFAULTS["dropout"])
    epochs = config.get("epochs", DEFAULTS["epochs"])
    batch_size = config.get("batch_size", DEFAULTS["batch_size"])
    grad_accum = config.get("grad_accum", DEFAULTS["grad_accum"])
    lr = config.get("lr", DEFAULTS["lr"])
    max_seq = config.get("max_seq_length", DEFAULTS["max_seq_length"])
    target_modules = config.get("target_modules", DEFAULTS["target_modules"])

    output_dir = "/tmp/lora-output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load training data ---
    print(f"Loading training data...")
    rows = []
    for line in training_jsonl.decode("utf-8").strip().split("\n"):
        d = json.loads(line)
        rows.append({"messages": d["messages"]})
    print(f"  {len(rows)} samples loaded")
    ds = Dataset.from_list(rows)

    # --- Load model in 4-bit ---
    print(f"Loading {base_model} in 4-bit QLoRA mode...")
    cache_dir = "/cache/models"
    os.makedirs(cache_dir, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
    )
    model = prepare_model_for_kbit_training(model)

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Format data ---
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(format_chat)

    # --- Train ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=max_seq,
        dataset_text_field="text",
        packing=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"Starting training: {epochs} epochs, batch {batch_size}×{grad_accum}, rank {rank}...")
    result = trainer.train()
    print(f"Training complete! Loss: {result.training_loss:.4f}")

    # --- Save adapter ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # --- Collect output files ---
    output_files = {}
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath) and not fname.startswith("checkpoint"):
            with open(fpath, "rb") as f:
                output_files[fname] = f.read()
    output_files["_training_loss"] = str(result.training_loss).encode()
    output_files["_metrics"] = json.dumps(result.metrics).encode()

    # Persist model cache for next run
    model_cache.commit()

    print(f"Returning {len(output_files)} files")
    return output_files


@app.function(
    gpu="H100",
    timeout=7200,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface", required=False)],
)
def train_dpo(
    base_model: str,
    preference_jsonl: bytes,
    config: dict,
) -> dict:
    """Run DPO training on preference pairs. Returns adapter files."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig

    rank = config.get("rank", DEFAULTS["rank"])
    alpha = config.get("alpha", DEFAULTS["alpha"])
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 2)
    lr = config.get("lr", 5e-5)

    output_dir = "/tmp/dpo-output"
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = "/cache/models"
    os.makedirs(cache_dir, exist_ok=True)

    # --- Load preference data ---
    # Expected format: {"prompt": "...", "chosen": "...", "rejected": "..."}
    print("Loading preference data...")
    rows = []
    for line in preference_jsonl.decode("utf-8").strip().split("\n"):
        rows.append(json.loads(line))
    print(f"  {len(rows)} preference pairs loaded")
    ds = Dataset.from_list(rows)

    # --- Load model ---
    print(f"Loading {base_model} for DPO...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=DEFAULTS["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- DPO Training ---
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_length=4096,
        max_prompt_length=2048,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"Starting DPO training: {epochs} epochs...")
    result = trainer.train()
    print(f"DPO complete! Loss: {result.training_loss:.4f}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    output_files = {}
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath) and not fname.startswith("checkpoint"):
            with open(fpath, "rb") as f:
                output_files[fname] = f.read()
    output_files["_training_loss"] = str(result.training_loss).encode()

    model_cache.commit()
    return output_files


# ---------------------------------------------------------------------------
# Local entrypoint (runs on your machine, orchestrates Modal)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    base: str = "allenai/SERA-14B",
    data: str = "/tmp/combined-training-data.jsonl",
    output: str = "./adapters/modal-output",
    method: str = "sft",
    rank: int = DEFAULTS["rank"],
    alpha: int = DEFAULTS["alpha"],
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    grad_accum: int = DEFAULTS["grad_accum"],
    lr: float = DEFAULTS["lr"],
    max_seq_length: int = DEFAULTS["max_seq_length"],
    dry_run: bool = False,
):
    """Train a LoRA adapter on Modal cloud GPUs."""
    data_path = Path(data)
    if not data_path.exists():
        print(f"ERROR: Training data not found: {data}", file=sys.stderr)
        sys.exit(1)

    training_data = data_path.read_bytes()
    num_samples = sum(1 for _ in training_data.decode().strip().split("\n"))

    config = {
        "rank": rank,
        "alpha": alpha,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "max_seq_length": max_seq_length,
    }

    # --- Cost estimate ---
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, num_samples // effective_batch)
    total_steps = steps_per_epoch * epochs
    # Rough estimate: ~2s/step on H100 for 14B, ~5s/step for 27B
    is_large = "27" in base or "32" in base
    secs_per_step = 5 if is_large else 2
    est_minutes = (total_steps * secs_per_step) / 60
    est_cost = est_minutes * (5.49 / 60)  # H100 ~$5.49/hr on Modal

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  beefcake-swarm Modal Training                  ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Base model:  {base:<35s}║")
    print(f"║  Method:      {method.upper():<35s}║")
    print(f"║  Samples:     {num_samples:<35d}║")
    print(f"║  Rank:        {rank:<35d}║")
    print(f"║  Epochs:      {epochs:<35d}║")
    print(f"║  Batch:       {batch_size}×{grad_accum} = {effective_batch:<26d}║")
    print(f"║  Steps:       {total_steps:<35d}║")
    print(f"║  Est. time:   {est_minutes:<33.0f}m ║")
    print(f"║  Est. cost:   ${est_cost:<34.2f}║")
    print(f"║  GPU:         H100 80GB + FlashAttention2       ║")
    print(f"╚══════════════════════════════════════════════════╝")

    if dry_run:
        print("\n[DRY RUN] Would upload data and train on Modal. Exiting.")
        return

    print(f"\nUploading {len(training_data) / 1e6:.1f}MB training data to Modal...")

    if method == "dpo":
        result = train_dpo.remote(base, training_data, config)
    else:
        result = train_lora_sft.remote(base, training_data, config)

    # --- Download results ---
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    loss = "unknown"
    metrics = {}
    for fname, content in result.items():
        if fname == "_training_loss":
            loss = content.decode()
            continue
        if fname == "_metrics":
            metrics = json.loads(content.decode())
            continue
        fpath = output_path / fname
        fpath.write_bytes(content)
        size_mb = len(content) / 1e6
        print(f"  Saved: {fpath} ({size_mb:.1f}MB)")

    print(f"\nTraining loss: {loss}")
    if metrics:
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
    print(f"\nAdapter saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Convert to GGUF:")
    print(f"     bash scripts/convert-lora-to-gguf.sh --adapter-dir {output_path}")
    print(f"  2. Deploy:")
    print(f"     bash scripts/deploy-lora.sh --adapter <output>.gguf --node vasp-03")
