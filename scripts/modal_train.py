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
    )
    # Note: flash-attn omitted — requires building from source with torch present.
    # Falls back to PyTorch SDPA attention, which is nearly equivalent on H100.
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

# Conservative overrides for large models (20B+) to fit H100 80GB.
# 27B in 4-bit ≈ 15GB weights + activations. batch=4 × seq=4096 OOMs.
# batch=1 × grad_accum=16 preserves effective batch size of 16.
LARGE_MODEL_OVERRIDES = {
    "batch_size": 1,
    "grad_accum": 16,
    "max_seq_length": 2048,
}


def _is_large_model(name: str) -> bool:
    """Detect models that need reduced batch/seq for H100 80GB."""
    return any(s in name.lower() for s in ["27b", "32b", "33b", "34b", "70b"])


# ---------------------------------------------------------------------------
# Core training function (runs on Modal GPU)
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=7200,  # 2 hour max
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface")],
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
        attn_implementation="sdpa",
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
        max_length=max_seq,
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
    secrets=[modal.Secret.from_name("huggingface")],
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
        attn_implementation="sdpa",
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
# Evaluation function (runs on Modal GPU)
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_adapter(
    base_model: str,
    adapter_files: dict,
    holdout_jsonl: bytes,
) -> dict:
    """Evaluate a LoRA adapter vs base model on a holdout set.

    Computes perplexity for both configurations on the same data.
    Lower perplexity = better model fit = the model is more confident
    about the correct completions.

    Returns dict with base_ppl, adapter_ppl, improvement_pct, and pass/fail.
    """
    import math
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    cache_dir = "/cache/models"
    os.makedirs(cache_dir, exist_ok=True)

    # --- Load holdout data ---
    print("Loading holdout data...")
    rows = []
    for line in holdout_jsonl.decode("utf-8").strip().split("\n"):
        d = json.loads(line)
        rows.append({"messages": d["messages"]})
    print(f"  {len(rows)} holdout samples")

    if len(rows) < 5:
        print("WARNING: Very small holdout set — evaluation may be noisy")

    # --- Load tokenizer and format data ---
    print(f"Loading tokenizer for {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format holdout as text
    texts = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    # Tokenize with truncation
    max_len = 2048
    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_len, padding=True
    )

    def compute_perplexity(model, encodings):
        """Compute perplexity over a set of encodings."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        device = next(model.parameters()).device

        with torch.no_grad():
            for i in range(len(texts)):
                input_ids = encodings["input_ids"][i:i+1].to(device)
                attention_mask = encodings["attention_mask"][i:i+1].to(device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item() * attention_mask.sum().item()
                total_tokens += attention_mask.sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        return math.exp(avg_loss), avg_loss

    # --- Evaluate base model ---
    print(f"Loading {base_model} in 4-bit for base evaluation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )

    print("Computing base model perplexity...")
    base_ppl, base_loss = compute_perplexity(base, encodings)
    print(f"  Base perplexity: {base_ppl:.2f} (loss: {base_loss:.4f})")

    # --- Save adapter files to disk for PeftModel.from_pretrained ---
    adapter_dir = "/tmp/eval-adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    for fname, content in adapter_files.items():
        if not fname.startswith("_"):
            with open(os.path.join(adapter_dir, fname), "wb") as f:
                f.write(content)

    # --- Evaluate adapter model ---
    print("Applying LoRA adapter...")
    adapter_model = PeftModel.from_pretrained(base, adapter_dir)
    adapter_model = adapter_model.merge_and_unload()

    print("Computing adapter perplexity...")
    adapter_ppl, adapter_loss = compute_perplexity(adapter_model, encodings)
    print(f"  Adapter perplexity: {adapter_ppl:.2f} (loss: {adapter_loss:.4f})")

    # --- Compare ---
    improvement = (base_ppl - adapter_ppl) / base_ppl * 100
    passed = adapter_ppl < base_ppl

    print(f"\n{'='*50}")
    print(f"  Base perplexity:    {base_ppl:.2f}")
    print(f"  Adapter perplexity: {adapter_ppl:.2f}")
    print(f"  Improvement:        {improvement:+.1f}%")
    print(f"  Verdict:            {'PASS' if passed else 'FAIL'}")
    print(f"{'='*50}")

    model_cache.commit()

    return {
        "base_ppl": base_ppl,
        "base_loss": base_loss,
        "adapter_ppl": adapter_ppl,
        "adapter_loss": adapter_loss,
        "improvement_pct": improvement,
        "holdout_samples": len(rows),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Generation-based evaluation (v3 compile-check proxy)
# ---------------------------------------------------------------------------

def _score_rust_fix(code_block: str, full_response: str) -> dict:
    """Score a single Rust code fix using structural heuristics.

    Returns a dict with individual scores and a total (0-10).
    Since we can't run cargo check on Modal (no Rust toolchain / project
    context), we use lightweight heuristics that catch the most common
    failure modes of LLM-generated Rust fixes.
    """
    import re

    scores = {}

    # (1) Contains a code block? (+2)
    scores["has_code_block"] = 2 if code_block else 0

    if not code_block:
        scores["balanced_braces"] = 0
        scores["no_stubs"] = 0
        scores["no_bare_unwrap"] = 0
        scores["has_explanation"] = 0
        scores["modifies_code"] = 0
        scores["total"] = 0
        return scores

    # (2) Balanced braces / parens / angle brackets? (+2)
    open_braces = code_block.count("{") - code_block.count("}")
    open_parens = code_block.count("(") - code_block.count(")")
    open_angles = code_block.count("<") - code_block.count(">")
    balanced = abs(open_braces) <= 1 and abs(open_parens) <= 1 and abs(open_angles) <= 1
    scores["balanced_braces"] = 2 if balanced else 0

    # (3) No todo!/unimplemented! stubs? (+2)
    has_stubs = bool(re.search(r'\b(todo|unimplemented|panic)\s*!\s*\(', code_block))
    scores["no_stubs"] = 0 if has_stubs else 2

    # (4) No bare unwrap()? (+1)
    # Allow unwrap in test code (lines containing #[test] or mod tests)
    non_test_lines = []
    in_test = False
    for line in code_block.split("\n"):
        if "#[test]" in line or "#[cfg(test)]" in line or "mod tests" in line:
            in_test = True
        if not in_test:
            non_test_lines.append(line)
    non_test_code = "\n".join(non_test_lines)
    has_bare_unwrap = bool(re.search(r'\.unwrap\(\)', non_test_code))
    scores["no_bare_unwrap"] = 0 if has_bare_unwrap else 1

    # (5) Response explains the fix? (+1)
    # Look for explanation text outside the code block
    explanation = full_response.replace(code_block, "").strip()
    has_explanation = len(explanation) > 30
    scores["has_explanation"] = 1 if has_explanation else 0

    # (6) Fix contains substantive code (not just comments/whitespace)? (+2)
    code_lines = [
        l.strip() for l in code_block.split("\n")
        if l.strip() and not l.strip().startswith("//")
    ]
    scores["modifies_code"] = 2 if len(code_lines) >= 2 else 0

    scores["total"] = sum(v for k, v in scores.items() if k != "total")
    return scores


def _extract_rust_code_blocks(text: str) -> list[str]:
    """Extract all ```rust ... ``` code blocks from text. Falls back to ``` ... ```."""
    import re

    # Try ```rust blocks first
    blocks = re.findall(r"```rust\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks

    # Fall back to generic code blocks
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return blocks


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate_generation(
    base_model: str,
    adapter_files: dict,
    eval_prompts_jsonl: bytes,
) -> dict:
    """Generate fixes with base model and adapter, compare structural quality.

    This is the v3 "compile-check" evaluation. Since we cannot run cargo check
    on Modal (no Rust toolchain or project context), we generate actual fixes
    from each model and score them with structural heuristics that catch the
    most common LLM failure modes: missing code blocks, unbalanced braces,
    stub placeholders, bare unwrap(), lack of explanation, and trivial output.

    For each eval prompt the function:
      1. Generates a fix with the base model (no adapter)
      2. Generates a fix with the adapter model
      3. Scores each fix 0-10 using heuristics
      4. Compares average scores

    Returns dict with base_avg, adapter_avg, improvement_pct, per-sample
    detail, and pass/fail verdict.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    cache_dir = "/cache/models"
    os.makedirs(cache_dir, exist_ok=True)

    # --- Load eval prompts ---
    # Each line is a JSON object with "messages" containing the conversation
    # up to (but not including) the assistant's fix. We strip the final
    # assistant message to create the generation prompt.
    print("Loading eval prompts...")
    prompts = []
    for line in eval_prompts_jsonl.decode("utf-8").strip().split("\n"):
        d = json.loads(line)
        msgs = d["messages"]
        # Keep only system + user messages (drop final assistant response)
        prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]
        if not prompt_msgs:
            prompt_msgs = msgs[:-1] if len(msgs) > 1 else msgs
        prompts.append(prompt_msgs)
    print(f"  {len(prompts)} eval prompts loaded")

    if not prompts:
        return {
            "error": "No eval prompts provided",
            "passed": False,
        }

    # --- Load tokenizer ---
    print(f"Loading tokenizer for {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load base model in 4-bit ---
    print(f"Loading {base_model} in 4-bit for generation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        cache_dir=cache_dir,
    )

    def generate_fix(current_model, prompt_messages, max_new_tokens=1024):
        """Generate a fix response from a model given chat messages."""
        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072)
        inputs = {k: v.to(current_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = current_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only the generated tokens (skip input)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    # --- Generate with base model ---
    print("Generating fixes with base model...")
    base_results = []
    for i, prompt_msgs in enumerate(prompts):
        response = generate_fix(model, prompt_msgs)
        code_blocks = _extract_rust_code_blocks(response)
        code_block = code_blocks[0] if code_blocks else ""
        score = _score_rust_fix(code_block, response)
        base_results.append({
            "index": i,
            "response_len": len(response),
            "code_block_count": len(code_blocks),
            "scores": score,
        })
        if (i + 1) % 5 == 0:
            print(f"  Base: {i + 1}/{len(prompts)} done")
    print(f"  Base model: {len(base_results)} generations complete")

    # --- Apply adapter ---
    print("Saving adapter files to disk...")
    adapter_dir = "/tmp/gen-eval-adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    for fname, content in adapter_files.items():
        if not fname.startswith("_"):
            with open(os.path.join(adapter_dir, fname), "wb") as f:
                f.write(content)

    print("Applying LoRA adapter...")
    adapter_model = PeftModel.from_pretrained(model, adapter_dir)
    adapter_model = adapter_model.merge_and_unload()

    # --- Generate with adapter model ---
    print("Generating fixes with adapter model...")
    adapter_results = []
    for i, prompt_msgs in enumerate(prompts):
        response = generate_fix(adapter_model, prompt_msgs)
        code_blocks = _extract_rust_code_blocks(response)
        code_block = code_blocks[0] if code_blocks else ""
        score = _score_rust_fix(code_block, response)
        adapter_results.append({
            "index": i,
            "response_len": len(response),
            "code_block_count": len(code_blocks),
            "scores": score,
        })
        if (i + 1) % 5 == 0:
            print(f"  Adapter: {i + 1}/{len(prompts)} done")
    print(f"  Adapter model: {len(adapter_results)} generations complete")

    # --- Compare ---
    base_avg = sum(r["scores"]["total"] for r in base_results) / len(base_results)
    adapter_avg = sum(r["scores"]["total"] for r in adapter_results) / len(adapter_results)
    improvement = ((adapter_avg - base_avg) / max(base_avg, 0.01)) * 100
    passed = adapter_avg >= base_avg

    # Per-category averages for diagnostics
    categories = ["has_code_block", "balanced_braces", "no_stubs",
                   "no_bare_unwrap", "has_explanation", "modifies_code"]
    base_category_avgs = {}
    adapter_category_avgs = {}
    for cat in categories:
        base_category_avgs[cat] = sum(
            r["scores"][cat] for r in base_results
        ) / len(base_results)
        adapter_category_avgs[cat] = sum(
            r["scores"][cat] for r in adapter_results
        ) / len(adapter_results)

    print(f"\n{'=' * 60}")
    print(f"  Generation Evaluation (v3 compile-check proxy)")
    print(f"  {'─' * 56}")
    print(f"  Base avg score:    {base_avg:.2f} / 10")
    print(f"  Adapter avg score: {adapter_avg:.2f} / 10")
    print(f"  Improvement:       {improvement:+.1f}%")
    print(f"  Verdict:           {'PASS' if passed else 'FAIL'}")
    print(f"  {'─' * 56}")
    print(f"  Category breakdown (base / adapter):")
    for cat in categories:
        print(f"    {cat:20s}  {base_category_avgs[cat]:.2f}  /  {adapter_category_avgs[cat]:.2f}")
    print(f"{'=' * 60}")

    model_cache.commit()

    return {
        "eval_mode": "generation",
        "base_avg_score": base_avg,
        "adapter_avg_score": adapter_avg,
        "improvement_pct": improvement,
        "eval_samples": len(prompts),
        "passed": passed,
        "base_category_avgs": base_category_avgs,
        "adapter_category_avgs": adapter_category_avgs,
        "base_results": base_results,
        "adapter_results": adapter_results,
    }


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

    # Auto-scale for large models to avoid OOM on H100 80GB.
    # Only override values that weren't explicitly changed from defaults by the caller.
    if _is_large_model(base):
        for key, override_val in LARGE_MODEL_OVERRIDES.items():
            if config[key] == DEFAULTS[key]:
                print(f"  Large model detected — overriding {key}: {config[key]} → {override_val}")
                config[key] = override_val
        batch_size = config["batch_size"]
        grad_accum = config["grad_accum"]
        max_seq_length = config["max_seq_length"]

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
