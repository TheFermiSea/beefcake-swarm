#!/usr/bin/env python3
"""Normalize TensorZero trajectory format to standard chat JSONL for SFT training.

TZ stores multi-turn agent trajectories with nested content arrays and tool_call
objects. SFT trainers (TRL, Axolotl, Unsloth) expect simple chat format:
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]}

This script converts:
- Nested content arrays → flat strings
- Tool calls → text descriptions (so the model learns the tool-use pattern)
- Tool results → text (so the model sees what tools return)
- System prompts → proper system message

Usage:
    python3 scripts/normalize-trajectories.py input.jsonl output.jsonl
    python3 scripts/normalize-trajectories.py input.jsonl output.jsonl --max-tokens 4096
"""

import json
import sys
import argparse


def flatten_content(content):
    """Convert nested content array to flat string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "tool_call":
                    name = item.get("name", "unknown_tool")
                    args = item.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args_parsed = json.loads(args)
                            args_str = json.dumps(args_parsed, indent=2)
                        except json.JSONDecodeError:
                            args_str = args
                    else:
                        args_str = json.dumps(args, indent=2)
                    parts.append(f"<tool_call>\n{name}({args_str})\n</tool_call>")
                elif item.get("type") == "tool_result":
                    result = item.get("content", item.get("text", ""))
                    if isinstance(result, list):
                        result = "\n".join(
                            r.get("text", str(r)) if isinstance(r, dict) else str(r)
                            for r in result
                        )
                    # Truncate very long tool results (file contents, etc.)
                    if len(str(result)) > 2000:
                        result = str(result)[:2000] + "\n... [truncated]"
                    parts.append(f"<tool_result>\n{result}\n</tool_result>")
                elif item.get("type") == "tool_use":
                    # Anthropic format
                    name = item.get("name", "unknown_tool")
                    inp = item.get("input", {})
                    parts.append(f"<tool_call>\n{name}({json.dumps(inp)})\n</tool_call>")
                else:
                    # Unknown type — include as text
                    parts.append(str(item))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def normalize_sample(raw):
    """Convert a TZ trajectory sample to standard chat format."""
    messages = raw.get("messages", [])
    metadata = raw.get("metadata", {})

    if not messages:
        return None

    first = messages[0]

    # Detect format
    if isinstance(first, dict) and "system" in first:
        # TZ format: {"messages": [{"system": "...", "messages": [...]}]}
        system_prompt = first.get("system", "")
        inner_messages = first.get("messages", [])
    elif isinstance(first, dict) and first.get("role") in ("system", "user", "assistant"):
        # Already standard chat format
        return raw
    else:
        return None

    # Build normalized messages
    normalized = []

    # System message (truncate if very long — training doesn't need the full rubric)
    if system_prompt:
        # Keep only the core instruction, not the full rubric
        if len(system_prompt) > 1500:
            # Extract first paragraph + key sections
            lines = system_prompt.split("\n")
            kept = []
            total = 0
            for line in lines:
                if total + len(line) > 1500:
                    kept.append("...")
                    break
                kept.append(line)
                total += len(line)
            system_prompt = "\n".join(kept)
        normalized.append({"role": "system", "content": system_prompt})

    # Convert inner messages
    for msg in inner_messages:
        role = msg.get("role", "user")
        content = flatten_content(msg.get("content", ""))

        if not content.strip():
            continue

        normalized.append({"role": role, "content": content})

    if len(normalized) < 2:
        return None

    return {"messages": normalized, "metadata": metadata}


def estimate_tokens(messages):
    """Rough token estimate: ~4 chars per token."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4


def main():
    parser = argparse.ArgumentParser(description="Normalize TZ trajectories for SFT training")
    parser.add_argument("input", help="Input JSONL (TZ format)")
    parser.add_argument("output", help="Output JSONL (standard chat format)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Maximum token estimate per sample (default: 4096)")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    args = parser.parse_args()

    kept = 0
    dropped_empty = 0
    dropped_long = 0
    dropped_format = 0
    total_messages = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            raw = json.loads(line)
            normalized = normalize_sample(raw)

            if normalized is None:
                dropped_format += 1
                continue

            messages = normalized["messages"]
            if len(messages) < 2:
                dropped_empty += 1
                continue

            tokens = estimate_tokens(messages)
            if tokens > args.max_tokens:
                dropped_long += 1
                continue

            fout.write(json.dumps(normalized) + "\n")
            kept += 1
            total_messages += len(messages)

    total = kept + dropped_empty + dropped_long + dropped_format
    print(f"Normalized: {kept}/{total} kept")
    print(f"  Dropped: {dropped_format} format, {dropped_empty} empty, {dropped_long} too long")
    if kept > 0:
        print(f"  Avg messages per sample: {total_messages / kept:.1f}")


if __name__ == "__main__":
    main()
