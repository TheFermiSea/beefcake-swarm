#!/usr/bin/env python3
"""Chat-to-Completion proxy for Qwen3.5-397B on llama.cpp.

Translates OpenAI /v1/chat/completions requests into /v1/completions requests
using pure text continuation prompts (no special tokens).

Background: Qwen3.5-397B at UD-Q4_K_XL quantization produces corrupted logits
after special token embeddings (<|im_start|>, <|im_end|>, <think>), causing
immediate EOS or degenerate output for any instruction-following prompt format.
The model generates coherent output with pure text continuation prompts.

The proxy:
1. Converts chat messages into document-continuation format (no special tokens)
2. Calls /v1/completions with tuned sampling parameters
3. Detects and truncates degenerate repetition in responses
4. Parses Qwen3 XML tool calls if present
5. Returns an OpenAI-compatible chat completion response

Remove this proxy when llama.cpp fixes the qwen35moe special token handling
or when switching to a working GGUF quantization (e.g., lmstudio Q4_K_M).
Tracked: llama.cpp #19690, #19858; beefcake-noqp.
"""

import json
import os
import re
import time
import uuid
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="qwen35-chat-proxy")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8080")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8180"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "900"))

# Sampling defaults tuned for qwen35moe text continuation
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0
DEFAULT_REPEAT_PENALTY = 1.2
DEFAULT_PRESENCE_PENALTY = 0.6

client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=REQUEST_TIMEOUT)


def _format_tool_definitions(tools: list[dict]) -> str:
    """Format tool definitions as plain text for the continuation prompt."""
    lines = ["Available functions:"]
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"\n- {name}: {desc}")
        if params.get("properties"):
            props = params["properties"]
            required = set(params.get("required", []))
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                lines.append(f"  - {pname}: {ptype}{req} — {pdesc}")
    lines.append(
        "\nTo call a function, output: <tool_call>{\"name\": \"function_name\", "
        "\"arguments\": {...}}</tool_call>"
    )
    return "\n".join(lines)


def format_as_continuation(messages: list[dict], tools: list[dict] | None = None) -> str:
    """Convert chat messages into a plain-text continuation prompt.

    Avoids ALL special tokens (<|im_start|>, <|im_end|>, <think>). The model's
    forward pass produces corrupted logits after special token embeddings in the
    UD-Q4_K_XL quantization, so we bypass the chat template entirely and use
    pure text continuation which produces sharp, coherent logits.

    The format mimics a code review / documentation document that the model
    continues naturally.
    """
    system_text = ""
    conversation: list[tuple[str, str]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "system":
            system_text = content
        elif role == "user":
            conversation.append(("user", content))
        elif role == "assistant":
            conversation.append(("assistant", content))
        elif role == "tool":
            name = msg.get("name", "function")
            conversation.append(("tool", f"[{name} returned: {content}]"))

    parts: list[str] = []

    # System context as document header
    if system_text:
        parts.append(system_text)

    # Tool definitions
    if tools:
        parts.append(_format_tool_definitions(tools))

    # Multi-turn history as document context
    if len(conversation) > 1:
        for role, content in conversation[:-1]:
            if role == "user":
                parts.append(f"Task: {content}")
            elif role == "assistant":
                parts.append(content)
            elif role == "tool":
                parts.append(content)
        parts.append("")  # blank line separator

    # Current request
    if conversation:
        last_role, last_content = conversation[-1]
        if last_role == "user":
            parts.append(f"Task: {last_content}")
        elif last_role == "tool":
            parts.append(last_content)

    # Continuation trigger — model completes naturally after this
    parts.append("\nComplete response:\n")

    return "\n\n".join(parts)


def _detect_degeneration(text: str) -> str:
    """Truncate response at the first sign of degeneration (repetitive patterns)."""
    # Detect repeating 3+ word sequences appearing 3+ times
    words = text.split()
    if len(words) < 12:
        return text

    # Sliding window: find the longest prefix before a 3-gram repeats 3+ times
    for window_size in (3, 4, 5):
        seen: dict[str, int] = {}
        for i in range(len(words) - window_size + 1):
            gram = " ".join(words[i : i + window_size])
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= 3:
                # Find where the first repeat started and cut there
                first_occurrence = text.find(gram)
                second_occurrence = text.find(gram, first_occurrence + len(gram))
                if second_occurrence > 0:
                    return text[:second_occurrence].rstrip()
    return text


def _clean_response(text: str) -> str:
    """Clean up the model's response: strip special tokens, degenerate tails."""
    # Remove any leaked special tokens
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove trailing special chars that indicate degeneration
    text = re.sub(r"[`)\]}>]{3,}\s*$", "", text)
    # Detect and truncate degenerate repetition
    text = _detect_degeneration(text)
    return text.strip()


def _parse_tool_calls(text: str) -> tuple[str, list[dict] | None]:
    """Extract <tool_call>...</tool_call> blocks from response text.
    Returns (cleaned_text, tool_calls_or_none)."""
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return text, None

    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": parsed.get("name", ""),
                    "arguments": json.dumps(parsed.get("arguments", {})),
                },
            })
        except json.JSONDecodeError:
            continue

    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return cleaned, tool_calls if tool_calls else None


def _build_chat_response(
    text: str, model: str, usage: dict | None = None, tool_calls: list[dict] | None = None
) -> dict:
    """Wrap completion text into an OpenAI chat completion response."""
    message: dict = {"role": "assistant"}
    if tool_calls:
        message["content"] = text if text else None
        message["tool_calls"] = tool_calls
    else:
        message["content"] = text

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Passthrough routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        resp = await client.get("/health")
        backend = resp.json() if resp.status_code == 200 else {"status": "error"}
    except Exception:
        backend = {"status": "unreachable"}
    return {"status": "ok", "proxy": True, "backend": backend}


@app.get("/v1/models")
async def models():
    resp = await client.get("/v1/models")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/props")
async def props():
    resp = await client.get("/props")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# Also pass through /v1/completions directly for clients that want raw access
@app.post("/v1/completions")
async def completions_passthrough(request: Request):
    body = await request.json()
    resp = await client.post("/v1/completions", json=body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Chat completions → Completions translation (with ignore_eos workaround)
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    tools = body.get("tools")
    stream = body.get("stream", False)
    model = body.get("model", "Qwen3.5-397B-A17B")

    # Convert to plain-text continuation format (no special tokens)
    prompt = format_as_continuation(messages, tools)

    # Build completion request with ignore_eos workaround
    completion_body = {
        "prompt": prompt,
        "model": model,
        "temperature": body.get("temperature", DEFAULT_TEMPERATURE),
        "top_p": body.get("top_p", DEFAULT_TOP_P),
        "top_k": body.get("top_k", DEFAULT_TOP_K),
        "min_p": body.get("min_p", DEFAULT_MIN_P),
        "repeat_penalty": body.get("repeat_penalty", DEFAULT_REPEAT_PENALTY),
        "presence_penalty": body.get("presence_penalty", DEFAULT_PRESENCE_PENALTY),
        "stream": stream,
        # Pure text continuation mode — no special tokens in the prompt means
        # the model generates with sharp logits (no EOS bug). Stop on common
        # document boundaries.
        "stop": ["Task:", "\n\nTask:"],
    }

    # max_tokens
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    if max_tokens:
        completion_body["n_predict"] = max_tokens
    else:
        completion_body["n_predict"] = 2048  # reasonable default

    # Additional stop sequences from the request
    if body.get("stop"):
        existing_stops = completion_body["stop"]
        extra = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
        completion_body["stop"] = existing_stops + extra

    if stream:
        return StreamingResponse(
            _stream_completions(completion_body, model, bool(tools)),
            media_type="text/event-stream",
        )

    # Non-streaming
    resp = await client.post("/v1/completions", json=completion_body)
    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    result = resp.json()
    text = result.get("choices", [{}])[0].get("text", "")

    # Clean up the response
    text = _clean_response(text)

    # Check for tool calls
    tool_calls = None
    if tools:
        text, tool_calls = _parse_tool_calls(text)

    usage = result.get("usage")
    return JSONResponse(content=_build_chat_response(text, model, usage, tool_calls))


async def _stream_completions(
    completion_body: dict, model: str, has_tools: bool
) -> AsyncIterator[str]:
    """Stream completion responses, translating to chat completion SSE format."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    async with client.stream("POST", "/v1/completions", json=completion_body) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue

            data = line[6:]
            if data.strip() == "[DONE]":
                yield "data: [DONE]\n\n"
                break

            try:
                chunk = json.loads(data)
                text = chunk.get("choices", [{}])[0].get("text", "")
                # Skip leaked special tokens in stream
                if text in ("<|im_end|>", "<|im_start|>"):
                    continue
                chat_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chat_chunk)}\n\n"
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
