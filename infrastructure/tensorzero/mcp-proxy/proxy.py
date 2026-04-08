#!/usr/bin/env python3
"""
TensorZero MCP schema-patching proxy.

Sits between MCP clients (ForgeCode, Claude Code, etc.) and the TensorZero MCP
server. Intercepts `tools/list` responses and recursively adds missing `"type"`
keys to any property schema that lacks one — the exact validation OpenAI enforces
at function-registration time.

Runs on PROXY_PORT (default 3001), proxies to UPSTREAM_MCP_URL (default
http://localhost:3000/mcp). Robust against TensorZero upstream releases.
"""

import asyncio
import json
import logging
import os

from aiohttp import ClientSession, ClientTimeout, web

UPSTREAM_URL = os.environ.get("UPSTREAM_MCP_URL", "http://localhost:3000/mcp")
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "3001"))
LISTEN_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mcp-proxy")

# SSE hop-by-hop headers to strip when forwarding
_HOP_BY_HOP = frozenset(
    {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}
)


def ensure_type(schema: object) -> object:
    """Recursively ensure every property schema has a 'type' key.

    Any dict that is a property schema (has 'description' or 'properties' but
    no 'type' and no '$ref') gets ``"type": "object"`` injected.  All nested
    structures (properties, items, additionalProperties, anyOf/oneOf/allOf) are
    visited recursively.
    """
    if not isinstance(schema, dict):
        return schema

    # Recurse into standard sub-schema locations before we touch this node
    if "properties" in schema and isinstance(schema["properties"], dict):
        schema["properties"] = {k: ensure_type(v) for k, v in schema["properties"].items()}

    if "items" in schema:
        items = schema["items"]
        schema["items"] = (
            [ensure_type(i) for i in items]
            if isinstance(items, list)
            else ensure_type(items)
        )

    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        schema["additionalProperties"] = ensure_type(schema["additionalProperties"])

    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in schema and isinstance(schema[combinator], list):
            schema[combinator] = [ensure_type(s) for s in schema[combinator]]

    # Inject 'type' only if missing and this isn't a $ref passthrough
    if "type" not in schema and "$ref" not in schema:
        schema["type"] = "object"

    return schema


def patch_tools_list(data: dict) -> dict:
    """Patch inputSchema for every tool in a tools/list result."""
    result = data.get("result")
    if not isinstance(result, dict):
        return data

    patched = 0
    for tool in result.get("tools", []):
        if "inputSchema" in tool and isinstance(tool["inputSchema"], dict):
            tool["inputSchema"] = ensure_type(tool["inputSchema"])
            patched += 1

    if patched:
        log.info("Patched inputSchema for %d tool(s) in tools/list response", patched)

    return data


def _is_tools_list(body: bytes) -> bool:
    try:
        return json.loads(body).get("method") == "tools/list"
    except Exception:
        return False


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    patching = _is_tools_list(body)

    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }

    timeout = ClientTimeout(total=300)
    async with ClientSession(timeout=timeout) as session:
        async with session.request(
            request.method,
            UPSTREAM_URL,
            data=body,
            headers=fwd_headers,
            allow_redirects=False,
        ) as upstream:
            resp = web.StreamResponse(status=upstream.status)
            for k, v in upstream.headers.items():
                if k.lower() not in _HOP_BY_HOP:
                    resp.headers[k] = v
            await resp.prepare(request)

            if not patching:
                # Fast path: pass bytes through unchanged
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
            else:
                # Intercept SSE stream and patch tools/list JSON payload
                buf = b""
                async for chunk in upstream.content.iter_any():
                    buf += chunk
                    # Emit all complete lines (\n-terminated)
                    while b"\n" in buf:
                        nl = buf.index(b"\n")
                        line_bytes = buf[: nl + 1]
                        buf = buf[nl + 1 :]

                        line = line_bytes.decode("utf-8", errors="replace")
                        if line.startswith("data: "):
                            payload = line[6:].rstrip("\r\n")
                            if payload:
                                try:
                                    msg = json.loads(payload)
                                    msg = patch_tools_list(msg)
                                    line_bytes = (
                                        "data: " + json.dumps(msg) + "\n"
                                    ).encode("utf-8")
                                except json.JSONDecodeError:
                                    pass
                        await resp.write(line_bytes)

                if buf:
                    await resp.write(buf)

    return resp


app = web.Application()
app.router.add_route("*", "/mcp", proxy_handler)
app.router.add_route("*", "/mcp/{tail:.*}", proxy_handler)


@app.on_startup.append
async def _startup(app: web.Application) -> None:  # noqa: ARG001
    log.info("MCP schema-patching proxy listening on %s:%d", LISTEN_HOST, LISTEN_PORT)
    log.info("Upstream TensorZero MCP: %s", UPSTREAM_URL)


if __name__ == "__main__":
    web.run_app(app, host=LISTEN_HOST, port=LISTEN_PORT, access_log=log)
