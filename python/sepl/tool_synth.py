"""Tool-generator agent (AGP Phase 3).

When Reflect identifies a capability gap (e.g., the verifier emits an
error pattern we have no helper for), ToolSynthesizer asks a model to
emit a Python function, validates it via AST walk (rejects dangerous
imports), writes it to /tmp/sepl-tools/, dynamic-imports it, and
registers the resulting callable as a RSPL Tool resource.

This is the swarm analog of Autogenesis §5.2's tool-generator agent —
the pattern that drove the paper's +33.3% gain on GAIA Level 3.

Safety model (intentionally conservative for MVP):
  - Tools live OUTSIDE the repo (/tmp/sepl-tools/) so a buggy tool can
    never land in git.
  - AST walk rejects imports of: subprocess, os.system/os.exec, socket,
    requests, urllib, http, ftplib, telnetlib, ctypes, __import__, eval, exec.
  - Tools must be pure-Python functions with explicit type hints on all
    parameters and a docstring (enforced at registration time).
  - Execution runs in the calling process (no subprocess sandbox yet —
    that's a follow-up).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable

from rspl import Resource, ResourceRegistry, ResourceType


TOOL_SOURCE_DIR = pathlib.Path("/tmp/sepl-tools")

_DISALLOWED_IMPORTS = frozenset({
    "subprocess", "socket", "requests", "urllib", "http", "ftplib",
    "telnetlib", "ctypes", "paramiko", "asyncssh", "httpx",
    "aiohttp", "pickle", "shelve", "marshal",
})
_DISALLOWED_OS_ATTRS = frozenset({
    "system", "exec", "execv", "execvp", "execve", "spawn",
    "spawnv", "spawnvp", "spawnve", "popen", "remove", "removedirs",
    "rmdir", "unlink",
})
_DISALLOWED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open",
    "input", "breakpoint", "globals", "locals", "vars",
})

_CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(?P<code>.*?)^```",
    re.MULTILINE | re.DOTALL,
)


class ToolValidationError(RuntimeError):
    """Raised when a synthesized tool fails safety/shape validation."""


@dataclass
class ToolSpec:
    """What the synthesizer asks the model to produce."""
    name: str                         # snake_case python identifier
    docstring: str                    # human description of intended behavior
    signature_hint: str               # e.g. "(stderr_tail: str) -> list[str]"
    gap_context: str = ""             # error excerpt or task the tool should address


# ──────────────────────────────────────────────────────────────────────────────

def _validate_source(src: str, expected_name: str) -> ast.Module:
    """AST-walk a synthesized source to reject dangerous patterns.
    Returns the parsed module if validation passes."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise ToolValidationError(f"syntax error: {e}")

    # Must define exactly one top-level function with expected_name
    fn_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fn_defs) != 1 or fn_defs[0].name != expected_name:
        raise ToolValidationError(
            f"expected exactly one top-level function named {expected_name!r}, "
            f"got {[n.name for n in fn_defs]}"
        )
    fn = fn_defs[0]
    if not (isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        raise ToolValidationError(f"{expected_name}: missing docstring")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DISALLOWED_IMPORTS:
                    raise ToolValidationError(
                        f"disallowed import: {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in _DISALLOWED_IMPORTS:
                raise ToolValidationError(
                    f"disallowed import: from {node.module!r}")
            if mod == "os":
                for alias in node.names:
                    if alias.name in _DISALLOWED_OS_ATTRS:
                        raise ToolValidationError(
                            f"disallowed os.{alias.name}")
        elif isinstance(node, ast.Name):
            if node.id in _DISALLOWED_NAMES:
                raise ToolValidationError(
                    f"disallowed name reference: {node.id!r}")
        elif isinstance(node, ast.Attribute):
            # os.system, os.popen, etc.
            if (isinstance(node.value, ast.Name)
                    and node.value.id == "os"
                    and node.attr in _DISALLOWED_OS_ATTRS):
                raise ToolValidationError(
                    f"disallowed call: os.{node.attr}")
    return tree


def _write_and_import(src: str, name: str) -> Callable:
    """Write src to TOOL_SOURCE_DIR/<name>.py, import it, return the
    callable. The file name includes a digest so repeated synthesis
    doesn't cache an older version through importlib."""
    TOOL_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(src.encode()).hexdigest()[:8]
    module_name = f"sepl_tool_{name}_{digest}"
    path = TOOL_SOURCE_DIR / f"{module_name}.py"
    path.write_text(src)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ToolValidationError("importlib could not build spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise ToolValidationError(f"import failed: {type(e).__name__}: {e}")
    fn = getattr(module, name, None)
    if not callable(fn):
        raise ToolValidationError(f"{name} is not callable after import")
    return fn


def extract_python(response: str) -> str:
    """Pull the first ```python ...``` fenced block from a model response.
    Accepts bare ``` too (some models forget the language tag)."""
    m = _CODE_BLOCK_RE.search(response)
    if not m:
        raise ToolValidationError("no ``` code block found in model response")
    return m.group("code")


class ToolSynthesizer:
    """Ask a model to synthesize a Python helper, validate, install,
    and register it."""

    def __init__(
        self,
        registry: ResourceRegistry,
        *,
        model_call: Callable[[list[dict]], str] | None = None,
    ):
        self.registry = registry
        # `model_call(messages) -> response_text`. Tests pass a stub;
        # production will wire to architect.call_architect.
        self.model_call = model_call

    def _build_prompt(self, spec: ToolSpec) -> list[dict]:
        sys_msg = (
            "You are a tool-generator agent. Produce a SINGLE Python "
            "function that addresses the described capability gap.\n\n"
            "Rules:\n"
            "  - Emit exactly one ```python ... ``` block.\n"
            "  - Pure Python. No subprocess/socket/requests/urllib/http.\n"
            "  - Type-hint every parameter and the return value.\n"
            "  - First line inside the function must be a docstring.\n"
            "  - No top-level code outside the function (no `if __name__ == \"__main__\"`).\n"
            "  - Function name MUST be exactly the one specified."
        )
        user_msg = (
            f"Function name: {spec.name}\n"
            f"Signature hint: {spec.signature_hint}\n"
            f"Purpose: {spec.docstring}\n\n"
            f"Context / error excerpt this tool must handle:\n"
            f"```\n{spec.gap_context[:2000]}\n```\n\n"
            "Emit the ```python``` block now."
        )
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}]

    def synthesize(self, spec: ToolSpec) -> Callable:
        """Call the model, extract + validate the source, dynamic-import,
        register in RSPL as a Tool resource. Returns the callable."""
        if self.model_call is None:
            raise RuntimeError("ToolSynthesizer.model_call not configured")
        response = self.model_call(self._build_prompt(spec))
        return self.install_from_response(response, spec)

    def install_from_response(self, response: str, spec: ToolSpec) -> Callable:
        """Pre-validation path used by tests + when the model response
        is already in hand (e.g. cached)."""
        src = extract_python(response)
        _validate_source(src, spec.name)
        fn = _write_and_import(src, spec.name)

        resource = Resource(
            name=spec.name,
            resource_type=ResourceType.TOOL,
            description=spec.docstring,
            mapping=fn,
            trainable=True,
            metadata={
                "signature_hint": spec.signature_hint,
                "synthesized": True,
                "source_len": len(src),
            },
        )
        self.registry.register(
            resource,
            implementation=f"sepl_tools:{spec.name}",
            exports={"source_preview": src[:300]},
        )
        return fn
