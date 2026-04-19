"""Tests for the Phase 3 tool-generator agent."""
from __future__ import annotations

import pathlib

import pytest

from rspl import ResourceRegistry, ResourceType
from sepl.tool_synth import (
    ToolSpec,
    ToolSynthesizer,
    ToolValidationError,
    extract_python,
)


# ──────────────────────────────────────────────────────────────────────────────
# extract_python
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_python_reads_fenced_block():
    resp = "intro text\n```python\ndef f(): ...\n```\nouttro"
    assert "def f()" in extract_python(resp)


def test_extract_python_tolerates_bare_fence():
    resp = "```\ndef f(): ...\n```"
    assert "def f()" in extract_python(resp)


def test_extract_python_raises_when_missing():
    with pytest.raises(ToolValidationError, match="no.*code block"):
        extract_python("no code here")


# ──────────────────────────────────────────────────────────────────────────────
# install_from_response: happy path + registration
# ──────────────────────────────────────────────────────────────────────────────

_GOOD_FN = '''```python
def count_compile_errors(stderr_tail: str) -> int:
    """Count `error[EXXXX]:` occurrences in a cargo-check stderr tail."""
    return stderr_tail.count("error[")
```'''


def test_synthesize_happy_path_registers_tool():
    reg = ResourceRegistry()
    synth = ToolSynthesizer(reg)
    spec = ToolSpec(
        name="count_compile_errors",
        docstring="Count cargo-check compile errors.",
        signature_hint="(stderr_tail: str) -> int",
        gap_context="error[E0308]: mismatched types",
    )
    fn = synth.install_from_response(_GOOD_FN, spec)

    # Tool is callable
    assert fn("error[E0308]: x\nerror[E0412]: y") == 2
    # Tool is in the registry
    rv = reg.get("tool:count_compile_errors")
    assert rv is not None
    assert rv.resource.resource_type == ResourceType.TOOL
    assert rv.resource.metadata["synthesized"] is True
    # Registry can execute it
    assert reg.run("tool:count_compile_errors", "error[E0001]: z") == 1


# ──────────────────────────────────────────────────────────────────────────────
# Validation rejections
# ──────────────────────────────────────────────────────────────────────────────

def _spec(name="f"):
    return ToolSpec(name=name, docstring="", signature_hint="()", gap_context="")


def test_rejects_subprocess_import():
    bad = '```python\nimport subprocess\ndef f() -> None:\n    """x"""\n    pass\n```'
    with pytest.raises(ToolValidationError, match="disallowed import.*subprocess"):
        ToolSynthesizer(ResourceRegistry()).install_from_response(bad, _spec())


def test_rejects_os_system():
    bad = ('```python\nimport os\n'
           'def f() -> None:\n    """x"""\n    os.system("ls")\n```')
    with pytest.raises(ToolValidationError, match="disallowed call.*os.system"):
        ToolSynthesizer(ResourceRegistry()).install_from_response(bad, _spec())


def test_rejects_eval_reference():
    bad = ('```python\ndef f() -> None:\n    """x"""\n'
           '    return eval("1+1")\n```')
    with pytest.raises(ToolValidationError, match="disallowed name.*eval"):
        ToolSynthesizer(ResourceRegistry()).install_from_response(bad, _spec())


def test_rejects_socket_from_import():
    bad = ('```python\nfrom socket import socket\n'
           'def f() -> None:\n    """x"""\n    pass\n```')
    with pytest.raises(ToolValidationError, match="disallowed import"):
        ToolSynthesizer(ResourceRegistry()).install_from_response(bad, _spec())


def test_rejects_wrong_function_name():
    bad = '```python\ndef other_fn() -> None:\n    """x"""\n    pass\n```'
    with pytest.raises(ToolValidationError, match="expected.*function named"):
        ToolSynthesizer(ResourceRegistry()).install_from_response(bad, _spec("f"))


def test_rejects_missing_docstring():
    bad = '```python\ndef f() -> None:\n    pass\n```'
    with pytest.raises(ToolValidationError, match="missing docstring"):
        ToolSynthesizer(ResourceRegistry()).install_from_response(bad, _spec("f"))


def test_rejects_syntax_error():
    bad = '```python\ndef f(\n    """x"""\n```'
    with pytest.raises(ToolValidationError, match="syntax error"):
        ToolSynthesizer(ResourceRegistry()).install_from_response(bad, _spec("f"))


# ──────────────────────────────────────────────────────────────────────────────
# model_call integration
# ──────────────────────────────────────────────────────────────────────────────

def test_synthesize_calls_model_and_installs():
    reg = ResourceRegistry()
    captured_prompts = []

    def stub_model(messages):
        captured_prompts.append(messages)
        return _GOOD_FN

    synth = ToolSynthesizer(reg, model_call=stub_model)
    fn = synth.synthesize(ToolSpec(
        name="count_compile_errors",
        docstring="Count cargo-check errors.",
        signature_hint="(stderr_tail: str) -> int",
    ))
    assert fn("error[E0001]: x") == 1
    # Prompt included the signature hint
    sys_msg = captured_prompts[0][0]["content"]
    user_msg = captured_prompts[0][1]["content"]
    assert "tool-generator agent" in sys_msg
    assert "count_compile_errors" in user_msg


def test_synthesize_without_model_call_raises():
    with pytest.raises(RuntimeError, match="model_call not configured"):
        ToolSynthesizer(ResourceRegistry()).synthesize(_spec("f"))
