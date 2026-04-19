from __future__ import annotations

import enum
import hashlib
import pathlib
from dataclasses import dataclass, field
from typing import Any


class ErrorCategory(str, enum.Enum):
    COMPILE = "compile"
    TEST = "test"
    CLIPPY = "clippy"
    FORMAT = "format"
    APPLY = "apply"
    PARSE = "parse"
    CONTEXT = "context"
    UNKNOWN = "unknown"
    NONE = "none"


class OperatorStatus(str, enum.Enum):
    OK = "ok"
    FAIL = "fail"
    SKIP = "skip"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class Trace:
    """Observational input to Reflect. Carries the most recent architect
    response and verifier result so Reflect can parse hypotheses."""
    iteration: int
    last_response: str | None
    last_eval: EvalResult | None


@dataclass(frozen=True)
class Hypothesis:
    """One candidate failure mode Reflect identifies in the trace.

    Example: ("compile", "src/foo.rs", "E0308 expected i32 found u32")."""
    category: ErrorCategory
    location: str | None
    summary: str
    raw_excerpt: str


@dataclass(frozen=True)
class Modification:
    """A proposed whole-file delta. Mirrors the existing
    architect.extract_whole_files return shape so Task 1.2 can plug in
    without reshaping."""
    files: dict[str, str]
    source_response: str
    served_model: str | None = None   # Phase 4: which TZ variant produced this

    def digest(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.files):
            h.update(p.encode())
            h.update(b"\x00")
            h.update(self.files[p].encode())
            h.update(b"\x01")
        return h.hexdigest()[:12]


@dataclass(frozen=True)
class GateResult:
    """One verifier gate (cargo check, cargo test, clippy, fmt)."""
    name: str
    passed: bool
    stderr_tail: str = ""
    category: ErrorCategory = ErrorCategory.UNKNOWN


@dataclass(frozen=True)
class EvalResult:
    """Structured verifier output. Constructed from the existing
    verifier_fn dict shape: {all_green: bool, gates: {name: {...}}}."""
    all_green: bool
    gates: tuple[GateResult, ...]
    duration_s: float = 0.0

    @property
    def first_failing(self) -> GateResult | None:
        return next((g for g in self.gates if not g.passed), None)

    @property
    def failing_categories(self) -> tuple[ErrorCategory, ...]:
        return tuple(g.category for g in self.gates if not g.passed)

    @classmethod
    def from_verifier_dict(cls, d: dict[str, Any], duration_s: float = 0.0) -> EvalResult:
        """Adapt the existing verifier_fn output shape."""
        raw_gates = d.get("gates") or {}
        gates: list[GateResult] = []
        for name, g in raw_gates.items():
            cat = _classify_gate(name, g.get("stderr_tail") or "")
            gates.append(GateResult(
                name=name,
                passed=bool(g.get("passed")),
                stderr_tail=(g.get("stderr_tail") or "")[-3000:],
                category=cat,
            ))
        return cls(
            all_green=bool(d.get("all_green")),
            gates=tuple(gates),
            duration_s=duration_s,
        )


def _classify_gate(name: str, stderr_tail: str) -> ErrorCategory:
    n = name.lower()
    if "fmt" in n or "format" in n:
        return ErrorCategory.FORMAT
    if "clippy" in n:
        return ErrorCategory.CLIPPY
    if "test" in n:
        return ErrorCategory.TEST
    if "check" in n or "build" in n or "compile" in n:
        return ErrorCategory.COMPILE
    return ErrorCategory.UNKNOWN


def text_digest(text: str | None) -> str:
    if text is None:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()[:12]
