"""Concrete SEPL operator implementations.

Each operator is a small class with a typed `.run()` method. Signatures
vary because the paper's operators have different arities (Reflect: trace
-> hyps; Select: hyps -> chosen; Improve: hyp+state -> modification;
Evaluate: modification -> eval; Commit: state -> status). The `Operator`
Protocol at the bottom is a loose marker (just `.name`), not a strict
interface.

Operators:
  - return (output, LineageRecord) pairs — the driver appends the record
    to the per-issue lineage JSONL.
  - catch their own expected failures and return a FAIL/SKIP record
    rather than propagating exceptions, so the driver always has an audit
    row per attempted step.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Protocol

from .lineage import LineageRecord, now_record
from .state import SEPLState
from .types import (
    ErrorCategory,
    EvalResult,
    GateResult,
    Hypothesis,
    Modification,
    OperatorStatus,
    text_digest,
)


# APPLY first (nothing downstream works until files land), then CONTEXT
# (model needs more info to proceed), then the standard compile-to-lint
# cascade.
_SELECT_PRIORITY: tuple[ErrorCategory, ...] = (
    ErrorCategory.APPLY,
    ErrorCategory.CONTEXT,
    ErrorCategory.PARSE,
    ErrorCategory.COMPILE,
    ErrorCategory.TEST,
    ErrorCategory.CLIPPY,
    ErrorCategory.FORMAT,
    ErrorCategory.UNKNOWN,
)

_COMPILE_ERROR_RE = re.compile(
    r"(error(?:\[[A-Z0-9]+\])?:\s*[^\n]+)(?:\n\s+-->\s+([^\n]+))?",
    re.MULTILINE,
)
_TEST_FAIL_RE = re.compile(r"test\s+(\S+)\s+\.\.\.\s+FAILED", re.MULTILINE)
_CLIPPY_WARN_RE = re.compile(
    r"((?:warning|error)(?:\[[A-Z0-9]+\])?:\s*[^\n]+)",
    re.MULTILINE,
)


def _parse_gate(gate: GateResult) -> list[Hypothesis]:
    """Parse one failing GateResult into up to 3 Hypotheses."""
    if gate.passed:
        return []
    tail = gate.stderr_tail or ""
    cat = gate.category

    if cat == ErrorCategory.COMPILE:
        matches = _COMPILE_ERROR_RE.findall(tail)
        if matches:
            return [
                Hypothesis(
                    category=ErrorCategory.COMPILE,
                    location=(loc or "").strip() or None,
                    summary=msg.strip()[:200],
                    raw_excerpt=tail[-1000:],
                )
                for msg, loc in matches[:3]
            ]
    elif cat == ErrorCategory.TEST:
        matches = _TEST_FAIL_RE.findall(tail)
        if matches:
            return [
                Hypothesis(
                    category=ErrorCategory.TEST,
                    location=None,
                    summary=f"test failed: {name}",
                    raw_excerpt=tail[-1000:],
                )
                for name in matches[:3]
            ]
    elif cat == ErrorCategory.CLIPPY:
        matches = _CLIPPY_WARN_RE.findall(tail)
        if matches:
            return [
                Hypothesis(
                    category=ErrorCategory.CLIPPY,
                    location=None,
                    summary=msg.strip()[:200],
                    raw_excerpt=tail[-1000:],
                )
                for msg in matches[:3]
            ]
    elif cat == ErrorCategory.APPLY:
        return [Hypothesis(
            category=ErrorCategory.APPLY,
            location=None,
            summary="failed to write files to worktree",
            raw_excerpt=tail,
        )]

    return [Hypothesis(
        category=cat,
        location=None,
        summary=f"{gate.name} failed",
        raw_excerpt=tail[-1000:],
    )]


class Reflect:
    """Parse (state.last_eval, state.last_response) into Hypotheses.

    First iteration (last_eval is None): returns empty tuple -- Select
    will short-circuit, Improve proceeds with no hypothesis."""
    name = "reflect"

    def run(self, state: SEPLState) -> tuple[tuple[Hypothesis, ...], LineageRecord]:
        t0 = time.time()
        ev = state.last_eval

        if ev is None:
            rec = now_record(
                self.name, state.iteration, OperatorStatus.OK,
                duration_s=time.time() - t0,
                metrics={"hyp_count": 0},
                note="first_iteration",
            )
            return (), rec

        if ev.all_green:
            rec = now_record(
                self.name, state.iteration, OperatorStatus.SKIP,
                duration_s=time.time() - t0,
                note="all_green",
            )
            return (), rec

        hyps: list[Hypothesis] = []
        for gate in ev.gates:
            hyps.extend(_parse_gate(gate))

        rec = now_record(
            self.name, state.iteration, OperatorStatus.OK,
            duration_s=time.time() - t0,
            metrics={
                "hyp_count": len(hyps),
                "categories": sorted({h.category.value for h in hyps}),
            },
        )
        return tuple(hyps), rec


class Select:
    """Pick the highest-priority hypothesis using _SELECT_PRIORITY.
    Break ties by input order (stable)."""
    name = "select"

    def run(
        self,
        hypotheses: tuple[Hypothesis, ...],
        state: SEPLState,
    ) -> tuple[Hypothesis | None, LineageRecord]:
        t0 = time.time()

        if not hypotheses:
            rec = now_record(
                self.name, state.iteration, OperatorStatus.SKIP,
                duration_s=time.time() - t0,
                note="no_hypotheses",
            )
            return None, rec

        order = {c: i for i, c in enumerate(_SELECT_PRIORITY)}
        ranked = sorted(
            enumerate(hypotheses),
            key=lambda ih: (order.get(ih[1].category, len(order)), ih[0]),
        )
        chosen = ranked[0][1]

        rec = now_record(
            self.name, state.iteration, OperatorStatus.OK,
            duration_s=time.time() - t0,
            metrics={
                "chosen_category": chosen.category.value,
                "pool_size": len(hypotheses),
            },
            note=chosen.summary[:100],
        )
        return chosen, rec


class Improve:
    """Call the architect model via TZ and parse the response into a
    Modification. Lazy-imports helpers from architect.py to avoid a
    hard module-load dependency (keeps sepl importable in isolation)."""
    name = "improve"

    def __init__(
        self,
        *,
        tz_url: str = "http://localhost:3000",
        function_name: str = "code_patch_architect",
        variant_name: str | None = None,
        max_tokens: int = 16_384,
    ):
        self.tz_url = tz_url
        self.function_name = function_name
        self.variant_name = variant_name
        self.max_tokens = max_tokens

    def run(
        self,
        hypothesis: Hypothesis | None,
        state: SEPLState,
        prior_attempts: list[dict] | None = None,
        extra_files: list | None = None,
    ) -> tuple[Modification | None, LineageRecord]:
        from architect import (
            build_architect_prompt,
            call_architect,
            extract_whole_files,
            needs_more_context,
        )
        t0 = time.time()

        messages = build_architect_prompt(
            state.issue, state.worktree,
            prior_attempts=prior_attempts,
            extra_files=extra_files,
        )

        try:
            response = call_architect(
                messages,
                tz_url=self.tz_url,
                function_name=self.function_name,
                variant_name=self.variant_name,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            rec = now_record(
                self.name, state.iteration, OperatorStatus.FAIL,
                duration_s=time.time() - t0,
                note=f"call_failed: {type(e).__name__}: {str(e)[:150]}",
            )
            return None, rec

        if ctx := needs_more_context(response):
            rec = now_record(
                self.name, state.iteration, OperatorStatus.SKIP,
                duration_s=time.time() - t0,
                metrics={"need_context": ctx, "response_len": len(response)},
                note="model_requested_context",
            )
            # Stash the response via an "empty" Modification so the driver
            # can preserve it as a prior attempt when re-calling with
            # the requested file attached.
            return Modification(files={}, source_response=response), rec

        files = extract_whole_files(response)
        if not files:
            rec = now_record(
                self.name, state.iteration, OperatorStatus.FAIL,
                duration_s=time.time() - t0,
                metrics={"response_len": len(response)},
                note="no_file_blocks_in_response",
            )
            return Modification(files={}, source_response=response), rec

        mod = Modification(files=files, source_response=response)
        rec = now_record(
            self.name, state.iteration, OperatorStatus.OK,
            input_digest=text_digest(hypothesis.summary if hypothesis else None),
            output_digest=mod.digest(),
            duration_s=time.time() - t0,
            metrics={
                "files_modified": sorted(files.keys()),
                "response_len": len(response),
            },
        )
        return mod, rec


class Evaluate:
    """Apply a Modification to the worktree and run the verifier.
    Combines architect.apply_whole_files + the externally-provided
    verifier_fn. On apply failure, skips the verifier and surfaces the
    apply error as an APPLY-category gate."""
    name = "evaluate"

    def __init__(self, verifier_fn: Callable[[Any], dict]):
        self.verifier_fn = verifier_fn

    def run(
        self,
        modification: Modification,
        state: SEPLState,
    ) -> tuple[EvalResult, LineageRecord]:
        from architect import apply_whole_files
        t0 = time.time()

        # Guard: empty-files Modification (need_context or parse fail)
        if not modification.files:
            ev = EvalResult(
                all_green=False,
                gates=(GateResult(
                    name="apply",
                    passed=False,
                    stderr_tail="no files to write (empty Modification)",
                    category=ErrorCategory.APPLY,
                ),),
                duration_s=time.time() - t0,
            )
            rec = now_record(
                self.name, state.iteration, OperatorStatus.SKIP,
                input_digest=modification.digest(),
                duration_s=time.time() - t0,
                metrics={"stage": "pre-apply", "reason": "empty_modification"},
            )
            return ev, rec

        applied = apply_whole_files(modification.files, state.worktree)
        if not applied.get("ok"):
            ev = EvalResult(
                all_green=False,
                gates=(GateResult(
                    name="apply",
                    passed=False,
                    stderr_tail=(applied.get("error") or "unknown apply failure")[:3000],
                    category=ErrorCategory.APPLY,
                ),),
                duration_s=time.time() - t0,
            )
            rec = now_record(
                self.name, state.iteration, OperatorStatus.FAIL,
                input_digest=modification.digest(),
                duration_s=time.time() - t0,
                metrics={"stage": "apply", "error": applied.get("error")},
            )
            return ev, rec

        try:
            verify = self.verifier_fn(state.worktree)
        except Exception as e:
            ev = EvalResult(
                all_green=False,
                gates=(GateResult(
                    name="verifier",
                    passed=False,
                    stderr_tail=f"{type(e).__name__}: {e}"[:3000],
                    category=ErrorCategory.UNKNOWN,
                ),),
                duration_s=time.time() - t0,
            )
            rec = now_record(
                self.name, state.iteration, OperatorStatus.FAIL,
                input_digest=modification.digest(),
                duration_s=time.time() - t0,
                metrics={"stage": "verifier", "exception": type(e).__name__},
            )
            return ev, rec

        ev = EvalResult.from_verifier_dict(verify, duration_s=time.time() - t0)
        status = OperatorStatus.OK if ev.all_green else OperatorStatus.FAIL
        rec = now_record(
            self.name, state.iteration, status,
            input_digest=modification.digest(),
            duration_s=time.time() - t0,
            metrics={
                "all_green": ev.all_green,
                "failing_gates": [g.name for g in ev.gates if not g.passed],
                "failing_categories": sorted({c.value for c in ev.failing_categories}),
                "files_written": applied.get("files", []),
            },
        )
        return ev, rec


class Commit:
    """Decide whether to commit or roll back the iteration. Mirrors the
    existing architect.py semantics:
      - green: leave files in place; outer run.py merges + closes
      - red:   reset_worktree so the next iteration's Improve starts from HEAD"""
    name = "commit"

    def run(self, state: SEPLState) -> tuple[str, LineageRecord]:
        from architect import reset_worktree
        t0 = time.time()

        if state.last_eval is None:
            rec = now_record(
                self.name, state.iteration, OperatorStatus.SKIP,
                duration_s=time.time() - t0,
                note="no_eval",
            )
            return "skipped", rec

        if state.last_eval.all_green:
            rec = now_record(
                self.name, state.iteration, OperatorStatus.OK,
                duration_s=time.time() - t0,
                note="committed",
            )
            return "committed", rec

        reset_worktree(state.worktree)
        rec = now_record(
            self.name, state.iteration, OperatorStatus.ROLLBACK,
            duration_s=time.time() - t0,
            note="reverted_failed_patch",
        )
        return "reverted", rec


class Operator(Protocol):
    """Loose marker: all SEPL operators have a name and a run method.
    Signatures of run() vary per operator (see class docstrings above)."""
    name: str
