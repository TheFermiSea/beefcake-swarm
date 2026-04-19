"""Unit tests for the 5 SEPL operators. Each operator is tested in
isolation with synthetic inputs; no live model calls."""
from __future__ import annotations

import pathlib

import pytest

from sepl import (
    Commit,
    ErrorCategory,
    EvalResult,
    Evaluate,
    GateResult,
    Hypothesis,
    Modification,
    OperatorStatus,
    Reflect,
    SEPLState,
    Select,
)


# ──────────────────────────────────────────────────────────────────────────────
# Reflect
# ──────────────────────────────────────────────────────────────────────────────

def _state(worktree, **adv) -> SEPLState:
    s = SEPLState(issue={"id": "t", "title": "t", "description": ""},
                  worktree=worktree)
    if adv:
        s = s.advance(**adv)
    return s


def test_reflect_first_iteration_returns_empty(fresh_worktree):
    hyps, rec = Reflect().run(_state(fresh_worktree))
    assert hyps == ()
    assert rec.status == OperatorStatus.OK
    assert rec.note == "first_iteration"


def test_reflect_all_green_skips(fresh_worktree):
    ev = EvalResult(all_green=True, gates=())
    hyps, rec = Reflect().run(_state(fresh_worktree, eval_result=ev))
    assert hyps == ()
    assert rec.status == OperatorStatus.SKIP
    assert rec.note == "all_green"


def test_reflect_parses_multiple_compile_errors(fresh_worktree):
    gate = GateResult(
        name="cargo-check", passed=False,
        stderr_tail=(
            "error[E0308]: mismatched types\n"
            "  --> src/foo.rs:10:5\n"
            "error[E0412]: cannot find type Bar\n"
            "  --> src/bar.rs:22:10"
        ),
        category=ErrorCategory.COMPILE,
    )
    ev = EvalResult(all_green=False, gates=(gate,))
    hyps, rec = Reflect().run(_state(fresh_worktree, eval_result=ev))
    assert len(hyps) == 2
    assert all(h.category == ErrorCategory.COMPILE for h in hyps)
    assert "E0308" in hyps[0].summary
    assert hyps[0].location == "src/foo.rs:10:5"
    assert hyps[1].location == "src/bar.rs:22:10"
    assert rec.metrics["hyp_count"] == 2


def test_reflect_caps_at_three_hypotheses_per_gate(fresh_worktree):
    lines = "\n".join(f"error[E{i:04d}]: x\n  --> a.rs:{i}:1" for i in range(10))
    gate = GateResult(name="cargo-check", passed=False, stderr_tail=lines,
                      category=ErrorCategory.COMPILE)
    ev = EvalResult(all_green=False, gates=(gate,))
    hyps, _ = Reflect().run(_state(fresh_worktree, eval_result=ev))
    assert len(hyps) == 3, f"expected 3, got {len(hyps)}"


def test_reflect_handles_test_failures(fresh_worktree):
    gate = GateResult(
        name="cargo-test", passed=False,
        stderr_tail="test foo::bar ... FAILED\ntest baz::qux ... FAILED",
        category=ErrorCategory.TEST,
    )
    ev = EvalResult(all_green=False, gates=(gate,))
    hyps, _ = Reflect().run(_state(fresh_worktree, eval_result=ev))
    assert len(hyps) == 2
    assert hyps[0].category == ErrorCategory.TEST
    assert "foo::bar" in hyps[0].summary


# ──────────────────────────────────────────────────────────────────────────────
# Select
# ──────────────────────────────────────────────────────────────────────────────

def test_select_empty_returns_none(fresh_worktree):
    chosen, rec = Select().run((), _state(fresh_worktree))
    assert chosen is None
    assert rec.status == OperatorStatus.SKIP


def test_select_prioritizes_compile_over_clippy(fresh_worktree):
    clippy_h = Hypothesis(category=ErrorCategory.CLIPPY, location=None,
                          summary="c", raw_excerpt="")
    compile_h = Hypothesis(category=ErrorCategory.COMPILE, location=None,
                           summary="E", raw_excerpt="")
    chosen, _ = Select().run((clippy_h, compile_h), _state(fresh_worktree))
    assert chosen.category == ErrorCategory.COMPILE


def test_select_prioritizes_apply_above_all(fresh_worktree):
    compile_h = Hypothesis(category=ErrorCategory.COMPILE, location=None,
                           summary="c", raw_excerpt="")
    apply_h = Hypothesis(category=ErrorCategory.APPLY, location=None,
                         summary="a", raw_excerpt="")
    test_h = Hypothesis(category=ErrorCategory.TEST, location=None,
                        summary="t", raw_excerpt="")
    chosen, _ = Select().run((compile_h, test_h, apply_h), _state(fresh_worktree))
    assert chosen.category == ErrorCategory.APPLY


def test_select_stable_tiebreak_by_input_order(fresh_worktree):
    h1 = Hypothesis(category=ErrorCategory.COMPILE, location=None,
                    summary="first", raw_excerpt="")
    h2 = Hypothesis(category=ErrorCategory.COMPILE, location=None,
                    summary="second", raw_excerpt="")
    chosen, _ = Select().run((h1, h2), _state(fresh_worktree))
    assert chosen.summary == "first"


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────

def test_evaluate_greens_on_successful_apply_and_verify(fresh_worktree, green_verifier):
    mod = Modification(
        files={"foo.rs": "#[derive(Debug, Clone)]\npub struct Foo;\n"},
        source_response="x",
    )
    ev, rec = Evaluate(green_verifier).run(mod, _state(fresh_worktree))
    assert ev.all_green
    assert rec.status == OperatorStatus.OK
    assert (fresh_worktree / "foo.rs").read_text().startswith("#[derive(Debug, Clone)]")


def test_evaluate_reds_with_category_classification(fresh_worktree, red_verifier):
    mod = Modification(
        files={"foo.rs": "pub struct Foo;\n"},
        source_response="x",
    )
    ev, rec = Evaluate(red_verifier).run(mod, _state(fresh_worktree))
    assert not ev.all_green
    assert rec.status == OperatorStatus.FAIL
    assert rec.metrics["failing_categories"] == ["compile"]


def test_evaluate_rejects_path_escape(fresh_worktree, green_verifier):
    bad = Modification(files={"../etc/passwd": "x"}, source_response="")
    ev, rec = Evaluate(green_verifier).run(bad, _state(fresh_worktree))
    assert not ev.all_green
    assert ev.gates[0].category == ErrorCategory.APPLY


def test_evaluate_skips_empty_modification(fresh_worktree, green_verifier):
    empty = Modification(files={}, source_response="NEED_CONTEXT: foo.rs")
    ev, rec = Evaluate(green_verifier).run(empty, _state(fresh_worktree))
    assert rec.status == OperatorStatus.SKIP
    assert ev.gates[0].category == ErrorCategory.APPLY


def test_evaluate_handles_verifier_exception(fresh_worktree):
    def boom(_wt):
        raise RuntimeError("verifier crash")

    mod = Modification(
        files={"foo.rs": "pub struct Foo;\n"},
        source_response="x",
    )
    ev, rec = Evaluate(boom).run(mod, _state(fresh_worktree))
    assert not ev.all_green
    assert rec.status == OperatorStatus.FAIL
    assert rec.metrics["exception"] == "RuntimeError"


# ──────────────────────────────────────────────────────────────────────────────
# Commit
# ──────────────────────────────────────────────────────────────────────────────

def test_commit_green_returns_committed(fresh_worktree):
    ev = EvalResult(all_green=True, gates=())
    status, rec = Commit().run(_state(fresh_worktree, eval_result=ev))
    assert status == "committed"
    assert rec.status == OperatorStatus.OK


def test_commit_red_resets_worktree(fresh_worktree):
    # Dirty the worktree with something that should get cleaned
    (fresh_worktree / "foo.rs").write_text("uncommitted garbage\n")
    red_gate = GateResult(name="x", passed=False, stderr_tail="",
                          category=ErrorCategory.COMPILE)
    ev = EvalResult(all_green=False, gates=(red_gate,))
    status, rec = Commit().run(_state(fresh_worktree, eval_result=ev))
    assert status == "reverted"
    assert rec.status == OperatorStatus.ROLLBACK
    # Foo.rs reverted back to original
    assert (fresh_worktree / "foo.rs").read_text().startswith("#[derive(Debug)]\n")


def test_commit_no_eval_skips(fresh_worktree):
    status, rec = Commit().run(_state(fresh_worktree))  # no advance -> no last_eval
    assert status == "skipped"
    assert rec.status == OperatorStatus.SKIP
