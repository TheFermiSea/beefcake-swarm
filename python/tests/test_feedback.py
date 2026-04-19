"""Phase 4: TZ feedback integration tests."""
from __future__ import annotations

import pathlib

import pytest

from rspl import ResourceRegistry, ResourceType, bootstrap_from_tensorzero_toml
from sepl import FeedbackEvent, OutcomeLogger, emit_outcome, parse_variant


# ──────────────────────────────────────────────────────────────────────────────
# parse_variant
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_variant_full_tz_string():
    m = "tensorzero::function_name::code_patch_architect::variant_name::minimax_m2_7"
    assert parse_variant(m) == ("code_patch_architect", "minimax_m2_7")


def test_parse_variant_inside_litellm_prefix():
    m = "openai/tensorzero::function_name::worker_code_edit::variant_name::qwen36_a3b"
    assert parse_variant(m) == ("worker_code_edit", "qwen36_a3b")


def test_parse_variant_returns_none_for_non_tz_string():
    assert parse_variant("gpt-4o") is None
    assert parse_variant(None) is None
    assert parse_variant("") is None


# ──────────────────────────────────────────────────────────────────────────────
# OutcomeLogger round-trip + tally
# ──────────────────────────────────────────────────────────────────────────────

def test_emit_and_read_round_trip(tmp_path):
    logger = OutcomeLogger(tmp_path)
    ev = FeedbackEvent(
        ts=1.0, issue_id="i1",
        function="code_patch_architect", variant="minimax_m2_7",
        outcome="resolved", iteration=1, wall_s=100.0,
    )
    logger.emit(ev)
    rows = logger.read("code_patch_architect", "minimax_m2_7")
    assert len(rows) == 1
    assert rows[0].outcome == "resolved"


def test_tally_counts_by_outcome(tmp_path):
    logger = OutcomeLogger(tmp_path)
    for outcome in ("resolved", "resolved", "resolved", "rollback", "parse_fail"):
        logger.emit(FeedbackEvent(
            ts=1.0, issue_id="x",
            function="f", variant="v",
            outcome=outcome, iteration=1,
        ))
    t = logger.tally("f", "v")
    assert t == {"total": 5, "resolved": 3, "rollback": 1, "parse_fail": 1}


def test_win_rate(tmp_path):
    logger = OutcomeLogger(tmp_path)
    for _ in range(3):
        logger.emit(FeedbackEvent(
            ts=1.0, issue_id="x", function="f", variant="v",
            outcome="resolved", iteration=1,
        ))
    for _ in range(1):
        logger.emit(FeedbackEvent(
            ts=1.0, issue_id="x", function="f", variant="v",
            outcome="rollback", iteration=1,
        ))
    assert logger.win_rate("f", "v") == 0.75


def test_win_rate_empty_returns_zero(tmp_path):
    logger = OutcomeLogger(tmp_path)
    assert logger.win_rate("f", "v") == 0.0


def test_read_is_robust_to_corrupt_lines(tmp_path):
    logger = OutcomeLogger(tmp_path)
    path = logger._path("f", "v")
    path.write_text(
        '{"ts":1,"issue_id":"x","function":"f","variant":"v","outcome":"resolved","iteration":1}\n'
        'not-json\n'
        '{"ts":2,"issue_id":"y","function":"f","variant":"v","outcome":"rollback","iteration":1}\n'
    )
    rows = logger.read("f", "v")
    assert len(rows) == 2
    assert [r.outcome for r in rows] == ["resolved", "rollback"]


# ──────────────────────────────────────────────────────────────────────────────
# emit_outcome helper (driver-side)
# ──────────────────────────────────────────────────────────────────────────────

def test_emit_outcome_routes_via_model_string(tmp_path):
    logger = OutcomeLogger(tmp_path)
    ev = emit_outcome(
        logger, issue_id="i1",
        model_str="tensorzero::function_name::code_patch_architect::variant_name::minimax_m2_7",
        outcome="resolved", iteration=1, wall_s=100.0,
    )
    assert ev is not None
    assert ev.function == "code_patch_architect"
    assert ev.variant == "minimax_m2_7"
    assert logger.tally("code_patch_architect", "minimax_m2_7")["resolved"] == 1


def test_emit_outcome_noop_on_unparsable_model(tmp_path):
    logger = OutcomeLogger(tmp_path)
    assert emit_outcome(
        logger, issue_id="i1", model_str="gpt-4o",
        outcome="resolved", iteration=1,
    ) is None


def test_emit_outcome_noop_when_logger_is_none():
    # No raise even with None logger
    assert emit_outcome(
        None, issue_id="i", model_str="t::function_name::f::variant_name::v",
        outcome="resolved", iteration=1,
    ) is None


# ──────────────────────────────────────────────────────────────────────────────
# Registry propagation (RSPL ↔ feedback integration)
# ──────────────────────────────────────────────────────────────────────────────

def test_propagate_to_registry_updates_prompt_metadata(tmp_path):
    # Set up a registry with 2 Prompt variants for a TZ function
    from rspl.types import Resource, ResourceType as RT
    reg = ResourceRegistry()
    for v, weight in [("good", 0.8), ("bad", 0.2)]:
        reg.register(
            Resource(
                name=f"code_patch_architect.{v}",
                resource_type=RT.PROMPT, description="",
                mapping=None, trainable=True,
                metadata={"function": "code_patch_architect", "variant": v, "weight": weight},
            ),
            implementation=f"tensorzero::function_name::code_patch_architect::variant_name::{v}",
        )

    # Emit outcomes
    logger = OutcomeLogger(tmp_path)
    for _ in range(9):
        logger.emit(FeedbackEvent(
            ts=1.0, issue_id="x", function="code_patch_architect",
            variant="good", outcome="resolved", iteration=1,
        ))
    logger.emit(FeedbackEvent(
        ts=1.0, issue_id="x", function="code_patch_architect",
        variant="good", outcome="rollback", iteration=3,
    ))
    for _ in range(3):
        logger.emit(FeedbackEvent(
            ts=1.0, issue_id="x", function="code_patch_architect",
            variant="bad", outcome="rollback", iteration=3,
        ))

    # Propagate
    touched = logger.propagate_to_registry(reg)
    assert touched == 2

    good = reg.get("prompt:code_patch_architect.good").resource.metadata
    assert good["win_count"] == 9
    assert good["loss_count"] == 1
    assert good["win_rate"] == 0.9

    bad = reg.get("prompt:code_patch_architect.bad").resource.metadata
    assert bad["win_count"] == 0
    assert bad["loss_count"] == 3
    assert bad["win_rate"] == 0.0


def test_propagate_skips_variants_with_no_feedback(tmp_path):
    from rspl.types import Resource, ResourceType as RT
    reg = ResourceRegistry()
    reg.register(
        Resource(
            name="f.untouched",
            resource_type=RT.PROMPT, description="",
            mapping=None, trainable=True,
            metadata={"function": "f", "variant": "untouched"},
        ),
        implementation="x",
    )
    logger = OutcomeLogger(tmp_path)
    touched = logger.propagate_to_registry(reg)
    assert touched == 0
    # Still only 1 version (no propagation update appended)
    assert len(reg.lineage("prompt:f.untouched").versions) == 1
