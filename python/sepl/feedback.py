"""Phase 4: TZ feedback integration.

On SEPL.Commit outcomes, emit a structured feedback event tagged with
the variant TZ actually served. Writes to `.swarm/feedback/<variant>.jsonl`
so Phase 5's validator can aggregate without hitting a live service.

Also exposes a registry-propagation helper that updates RSPL Prompt
resources' `win_count`/`loss_count` metadata based on the accumulated
JSONL, giving us a unified view of variant performance in one place.

A future extension can POST these events to TZ's `/feedback` endpoint
to drive weight evolution; for now we write locally so the loop is
closed without depending on TZ uptime."""
from __future__ import annotations

import json
import pathlib
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from rspl import ResourceRegistry, ResourceType


FEEDBACK_DIR = pathlib.Path(".swarm/feedback")  # relative to repo root

# Model names from TZ look like:
#   tensorzero::function_name::<fn>::variant_name::<variant>
_VARIANT_RE = re.compile(
    r"tensorzero::function_name::(?P<fn>[^:]+)::variant_name::(?P<variant>[^:]+)"
)


def parse_variant(model_str: str | None) -> tuple[str, str] | None:
    """Extract (function, variant) from TZ's model string, or None if
    this wasn't a TZ-routed call."""
    if not model_str:
        return None
    m = _VARIANT_RE.search(model_str)
    if not m:
        return None
    return m.group("fn"), m.group("variant")


@dataclass
class FeedbackEvent:
    """One SEPL iteration outcome, tagged with the variant TZ served."""
    ts: float
    issue_id: str
    function: str
    variant: str
    outcome: str            # "resolved" | "rollback" | "parse_fail" | "context_needed"
    iteration: int
    wall_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class OutcomeLogger:
    """Append-only writer of FeedbackEvents, plus read helpers."""

    def __init__(self, feedback_root: pathlib.Path | None = None):
        self.root = feedback_root or FEEDBACK_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, function: str, variant: str) -> pathlib.Path:
        safe = f"{function}__{variant}".replace("/", "_").replace(":", "_")
        return self.root / f"{safe}.jsonl"

    def emit(self, event: FeedbackEvent) -> None:
        """Best-effort append. Swallows IO errors (feedback is
        observability, not the critical path)."""
        try:
            with self._path(event.function, event.variant).open("a") as f:
                f.write(event.to_json())
                f.write("\n")
        except OSError:
            pass

    def read(self, function: str, variant: str) -> list[FeedbackEvent]:
        path = self._path(function, variant)
        if not path.exists():
            return []
        out: list[FeedbackEvent] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    out.append(FeedbackEvent(**d))
                except (json.JSONDecodeError, TypeError):
                    continue
        return out

    def tally(self, function: str, variant: str) -> dict[str, int]:
        """Aggregate a variant's outcomes: {resolved, rollback, parse_fail, ..., total}."""
        counts: dict[str, int] = {"total": 0}
        for ev in self.read(function, variant):
            counts[ev.outcome] = counts.get(ev.outcome, 0) + 1
            counts["total"] += 1
        return counts

    def win_rate(self, function: str, variant: str) -> float:
        """resolved / total. Returns 0.0 for empty logs."""
        t = self.tally(function, variant)
        if not t["total"]:
            return 0.0
        return t.get("resolved", 0) / t["total"]

    def propagate_to_registry(self, registry: ResourceRegistry) -> int:
        """Update each RSPL Prompt resource with its accumulated
        win_count / loss_count / total metadata. Returns number of
        resources touched."""
        touched = 0
        for rv in registry.list_by_type(ResourceType.PROMPT):
            fn = rv.resource.metadata.get("function")
            variant = rv.resource.metadata.get("variant")
            if not (fn and variant):
                continue
            t = self.tally(fn, variant)
            if t["total"] == 0:
                continue
            rid = f"{ResourceType.PROMPT.value}:{rv.resource.name}"
            registry.update(
                rid,
                metadata_patch={
                    "win_count": t.get("resolved", 0),
                    "loss_count": t.get("rollback", 0) + t.get("parse_fail", 0),
                    "feedback_total": t["total"],
                    "win_rate": round(
                        t.get("resolved", 0) / t["total"], 4,
                    ),
                },
                reason="phase4_feedback_propagation",
            )
            touched += 1
        return touched


# ──────────────────────────────────────────────────────────────────────────────
# Driver-side emit helper for SEPL
# ──────────────────────────────────────────────────────────────────────────────

def emit_outcome(
    logger: OutcomeLogger | None,
    *,
    issue_id: str,
    model_str: str | None,
    outcome: str,
    iteration: int,
    wall_s: float = 0.0,
    metrics: dict[str, Any] | None = None,
) -> FeedbackEvent | None:
    """Factor: parse variant + emit. Returns the event or None if we
    can't parse the variant string."""
    if logger is None:
        return None
    parsed = parse_variant(model_str)
    if parsed is None:
        return None
    function, variant = parsed
    ev = FeedbackEvent(
        ts=time.time(),
        issue_id=issue_id,
        function=function,
        variant=variant,
        outcome=outcome,
        iteration=iteration,
        wall_s=wall_s,
        metrics=metrics or {},
    )
    logger.emit(ev)
    return ev
