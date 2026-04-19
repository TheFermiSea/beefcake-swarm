from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .types import OperatorStatus


@dataclass
class LineageRecord:
    """One row of the SEPL audit trail. Written as a JSONL line to
    .swarm/lineage/<issue-id>.jsonl after every operator call."""
    op: str
    iteration: int
    ts: float
    status: OperatorStatus
    input_digest: str = ""
    output_digest: str = ""
    duration_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, sort_keys=True)


class LineageWriter:
    """Append-only JSONL writer. One file per issue. Never raises on
    write failure (lineage is best-effort observability, not the critical
    path). Thread-safe for append-only use."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: LineageRecord) -> None:
        try:
            with self.path.open("a") as f:
                f.write(record.to_json())
                f.write("\n")
        except Exception:
            pass

    def read_all(self) -> list[dict[str, Any]]:
        """Utility for tests + the Phase 5 validator. Tolerates partial
        writes and a missing file."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def now_record(
    op: str,
    iteration: int,
    status: OperatorStatus,
    *,
    input_digest: str = "",
    output_digest: str = "",
    duration_s: float = 0.0,
    metrics: dict[str, Any] | None = None,
    note: str = "",
) -> LineageRecord:
    """Convenience constructor for operators."""
    return LineageRecord(
        op=op,
        iteration=iteration,
        ts=time.time(),
        status=status,
        input_digest=input_digest,
        output_digest=output_digest,
        duration_s=duration_s,
        metrics=metrics or {},
        note=note,
    )
