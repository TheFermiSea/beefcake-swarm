"""Self-Evolution Protocol Layer (SEPL) for beefcake-swarm.

Implements the 5-operator closed loop from Zhang 2026 (arXiv:2604.15034):
  Reflect -> Select -> Improve -> Evaluate -> Commit

Phase 1 scope: typed skeletons + LineageWriter.
Phase 1.2 adds concrete operator implementations.
Phase 1.3 wires the loop into python/architect.py.
"""

from .lineage import LineageRecord, LineageWriter, now_record
from .operators import Commit, Evaluate, Improve, Operator, Reflect, Select
from .state import SEPLState
from .types import (
    ErrorCategory,
    EvalResult,
    GateResult,
    Hypothesis,
    Modification,
    OperatorStatus,
    Trace,
    text_digest,
)

__all__ = [
    "Commit",
    "ErrorCategory",
    "EvalResult",
    "Evaluate",
    "GateResult",
    "Hypothesis",
    "Improve",
    "LineageRecord",
    "LineageWriter",
    "Modification",
    "Operator",
    "OperatorStatus",
    "Reflect",
    "SEPLState",
    "Select",
    "Trace",
    "now_record",
    "text_digest",
]
