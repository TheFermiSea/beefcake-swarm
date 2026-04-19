from __future__ import annotations

from typing import Protocol, runtime_checkable

from .lineage import LineageRecord
from .state import SEPLState
from .types import EvalResult, Hypothesis, Modification


@runtime_checkable
class Operator(Protocol):
    """Protocol for every SEPL operator. Implementations live in Task 1.2.

    An operator takes the current SEPLState, performs one atomic step
    (reflection, selection, improvement, evaluation, or commit), and
    returns both a domain-specific output and a LineageRecord that
    describes what happened for audit purposes.

    Operators MUST NOT mutate SEPLState — they return a tuple and let
    the driver thread new state via SEPLState.advance()."""

    name: str

    def run(self, state: SEPLState) -> tuple[object, LineageRecord]:
        ...


class Reflect:
    """Parses the last verifier output + response into Hypotheses.

    Input: state.last_eval + state.last_response
    Output: tuple[Hypothesis, ...]

    Implementation deferred to Phase 1.2 (beefcake-ezi6a)."""
    name = "reflect"

    def run(self, state: SEPLState) -> tuple[tuple[Hypothesis, ...], LineageRecord]:
        raise NotImplementedError("Reflect.run: implement in Phase 1.2 (beefcake-ezi6a)")


class Select:
    """Picks the hypothesis to act on this iteration. Current policy:
    compile errors > test failures > clippy > format > other.

    Input: tuple[Hypothesis, ...]
    Output: Hypothesis | None (None signals: give up this iter)

    Implementation deferred to Phase 1.2 (beefcake-ezi6a)."""
    name = "select"

    def run(self, state: SEPLState) -> tuple[Hypothesis | None, LineageRecord]:
        raise NotImplementedError("Select.run: implement in Phase 1.2 (beefcake-ezi6a)")


class Improve:
    """Calls the architect model with the selected hypothesis as focus
    and parses the response into a Modification.

    Input: selected Hypothesis + SEPLState (for prompt assembly)
    Output: Modification | None (None = NEED_CONTEXT or parse failure)

    Implementation deferred to Phase 1.2 (beefcake-ezi6a)."""
    name = "improve"

    def run(self, state: SEPLState) -> tuple[Modification | None, LineageRecord]:
        raise NotImplementedError("Improve.run: implement in Phase 1.2 (beefcake-ezi6a)")


class Evaluate:
    """Applies the Modification to the worktree and runs the verifier.
    Translates verifier output into EvalResult.

    Input: Modification + SEPLState (for worktree path + verifier_fn)
    Output: EvalResult

    Implementation deferred to Phase 1.2 (beefcake-ezi6a)."""
    name = "evaluate"

    def run(self, state: SEPLState) -> tuple[EvalResult, LineageRecord]:
        raise NotImplementedError("Evaluate.run: implement in Phase 1.2 (beefcake-ezi6a)")


class Commit:
    """On green: stage a git commit recording the successful iteration.
    On red: leave the worktree in its pre-Apply state (or reset if the
    driver decides to roll back).

    Input: SEPLState (post-Evaluate)
    Output: status string ("committed" | "skipped" | "rolled_back")

    Implementation deferred to Phase 1.2 (beefcake-ezi6a)."""
    name = "commit"

    def run(self, state: SEPLState) -> tuple[str, LineageRecord]:
        raise NotImplementedError("Commit.run: implement in Phase 1.2 (beefcake-ezi6a)")
