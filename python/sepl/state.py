from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, replace
from typing import Any

from .types import ErrorCategory, EvalResult, Modification


@dataclass(frozen=True)
class SEPLState:
    """Immutable-ish snapshot threaded through the operator loop.

    Operators return a new SEPLState via `state.advance(...)`; they never
    mutate in-place. This gives us free rollback: `state.history[-2]` is
    the prior-iteration snapshot."""
    issue: dict[str, Any]
    worktree: pathlib.Path
    iteration: int = 0
    last_response: str | None = None
    last_modification: Modification | None = None
    last_eval: EvalResult | None = None
    history: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def issue_id(self) -> str:
        return str(self.issue.get("id", "unknown"))

    @property
    def consecutive_same_category_fails(self) -> int:
        """How many recent iterations failed with the same top error
        category. Used by the Phase 1.3 driver to trigger rollback."""
        categories: list[ErrorCategory] = []
        for h in reversed(self.history):
            cat = h.get("top_category")
            if cat is None:
                break
            categories.append(ErrorCategory(cat))
            if len(categories) >= 2 and categories[-1] != categories[0]:
                break
        if not categories:
            return 0
        first = categories[0]
        streak = 0
        for c in categories:
            if c == first:
                streak += 1
            else:
                break
        return streak

    def advance(
        self,
        *,
        response: str | None = None,
        modification: Modification | None = None,
        eval_result: EvalResult | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SEPLState:
        """Produce a new state for the next iteration. Appends the
        current iteration to history with a compact snapshot."""
        top_cat: ErrorCategory | None = None
        if eval_result is not None and not eval_result.all_green:
            first_fail = eval_result.first_failing
            if first_fail is not None:
                top_cat = first_fail.category
        snapshot: dict[str, Any] = {
            "iteration": self.iteration,
            "response_present": self.last_response is not None,
            "mod_digest": self.last_modification.digest() if self.last_modification else "",
            "eval_all_green": self.last_eval.all_green if self.last_eval else None,
            "top_category": top_cat.value if top_cat else None,
        }
        if extra:
            snapshot.update(extra)
        return replace(
            self,
            iteration=self.iteration + 1,
            last_response=response if response is not None else self.last_response,
            last_modification=modification if modification is not None else self.last_modification,
            last_eval=eval_result if eval_result is not None else self.last_eval,
            history=self.history + (snapshot,),
        )
