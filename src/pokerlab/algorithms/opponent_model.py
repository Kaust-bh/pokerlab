"""Protocols and base classes for Bayesian opponent modeling."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pokerlab.agents.base import ActionRecord


@runtime_checkable
class OpponentRangeTracker(Protocol):
    """Maintains a belief distribution over opponent private cards."""

    def update(self, action_record: ActionRecord) -> None:
        """Update posterior belief given an observed opponent action."""
        ...

    def range_distribution(self) -> dict[Any, float]:
        """Return the current posterior distribution over opponent hands."""
        ...
