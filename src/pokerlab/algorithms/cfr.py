"""Core interfaces and abstractions for Counterfactual Regret Minimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CFRIterationRecord:
    """Metadata recorded during a single CFR iteration."""

    iteration: int
    training_seconds: float
    nodes_visited: int
    information_sets: int
    regret_sum_positive: float = 0.0
    regret_sum_negative: float = 0.0


@runtime_checkable
class CFRSolver(Protocol):
    """Protocol for CFR solver implementations across games."""

    iterations: int

    def run_iteration(self) -> int:
        """Execute one complete training iteration and return nodes traversed."""
        ...

    def average_strategy_profile(self) -> dict[Any, dict[Any, float]]:
        """Return the current normalized average strategy profile."""
        ...
