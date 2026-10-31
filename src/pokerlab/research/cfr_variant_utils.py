"""Shared mathematical helpers for full-tree CFR variants."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CFRVariantRecord:
    iteration: int
    expected_game_value_player0: float
    expected_game_value_player1: float
    nashconv: float
    exploitability: float
    training_elapsed_seconds: float
    evaluation_elapsed_seconds: float
    iterations_per_second: float
    seed: int
    information_sets: int
    traversals: int
    player_updates: int


def train_variant(
    solver,
    evaluator: Callable,
    *,
    iterations: int | None = None,
    evaluation_interval: int | None = None,
    logging_interval: int | None = None,
    callback: Callable[[CFRVariantRecord], None] | None = None,
) -> tuple[CFRVariantRecord, ...]:
    """Train a variant while timing training and exact evaluation separately."""
    count = solver.config.iterations if iterations is None else iterations
    eval_every = (
        solver.config.evaluation_interval
        if evaluation_interval is None
        else evaluation_interval
    )
    log_every = (
        solver.config.logging_interval if logging_interval is None else logging_interval
    )
    if count < 1 or eval_every < 1 or log_every < 1:
        raise ValueError("training intervals and iteration count must be positive")
    previous_records = len(solver.records)
    training_elapsed = 0.0
    initial_iteration = solver.iterations
    for local_iteration in range(1, count + 1):
        started = time.perf_counter()
        solver.run_iteration()
        training_elapsed += time.perf_counter() - started
        final = local_iteration == count
        if solver.iterations % eval_every != 0 and not final:
            continue
        evaluation_started = time.perf_counter()
        result = evaluator(solver.average_strategy_profile())
        evaluation_elapsed = time.perf_counter() - evaluation_started
        record = CFRVariantRecord(
            iteration=solver.iterations,
            expected_game_value_player0=result.expected_value_player0,
            expected_game_value_player1=-result.expected_value_player0,
            nashconv=result.nashconv,
            exploitability=result.exploitability,
            training_elapsed_seconds=training_elapsed,
            evaluation_elapsed_seconds=evaluation_elapsed,
            iterations_per_second=(
                local_iteration / training_elapsed if training_elapsed else math.inf
            ),
            seed=solver.config.seed,
            information_sets=len(solver._legal),
            traversals=solver.traversals,
            player_updates=solver.player_updates,
        )
        solver.records.append(record)
        if callback is not None and (solver.iterations % log_every == 0 or final):
            callback(record)
    if solver.iterations != initial_iteration + count:
        raise RuntimeError("training iteration count changed unexpectedly")
    return tuple(solver.records[previous_records:])


def regret_matching_plus(
    regrets: Mapping[Any, float], actions: Sequence[Any]
) -> dict[Any, float]:
    """Return RM+ over positive cumulative regrets, with uniform fallback."""
    if not actions:
        raise ValueError("regret matching requires at least one legal action")
    positive = {action: max(0.0, float(regrets[action])) for action in actions}
    total = math.fsum(positive.values())
    if total == 0.0:
        probability = 1.0 / len(actions)
        return dict.fromkeys(actions, probability)
    return {action: positive[action] / total for action in actions}


def dcfr_positive_discount(iteration: int, alpha: float) -> float:
    """Stable form of t**alpha / (t**alpha + 1)."""
    if iteration < 1:
        raise ValueError("iteration must be positive")
    return _logistic(math.log(iteration) * alpha)


def dcfr_negative_discount(iteration: int, beta: float) -> float:
    """Stable form of t**beta / (t**beta + 1)."""
    if iteration < 1:
        raise ValueError("iteration must be positive")
    return _logistic(math.log(iteration) * beta)


def dcfr_strategy_discount(iteration: int, gamma: float) -> float:
    """Stable form of (t / (t + 1))**gamma."""
    if iteration < 1:
        raise ValueError("iteration must be positive")
    if iteration <= 2**53:
        return math.exp(-gamma * math.log1p(1.0 / iteration))
    if gamma == 0.0:
        return 1.0
    # For large integer t, log(1 + 1/t) is indistinguishable from 1/t in
    # binary64. Keep gamma/t in log space so enormous integers never need to
    # be converted to float and large gamma values remain well-defined.
    log_gamma_over_t = math.log(gamma) - math.log(iteration)
    if log_gamma_over_t < math.log(float.fromhex("0x0.0000000000001p-1022")):
        exponent = 0.0
    elif log_gamma_over_t > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return 0.0
    else:
        exponent = math.exp(log_gamma_over_t)
    return math.exp(-exponent)


def _logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def discount_regret_sum(
    cumulative: float,
    instantaneous: float,
    iteration: int,
    alpha: float,
    beta: float,
) -> float:
    """Discount old regret by its sign, then add this iteration's regret."""
    factor = (
        dcfr_positive_discount(iteration, alpha)
        if cumulative > 0.0
        else dcfr_negative_discount(iteration, beta)
    )
    return factor * cumulative + instantaneous


def discount_strategy_sum(
    cumulative: float, own_reach_contribution: float, iteration: int, gamma: float
) -> float:
    """Discount old average mass and add the current own-reach mass."""
    return (
        dcfr_strategy_discount(iteration, gamma) * cumulative + own_reach_contribution
    )


def validate_dcfr_parameters(alpha: float, beta: float, gamma: float) -> None:
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("DCFR alpha and beta must be finite")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("DCFR gamma must be finite and nonnegative")
