"""Regret matching and average-strategy persistence for Leduc Poker."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cards import Card, Rank
from pokerlab.research.leduc.information_sets import InformationSet


def regret_matching(
    cumulative_regrets: Mapping[Action, float], legal: Sequence[Action]
) -> dict[Action, float]:
    if not legal or len(set(legal)) != len(legal):
        raise ValueError("legal actions must be nonempty and unique")
    values = {action: cumulative_regrets.get(action, 0.0) for action in legal}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("regrets must be finite")
    positive = {action: max(0.0, value) for action, value in values.items()}
    total = sum(positive.values())
    if total == 0:
        return {action: 1.0 / len(legal) for action in legal}
    return {action: value / total for action, value in positive.items()}


def normalize_average_strategy(
    weights: Mapping[Action, float], legal: Sequence[Action]
) -> dict[Action, float]:
    if not legal or len(set(legal)) != len(legal):
        raise ValueError("legal actions must be nonempty and unique")
    values = {action: weights.get(action, 0.0) for action in legal}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("strategy weights must be finite and nonnegative")
    total = sum(values.values())
    if total <= 0:
        return {action: 1.0 / len(legal) for action in legal}
    return {action: value / total for action, value in values.items()}


def save_strategy_snapshot(
    strategy: Mapping[InformationSet, Mapping[Action, float]], path: str | Path
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for info, distribution in sorted(
        strategy.items(), key=lambda item: item[0].identifier
    ):
        records.append(
            {
                "player": info.player,
                "private_card": info.private_card.label,
                "public_card": None
                if info.public_card is None
                else info.public_card.label,
                "round_index": info.round_index,
                "histories": [
                    [action.value for action in history] for history in info.histories
                ],
                "strategy": {
                    action.value: probability
                    for action, probability in distribution.items()
                },
            }
        )
    target.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return target


def load_strategy_snapshot(
    path: str | Path,
) -> dict[InformationSet, dict[Action, float]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    cards = {
        card.label: card for rank in Rank for card in (Card(rank, 0), Card(rank, 1))
    }
    result: dict[InformationSet, dict[Action, float]] = {}
    for record in records:
        histories = tuple(
            tuple(Action(action) for action in history)
            for history in record["histories"]
        )
        info = InformationSet(
            int(record["player"]),
            cards[record["private_card"]],
            None if record["public_card"] is None else cards[record["public_card"]],
            int(record["round_index"]),
            histories,  # type: ignore[arg-type]
        )
        distribution = {
            Action(action): float(probability)
            for action, probability in record["strategy"].items()
        }
        from pokerlab.research.leduc.betting import legal_actions

        legal = legal_actions(histories[info.round_index])
        if set(distribution) != set(legal):
            raise ValueError(f"strategy has illegal actions for {info.identifier}")
        if (
            not distribution
            or any(
                not math.isfinite(value) or value < 0 for value in distribution.values()
            )
            or not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-9)
        ):
            raise ValueError(f"invalid strategy distribution for {info.identifier}")
        result[info] = distribution
    return result
