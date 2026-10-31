"""Regret matching, average strategies, and strategy snapshot serialization."""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from pokerlab.research.kuhn.game import KuhnAction, KuhnCard
from pokerlab.research.kuhn.information_sets import InformationSet


def regret_matching(
    cumulative_regrets: Mapping[KuhnAction, float],
    legal_actions: Sequence[KuhnAction],
) -> dict[KuhnAction, float]:
    """Normalize positive regrets; use uniform play when none are positive."""
    if not legal_actions or len(set(legal_actions)) != len(legal_actions):
        raise ValueError("legal actions must be nonempty and unique")
    raw = {action: cumulative_regrets.get(action, 0.0) for action in legal_actions}
    if any(not math.isfinite(value) for value in raw.values()):
        raise ValueError("regrets must be finite")
    positive = {action: max(0.0, value) for action, value in raw.items()}
    total = sum(positive.values())
    if total <= 0:
        probability = 1.0 / len(legal_actions)
        return {action: probability for action in legal_actions}
    return {action: value / total for action, value in positive.items()}


def normalize_average_strategy(
    strategy_sum: Mapping[KuhnAction, float],
    legal_actions: Sequence[KuhnAction],
) -> dict[KuhnAction, float]:
    """Normalize cumulative average-strategy mass over legal actions only."""
    if not legal_actions or len(set(legal_actions)) != len(legal_actions):
        raise ValueError("legal actions must be nonempty and unique")
    values = {action: strategy_sum.get(action, 0.0) for action in legal_actions}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("strategy weights must be finite and nonnegative")
    total = sum(values.values())
    if total <= 0:
        return {action: 1.0 / len(legal_actions) for action in legal_actions}
    return {action: value / total for action, value in values.items()}


def save_strategy_snapshot(
    strategy: Mapping[InformationSet, Mapping[KuhnAction, float]], path: str | Path
) -> Path:
    """Write average strategy probabilities in a stable, readable JSON format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for info_set, distribution in sorted(
        strategy.items(), key=lambda item: item[0].identifier
    ):
        records.append(
            {
                "player": info_set.player,
                "private_card": info_set.private_card.label,
                "history": [action.value for action in info_set.history],
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
) -> dict[InformationSet, dict[KuhnAction, float]]:
    """Load a strategy snapshot and reject malformed probability distributions."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[InformationSet, dict[KuhnAction, float]] = {}
    card_by_label = {card.label: card for card in KuhnCard}
    for record in records:
        info_set = InformationSet(
            int(record["player"]),
            card_by_label[record["private_card"]],
            tuple(KuhnAction(value) for value in record["history"]),
        )
        distribution = {
            KuhnAction(action): float(probability)
            for action, probability in record["strategy"].items()
        }
        from pokerlab.research.kuhn.game import legal_actions_for_history

        if set(distribution) != set(legal_actions_for_history(info_set.history)):
            raise ValueError(f"strategy has illegal actions for {info_set.identifier}")
        if (
            not distribution
            or any(
                probability < 0 or not math.isfinite(probability)
                for probability in distribution.values()
            )
            or not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-9)
        ):
            raise ValueError(f"invalid strategy distribution for {info_set.identifier}")
        result[info_set] = distribution
    return result
