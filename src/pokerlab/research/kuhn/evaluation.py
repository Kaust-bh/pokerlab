"""Exact expected value, best responses, NashConv, and exploitability."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from pokerlab.research.kuhn.game import KuhnAction, KuhnGame, KuhnState
from pokerlab.research.kuhn.information_sets import InformationSet

StrategyProfile = Mapping[InformationSet, Mapping[KuhnAction, float]]


@dataclass(frozen=True, slots=True)
class NashEvaluation:
    expected_value_player0: float
    best_response_value_player0: float
    best_response_value_player1: float
    nashconv: float
    exploitability: float


def _distribution(
    profile: StrategyProfile,
    info_set: InformationSet,
    legal_actions: tuple[KuhnAction, ...],
) -> dict[KuhnAction, float]:
    supplied = profile.get(info_set)
    if supplied is None:
        return {action: 1 / len(legal_actions) for action in legal_actions}
    if set(supplied) != set(legal_actions):
        raise ValueError(
            f"strategy has illegal or missing actions at {info_set.identifier}"
        )
    if any(
        not isinstance(probability, (int, float)) for probability in supplied.values()
    ):
        raise ValueError("strategy probabilities must be numeric")
    if any(not math.isfinite(probability) for probability in supplied.values()):
        raise ValueError("strategy probabilities must be finite")
    if any(probability < 0 for probability in supplied.values()):
        raise ValueError("strategy probabilities cannot be negative")
    if not math.isclose(sum(supplied.values()), 1.0, abs_tol=1e-9):
        raise ValueError("strategy probabilities must sum to one")
    return dict(supplied)


def expected_game_value(profile: StrategyProfile) -> float:
    """Return exact expected net chips for Player 0 against the supplied profile."""

    def value(state: KuhnState) -> float:
        if state.is_terminal:
            return float(state.utility(0))
        info_set = InformationSet.from_state(state)
        distribution = _distribution(profile, info_set, state.legal_actions)
        return sum(
            probability * value(state.act(action))
            for action, probability in distribution.items()
        )

    deals = tuple(KuhnGame.deals())
    return sum(value(KuhnGame.initial_state(deal)) for deal in deals) / len(deals)


def _all_information_sets(player: int) -> dict[InformationSet, tuple[KuhnAction, ...]]:
    found: dict[InformationSet, tuple[KuhnAction, ...]] = {}

    def visit(state: KuhnState) -> None:
        if state.is_terminal:
            return
        if state.current_player == player:
            info_set = InformationSet.from_state(state)
            found[info_set] = state.legal_actions
        for action in state.legal_actions:
            visit(state.act(action))

    for deal in KuhnGame.deals():
        visit(KuhnGame.initial_state(deal))
    return found


def best_response_value(profile: StrategyProfile, player: int) -> float:
    """Find the exact pure best response, constrained to the player's information sets."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    information_sets = _all_information_sets(player)
    ordered = sorted(information_sets.items(), key=lambda item: item[0].identifier)
    best_value = float("-inf")
    for choices in product(*(actions for _, actions in ordered)):
        response: dict[InformationSet, dict[KuhnAction, float]] = {
            info_set: {action: float(action == chosen) for action in actions}
            for (info_set, actions), chosen in zip(ordered, choices, strict=True)
        }
        combined: dict[InformationSet, Mapping[KuhnAction, float]] = {
            info_set: distribution
            for info_set, distribution in profile.items()
            if info_set.player != player
        }
        combined.update(response)
        value0 = expected_game_value(combined)
        player_value = value0 if player == 0 else -value0
        best_value = max(best_value, player_value)
    return best_value


def evaluate_nash(profile: StrategyProfile) -> NashEvaluation:
    """Compute NashConv as both players' unilateral gains; exploitability=NC/2."""
    value0 = expected_game_value(profile)
    br0 = best_response_value(profile, 0)
    br1 = best_response_value(profile, 1)
    gain0 = max(0.0, br0 - value0)
    gain1 = max(0.0, br1 + value0)
    nashconv = gain0 + gain1
    return NashEvaluation(value0, br0, br1, nashconv, nashconv / 2)
