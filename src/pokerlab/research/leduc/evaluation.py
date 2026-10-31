"""Exact profile values, information-set best responses, and exploitability."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.game import LeducGame, LeducState
from pokerlab.research.leduc.information_sets import InformationSet

StrategyProfile = Mapping[InformationSet, Mapping[Action, float]]


@dataclass(frozen=True, slots=True)
class NashEvaluation:
    expected_value_player0: float
    best_response_value_player0: float
    best_response_value_player1: float
    nashconv: float
    exploitability: float


def _distribution(
    profile: StrategyProfile, info: InformationSet, legal: tuple[Action, ...]
) -> dict[Action, float]:
    supplied = profile.get(info)
    if supplied is None:
        return {action: 1.0 / len(legal) for action in legal}
    if set(supplied) != set(legal):
        raise ValueError(
            f"strategy has illegal or missing actions at {info.identifier}"
        )
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in supplied.values()
    ):
        raise ValueError("strategy probabilities must be finite numbers")
    if any(value < 0 for value in supplied.values()) or not math.isclose(
        sum(supplied.values()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("strategy probabilities must be nonnegative and sum to one")
    return dict(supplied)


def _value0(state: LeducState, profile: StrategyProfile) -> float:
    if state.is_terminal:
        return state.utility(0)
    if state.is_chance_node:
        return sum(
            probability * _value0(state.deal_public(card), profile)
            for card, probability in state.chance_outcomes
        )
    info = InformationSet.from_state(state)
    distribution = _distribution(profile, info, state.legal_actions)
    return sum(
        probability * _value0(state.act(action), profile)
        for action, probability in distribution.items()
    )


def expected_game_value(profile: StrategyProfile) -> float:
    """Exact expected net chips for Player 0 under a behavioral profile."""
    deals = LeducGame.deals()
    return sum(_value0(LeducGame.initial_state(deal), profile) for deal in deals) / len(
        deals
    )


def _counterfactual_nodes(
    profile: StrategyProfile, player: int
) -> dict[InformationSet, list[tuple[LeducState, float]]]:
    """Collect info-set histories weighted by chance and opponent reach only."""
    nodes: dict[InformationSet, list[tuple[LeducState, float]]] = defaultdict(list)

    def visit(state: LeducState, weight: float) -> None:
        if state.is_terminal:
            return
        if state.is_chance_node:
            for card, probability in state.chance_outcomes:
                visit(state.deal_public(card), weight * probability)
            return
        actor = state.current_player
        if actor is None:
            raise RuntimeError("nonterminal state has no acting player")
        if actor == player:
            nodes[InformationSet.from_state(state)].append((state, weight))
            # Exclude the responder's own reach, so traverse all own actions at unit reach.
            for action in state.legal_actions:
                visit(state.act(action), weight)
        else:
            info = InformationSet.from_state(state)
            distribution = _distribution(profile, info, state.legal_actions)
            for action, probability in distribution.items():
                visit(state.act(action), weight * probability)

    deals = LeducGame.deals()
    for deal in deals:
        visit(LeducGame.initial_state(deal), 1.0 / len(deals))
    return nodes


def best_response_value(profile: StrategyProfile, player: int) -> float:
    """Compute an exact pure BR, sharing one action across each information set."""
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    grouped = _counterfactual_nodes(profile, player)
    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            -(sum(map(len, item[0].histories))),
            item[0].identifier,
        ),
    )
    response: dict[InformationSet, Action] = {}
    sign = 1.0 if player == 0 else -1.0

    def continuation(state: LeducState) -> float:
        if state.is_terminal:
            return state.utility(0)
        if state.is_chance_node:
            return sum(
                prob * continuation(state.deal_public(card))
                for card, prob in state.chance_outcomes
            )
        actor = state.current_player
        if actor is None:
            raise RuntimeError("nonterminal state has no acting player")
        info = InformationSet.from_state(state)
        if actor == player:
            chosen = response.get(info)
            if chosen is None:
                raise RuntimeError(
                    "best-response information sets were not processed backwards"
                )
            return continuation(state.act(chosen))
        distribution = _distribution(profile, info, state.legal_actions)
        return sum(
            prob * continuation(state.act(action))
            for action, prob in distribution.items()
        )

    for info, histories in ordered:
        legal = histories[0][0].legal_actions
        action_values = {action: 0.0 for action in legal}
        for state, cf_weight in histories:
            for action in legal:
                action_values[action] += (
                    cf_weight * sign * continuation(state.act(action))
                )
        response[info] = max(legal, key=lambda action: action_values[action])

    br_profile: dict[InformationSet, Mapping[Action, float]] = {
        info: distribution
        for info, distribution in profile.items()
        if info.player != player
    }
    for info, action in response.items():
        br_profile[info] = {
            candidate: float(candidate is action)
            for candidate in grouped[info][0][0].legal_actions
        }
    value0 = expected_game_value(br_profile)
    return value0 if player == 0 else -value0


def evaluate_nash(profile: StrategyProfile) -> NashEvaluation:
    """NashConv sums unilateral gains; project convention is exploitability=NC/2."""
    value0 = expected_game_value(profile)
    br0 = best_response_value(profile, 0)
    br1 = best_response_value(profile, 1)
    nashconv = max(0.0, br0 - value0) + max(0.0, br1 + value0)
    return NashEvaluation(value0, br0, br1, nashconv, nashconv / 2)
