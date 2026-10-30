"""Transparent baseline poker policy based on hand class and pot odds."""

from dataclasses import dataclass

from pokerlab.agents.base import Agent, Observation
from pokerlab.engine import Action, ActionType, HandCategory, evaluate_hand


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    call_margin: float = 0.08
    value_bet_strength: float = 0.72
    strong_raise_strength: float = 0.86
    position_bonus: float = 0.04


class HeuristicAgent(Agent):
    """Uses approximate strength bands; these are not calibrated equity estimates."""

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()

    @staticmethod
    def _preflop_strength(observation: Observation) -> float:
        first, second = observation.hole_cards
        high, low = sorted((first.rank, second.rank), reverse=True)
        if high == low:
            return 0.63 + (high - 2) * 0.025
        strength = 0.20 + (high - 2) * 0.022 + (low - 2) * 0.016
        if first.suit == second.suit:
            strength += 0.07
        if high - low == 1:
            strength += 0.04
        if high >= 11 and low >= 10:
            strength += 0.04
        return min(strength, 0.82)

    @staticmethod
    def _postflop_strength(observation: Observation) -> tuple[float, bool]:
        cards = (*observation.hole_cards, *observation.community_cards)
        rank = evaluate_hand(cards)
        bands = {
            HandCategory.HIGH_CARD: 0.28,
            HandCategory.ONE_PAIR: 0.48,
            HandCategory.TWO_PAIR: 0.68,
            HandCategory.THREE_OF_A_KIND: 0.82,
            HandCategory.STRAIGHT: 0.91,
            HandCategory.FLUSH: 0.94,
            HandCategory.FULL_HOUSE: 0.98,
            HandCategory.FOUR_OF_A_KIND: 0.995,
            HandCategory.STRAIGHT_FLUSH: 0.999,
            HandCategory.ROYAL_FLUSH: 1.0,
        }
        suits = {}
        for card in cards:
            suits[card.suit] = suits.get(card.suit, 0) + 1
        flush_draw = len(observation.community_cards) < 5 and max(suits.values()) == 4
        ranks = {1 if card.rank == 14 else card.rank for card in cards}
        straight_draw = False
        if len(observation.community_cards) < 5:
            for high in range(5, 15):
                needed = set(range(high - 4, high + 1))
                if high == 5:
                    needed = {14, 2, 3, 4, 5}
                if len(needed & ranks) == 4:
                    straight_draw = True
                    break
        return min(
            1.0,
            bands[rank.category]
            + (0.12 if flush_draw else 0.0)
            + (0.06 if straight_draw else 0.0),
        ), flush_draw or straight_draw

    def strength(self, observation: Observation) -> tuple[float, bool]:
        if not observation.community_cards:
            return self._preflop_strength(observation), False
        return self._postflop_strength(observation)

    def act(self, observation: Observation) -> Action:
        actions = observation.legal_actions
        if not actions:
            raise ValueError("no legal actions are available")
        strength, has_draw = self.strength(observation)
        position_adjustment = (
            self.config.position_bonus if observation.in_position else 0.0
        )
        decision_strength = min(1.0, strength + position_adjustment)
        find = lambda kind: next(
            (action for action in actions if action.type == kind), None
        )
        if observation.to_call:
            call_cost = observation.to_call / max(1, observation.effective_stack)
            required = observation.pot_odds + self.config.call_margin
            if decision_strength >= self.config.strong_raise_strength:
                raise_action = find(ActionType.RAISE)
                if raise_action is not None and call_cost < 0.55:
                    return raise_action
            if decision_strength >= required or (
                has_draw and decision_strength + 0.04 >= required
            ):
                call = find(ActionType.CALL)
                if call is not None:
                    return call
            fold = find(ActionType.FOLD)
            if fold is not None:
                return fold
            return find(ActionType.CALL) or find(ActionType.CHECK) or actions[0]

        check = find(ActionType.CHECK)
        if decision_strength >= self.config.value_bet_strength:
            bet = find(ActionType.BET)
            if bet is not None:
                return bet
        return check or actions[0]
