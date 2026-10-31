"""Immutable full states for the standalone fixed-limit Leduc Poker game."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from pokerlab.research.leduc.betting import (
    BET_SIZES,
    Action,
    is_closed,
    legal_actions,
    next_player,
    validate_history,
)
from pokerlab.research.leduc.cards import Card, ordered_private_deals, remaining_cards
from pokerlab.research.leduc.hand_evaluator import compare_hands


@dataclass(frozen=True, slots=True)
class LeducState:
    """Complete state, including hidden cards, intended for game-tree algorithms."""

    private_cards: tuple[Card, Card]
    public_card: Card | None = None
    round_index: int = 0
    histories: tuple[tuple[Action, ...], tuple[Action, ...]] = ((), ())
    contributions: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        if not isinstance(self.private_cards, tuple) or len(self.private_cards) != 2:
            raise ValueError("private_cards must be an ordered two-card tuple")
        if self.private_cards[0] == self.private_cards[1]:
            raise ValueError("private cards must be physically distinct")
        if self.public_card is not None and self.public_card in self.private_cards:
            raise ValueError("public card must be distinct from both private cards")
        remaining_cards(
            *self.private_cards,
            *(() if self.public_card is None else (self.public_card,)),
        )
        if self.round_index not in (0, 1):
            raise ValueError("round_index must be 0 (pre-flop) or 1 (post-flop)")
        if not isinstance(self.histories, tuple) or len(self.histories) != 2:
            raise ValueError("histories must contain one action tuple per round")
        if not all(validate_history(history) for history in self.histories):
            raise ValueError("invalid betting history")
        if not isinstance(self.contributions, tuple) or len(self.contributions) != 2:
            raise ValueError("contributions must contain one total per player")
        if any(not isinstance(value, int) or value < 1 for value in self.contributions):
            raise ValueError(
                "contributions must be integer chip totals including antes"
            )
        expected = [1, 1]
        for round_index, history in enumerate(self.histories):
            for action_index, action in enumerate(history):
                actor = round_index ^ (action_index % 2)
                if action in (Action.BET, Action.CALL):
                    expected[actor] += BET_SIZES[round_index]
                elif action is Action.RAISE:
                    expected[actor] += 2 * BET_SIZES[round_index]
        if tuple(expected) != self.contributions:
            raise ValueError("contributions do not match the betting histories")
        if self.round_index == 0 and (
            self.public_card is not None or self.histories[1]
        ):
            raise ValueError(
                "pre-flop states cannot have a public card or later history"
            )
        first_history = self.histories[0]
        first_was_folded = bool(first_history) and first_history[-1] is Action.FOLD
        if self.round_index == 0 and is_closed(first_history) and not first_was_folded:
            raise ValueError(
                "a closed first round must advance to the public-card node"
            )
        if self.round_index == 1 and not is_closed(self.histories[0]):
            raise ValueError("post-flop states require a closed pre-flop round")
        if self.round_index == 1 and first_was_folded:
            raise ValueError("a fold in round one ends the hand before the board")
        if (
            self.public_card is None
            and self.round_index == 1
            and (self.histories[1] or self.is_terminal)
        ):
            raise ValueError("the public-card chance state has no post-flop actions")

    @property
    def round_history(self) -> tuple[Action, ...]:
        return self.histories[self.round_index]

    @property
    def is_chance_node(self) -> bool:
        return self.round_index == 1 and self.public_card is None

    @property
    def is_terminal(self) -> bool:
        return any(
            history and history[-1] is Action.FOLD for history in self.histories
        ) or (
            self.round_index == 1
            and self.public_card is not None
            and is_closed(self.histories[1])
        )

    @property
    def current_player(self) -> int | None:
        if self.is_terminal or self.is_chance_node:
            return None
        return next_player(self.round_history, self.round_index)

    @property
    def legal_actions(self) -> tuple[Action, ...]:
        if self.is_terminal or self.is_chance_node:
            return ()
        return legal_actions(self.round_history)

    @property
    def chance_outcomes(self) -> tuple[tuple[Card, float], ...]:
        if not self.is_chance_node:
            return ()
        cards = remaining_cards(*self.private_cards)
        probability = 1.0 / len(cards)
        return tuple((card, probability) for card in cards)

    @cache  # noqa: B019 - the immutable exhaustive Leduc state space is finite
    def deal_public(self, card: Card) -> LeducState:
        if not self.is_chance_node:
            raise ValueError("public card can only be dealt at the chance node")
        if card not in remaining_cards(*self.private_cards):
            raise ValueError("public card is unavailable")
        return LeducState(
            self.private_cards,
            card,
            1,
            self.histories,
            self.contributions,
        )

    @cache  # noqa: B019 - the immutable exhaustive Leduc state space is finite
    def act(self, action: Action | str) -> LeducState:
        try:
            resolved = Action(action)
        except ValueError as error:
            raise ValueError(f"unknown Leduc action: {action!r}") from error
        if resolved not in self.legal_actions:
            raise ValueError(
                f"action {resolved.value!r} is illegal after {self.round_history!r}"
            )
        actor = self.current_player
        if actor is None:
            raise RuntimeError("nonterminal betting state has no acting player")
        contributions = list(self.contributions)
        if resolved in (Action.BET, Action.CALL):
            contributions[actor] += BET_SIZES[self.round_index]
        elif resolved is Action.RAISE:
            # A raise first calls the outstanding wager, then adds one raise unit.
            contributions[actor] += 2 * BET_SIZES[self.round_index]
        updated = list(self.histories)
        updated[self.round_index] = (*self.round_history, resolved)
        new_history = (updated[0], updated[1])
        if (
            self.round_index == 0
            and is_closed(new_history[0])
            and not (new_history[0][-1] is Action.FOLD)
        ):
            return LeducState(
                self.private_cards,
                None,
                1,
                new_history,
                (contributions[0], contributions[1]),
            )
        return LeducState(
            self.private_cards,
            self.public_card,
            self.round_index,
            new_history,
            (contributions[0], contributions[1]),
        )

    @property
    def pot(self) -> int:
        return sum(self.contributions)

    def utilities(self) -> tuple[float, float]:
        """Net-chip zero-sum utilities after antes, bets, calls, and raises."""
        if not self.is_terminal:
            raise ValueError("utilities are defined only for terminal states")
        folded_round = next(
            (
                round_index
                for round_index, history in enumerate(self.histories)
                if history and history[-1] is Action.FOLD
            ),
            None,
        )
        if folded_round is not None:
            fold_history = self.histories[folded_round]
            folded = folded_round ^ ((len(fold_history) - 1) % 2)
            winner = 1 - folded
            value = float(self.contributions[folded])
            result = [0.0, 0.0]
            result[winner] = value
            result[folded] = -value
            return result[0], result[1]
        if self.public_card is None:
            raise RuntimeError("showdown requires a public card")
        comparison = compare_hands(
            self.private_cards[0], self.private_cards[1], self.public_card
        )
        if comparison == 0:
            return 0.0, 0.0
        winner = 0 if comparison > 0 else 1
        loser = 1 - winner
        result = [0.0, 0.0]
        result[winner] = float(self.contributions[loser])
        result[loser] = -float(self.contributions[winner])
        if sum(result) != 0:
            raise RuntimeError("showdown contributions must balance")
        return result[0], result[1]

    def utility(self, player: int) -> float:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return self.utilities()[player]


class LeducGame:
    """Exact chance enumeration helpers for the selected six-card game."""

    @staticmethod
    def deals() -> tuple[tuple[Card, Card], ...]:
        return ordered_private_deals()

    @staticmethod
    def initial_state(cards: tuple[Card, Card]) -> LeducState:
        return LeducState(cards)
