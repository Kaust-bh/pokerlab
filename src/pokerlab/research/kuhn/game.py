"""Minimal standard two-player Kuhn Poker game tree."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import permutations


class KuhnCard(IntEnum):
    JACK = 0
    QUEEN = 1
    KING = 2

    @property
    def label(self) -> str:
        return {KuhnCard.JACK: "J", KuhnCard.QUEEN: "Q", KuhnCard.KING: "K"}[self]


class KuhnAction(StrEnum):
    CHECK = "check"
    BET = "bet"
    CALL = "call"
    FOLD = "fold"


TERMINAL_HISTORIES = frozenset(
    {
        (KuhnAction.CHECK, KuhnAction.CHECK),
        (KuhnAction.BET, KuhnAction.FOLD),
        (KuhnAction.BET, KuhnAction.CALL),
        (KuhnAction.CHECK, KuhnAction.BET, KuhnAction.FOLD),
        (KuhnAction.CHECK, KuhnAction.BET, KuhnAction.CALL),
    }
)


@dataclass(frozen=True, slots=True)
class KuhnState:
    """Full game-tree state; only the solver sees both private cards."""

    cards: tuple[KuhnCard, KuhnCard]
    history: tuple[KuhnAction, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cards, tuple):
            raise TypeError("cards must be an ordered two-card tuple")
        if len(self.cards) != 2 or self.cards[0] == self.cards[1]:
            raise ValueError("Kuhn Poker deals two distinct cards")
        if not all(isinstance(card, KuhnCard) for card in self.cards):
            raise ValueError("cards must be KuhnCard values")
        if not _valid_history(self.history):
            raise ValueError(f"invalid betting history: {self.history!r}")

    @property
    def is_terminal(self) -> bool:
        return self.history in TERMINAL_HISTORIES

    @property
    def current_player(self) -> int | None:
        return player_for_history(self.history)

    @property
    def legal_actions(self) -> tuple[KuhnAction, ...]:
        return legal_actions_for_history(self.history)

    def act(self, action: KuhnAction | str) -> KuhnState:
        try:
            resolved = KuhnAction(action)
        except ValueError as error:
            raise ValueError(f"unknown Kuhn action: {action!r}") from error
        if resolved not in self.legal_actions:
            raise ValueError(
                f"action {resolved.value!r} is illegal after history {self.history!r}"
            )
        return KuhnState(self.cards, (*self.history, resolved))

    def utilities(self) -> tuple[int, int]:
        """Return net-chip payoffs after 1-chip antes; utilities sum to zero."""
        if not self.is_terminal:
            raise ValueError("utilities are defined only for terminal states")
        if self.history == (KuhnAction.BET, KuhnAction.FOLD):
            player0 = 1
        elif self.history == (
            KuhnAction.CHECK,
            KuhnAction.BET,
            KuhnAction.FOLD,
        ):
            player0 = -1
        else:
            bet_was_called = self.history in {
                (KuhnAction.BET, KuhnAction.CALL),
                (KuhnAction.CHECK, KuhnAction.BET, KuhnAction.CALL),
            }
            winner_utility = 2 if bet_was_called else 1
            player0 = (
                winner_utility if self.cards[0] > self.cards[1] else -winner_utility
            )
        return player0, -player0

    def utility(self, player: int) -> int:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return self.utilities()[player]


def _valid_history(history: tuple[KuhnAction, ...]) -> bool:
    if not isinstance(history, tuple) or not all(
        isinstance(action, KuhnAction) for action in history
    ):
        return False
    if history in TERMINAL_HISTORIES:
        return True
    return history in {
        (),
        (KuhnAction.CHECK,),
        (KuhnAction.BET,),
        (KuhnAction.CHECK, KuhnAction.BET),
    }


def legal_actions_for_history(
    history: tuple[KuhnAction, ...],
) -> tuple[KuhnAction, ...]:
    if not _valid_history(history):
        raise ValueError(f"invalid betting history: {history!r}")
    return {
        (): (KuhnAction.CHECK, KuhnAction.BET),
        (KuhnAction.CHECK,): (KuhnAction.CHECK, KuhnAction.BET),
        (KuhnAction.BET,): (KuhnAction.FOLD, KuhnAction.CALL),
        (KuhnAction.CHECK, KuhnAction.BET): (KuhnAction.FOLD, KuhnAction.CALL),
    }.get(history, ())


def player_for_history(history: tuple[KuhnAction, ...]) -> int | None:
    if not _valid_history(history) or history in TERMINAL_HISTORIES:
        return None
    return {
        (): 0,
        (KuhnAction.CHECK,): 1,
        (KuhnAction.BET,): 1,
        (KuhnAction.CHECK, KuhnAction.BET): 0,
    }[history]


class KuhnGame:
    """Chance-deal helper for the six equally likely ordered card deals."""

    @staticmethod
    def deals() -> Iterator[tuple[KuhnCard, KuhnCard]]:
        return permutations(tuple(KuhnCard), 2)

    @staticmethod
    def initial_state(cards: tuple[KuhnCard, KuhnCard]) -> KuhnState:
        return KuhnState(cards)
