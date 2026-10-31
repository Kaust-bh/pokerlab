"""Player-observable states; opponent cards and undealt cards are omitted."""

from __future__ import annotations

from dataclasses import dataclass

from pokerlab.research.leduc.betting import (
    Action,
    is_closed,
    legal_actions,
    validate_history,
)
from pokerlab.research.leduc.cards import Card
from pokerlab.research.leduc.game import LeducState


@dataclass(frozen=True, slots=True)
class InformationSet:
    player: int
    private_card: Card
    public_card: Card | None
    round_index: int
    histories: tuple[tuple[Action, ...], tuple[Action, ...]]

    def __post_init__(self) -> None:
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if self.round_index not in (0, 1):
            raise ValueError("round_index must be 0 or 1")
        if (self.round_index == 0) != (self.public_card is None):
            raise ValueError("the public card is visible exactly in round two")
        if self.public_card == self.private_card:
            raise ValueError("the public and private cards must be distinct")
        if len(self.histories) != 2 or not all(
            validate_history(history) for history in self.histories
        ):
            raise ValueError(
                "information-set histories must be valid betting histories"
            )
        current = self.histories[self.round_index]
        if not legal_actions(current):
            raise ValueError("terminal observations have no information set")
        first = self.histories[0]
        folded = bool(first) and first[-1] is Action.FOLD
        if self.round_index == 0:
            if self.histories[1] or (is_closed(first) and not folded):
                raise ValueError("invalid first-round information state")
        elif not is_closed(first) or folded:
            raise ValueError("round-two information requires a non-folded first round")
        if self.player != (self.round_index ^ (len(current) % 2)):
            raise ValueError("player does not match the acting seat in this history")

    @property
    def identifier(self) -> str:
        card = self.private_card.label
        board = "-" if self.public_card is None else self.public_card.label
        first = "/".join(action.value for action in self.histories[0]) or "root"
        second = "/".join(action.value for action in self.histories[1]) or "-"
        return f"p{self.player}|{card}|{board}|r{self.round_index}|{first}|{second}"

    @classmethod
    def from_state(cls, state: LeducState) -> InformationSet:
        player = state.current_player
        if player is None:
            raise ValueError(
                "chance and terminal states have no acting information set"
            )
        return cls(
            player,
            state.private_cards[player],
            state.public_card,
            state.round_index,
            state.histories,
        )
