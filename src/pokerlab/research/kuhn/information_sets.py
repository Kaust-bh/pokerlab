"""Player-observable state abstraction for Kuhn Poker."""

from __future__ import annotations

from dataclasses import dataclass

from pokerlab.research.kuhn.game import (
    KuhnAction,
    KuhnCard,
    KuhnState,
    player_for_history,
)


@dataclass(frozen=True, slots=True)
class InformationSet:
    """An acting player's private card, identity, and public action history.

    A game-tree history includes the full deal, including both private cards.
    An information set deliberately omits the opponent's private card and groups
    all full histories that the acting player cannot distinguish.
    """

    player: int
    private_card: KuhnCard
    history: tuple[KuhnAction, ...]

    def __post_init__(self) -> None:
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if not isinstance(self.private_card, KuhnCard):
            raise TypeError("private_card must be a KuhnCard")
        if player_for_history(self.history) != self.player:
            raise ValueError(
                "history is invalid or belongs to a different acting player"
            )

    @property
    def identifier(self) -> str:
        history_id = "/".join(action.value for action in self.history) or "root"
        return f"p{self.player}|{self.private_card.label}|{history_id}"

    @classmethod
    def from_state(cls, state: KuhnState) -> InformationSet:
        player = state.current_player
        if player is None:
            raise ValueError("terminal states have no acting-player information set")
        return cls(player, state.cards[player], state.history)
