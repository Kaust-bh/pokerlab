"""Physical cards and exact chance outcomes for two-player Leduc Poker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations


class Rank(IntEnum):
    JACK = 0
    QUEEN = 1
    KING = 2

    @property
    def label(self) -> str:
        return ("J", "Q", "K")[int(self)]


@dataclass(frozen=True, order=True, slots=True)
class Card:
    """A physical card; suit distinguishes the two copies of each rank."""

    rank: Rank
    suit: int

    def __post_init__(self) -> None:
        if not isinstance(self.rank, Rank):
            raise TypeError("rank must be a Rank")
        if self.suit not in (0, 1):
            raise ValueError("Leduc has exactly two suits, numbered 0 and 1")

    @property
    def label(self) -> str:
        return f"{self.rank.label}{self.suit}"


DECK: tuple[Card, ...] = tuple(Card(rank, suit) for rank in Rank for suit in range(2))


def ordered_private_deals() -> tuple[tuple[Card, Card], ...]:
    """All 30 equally likely ordered deals of two distinct physical cards."""
    return tuple((first, second) for first, second in combinations(DECK, 2)) + tuple(
        (second, first) for first, second in combinations(DECK, 2)
    )


def remaining_cards(*known: Card) -> tuple[Card, ...]:
    if len(set(known)) != len(known):
        raise ValueError("known cards must be physically distinct")
    if any(card not in DECK for card in known):
        raise ValueError("known card is not in the Leduc deck")
    return tuple(card for card in DECK if card not in known)
