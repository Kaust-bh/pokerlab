"""Immutable playing cards used throughout the engine."""

from dataclasses import dataclass

SUITS = ("clubs", "diamonds", "hearts", "spades")
RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}


@dataclass(frozen=True, slots=True, order=True)
class Card:
    """A standard card, with ranks 2..14 and a lowercase suit name."""

    rank: int
    suit: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 2 <= self.rank <= 14
        ):
            raise ValueError("rank must be an integer from 2 through 14")
        if not isinstance(self.suit, str):
            raise TypeError("suit must be a string")
        normalized = self.suit.lower()
        if normalized not in SUITS:
            raise ValueError(f"suit must be one of {', '.join(SUITS)}")
        object.__setattr__(self, "suit", normalized)

    def __str__(self) -> str:
        rank = RANK_NAMES.get(self.rank, str(self.rank))
        return f"{rank} of {self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank}, {self.suit!r})"
