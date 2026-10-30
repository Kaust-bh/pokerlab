"""Standard 52-card deck with reproducible shuffling."""

import random
from collections.abc import Sequence

from pokerlab.engine.cards import SUITS, Card


class Deck:
    """A shuffled deck; draw order is from the end of its internal list."""

    def __init__(
        self, seed: int | None = None, *, rng: random.Random | None = None
    ) -> None:
        if seed is not None and rng is not None:
            raise ValueError("provide either seed or rng, not both")
        self._rng = rng if rng is not None else random.Random(seed)
        self._cards = [Card(rank, suit) for suit in SUITS for rank in range(2, 15)]
        self.shuffle()

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def draw(self, count: int = 1) -> Card | list[Card]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        if count > len(self._cards):
            raise ValueError("not enough cards remain in the deck")
        if count == 1:
            return self._cards.pop()
        return [self._cards.pop() for _ in range(count)]

    @property
    def remaining(self) -> int:
        return len(self._cards)

    @property
    def cards(self) -> Sequence[Card]:
        """A read-only snapshot of cards that have not been drawn."""
        return tuple(self._cards)
