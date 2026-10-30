"""Five- and seven-card Texas Hold'em hand evaluation."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations

from pokerlab.engine.cards import Card


class HandCategory(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


@dataclass(frozen=True, slots=True, eq=False)
class HandRank:
    """Comparable rank; lexicographically larger values represent stronger hands."""

    category: HandCategory
    tiebreak: tuple[int, ...]
    best_five: tuple[Card, ...]

    @property
    def value(self) -> tuple[int, ...]:
        return (int(self.category), *self.tiebreak)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.value < other.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


def _straight_high(ranks: Iterable[int]) -> int | None:
    values = set(ranks)
    if 14 in values:
        values.add(1)
    rank_mask = sum(1 << rank for rank in values)
    for high in range(14, 4, -1):
        straight_mask = 31 << (high - 4)
        if rank_mask & straight_mask == straight_mask:
            return high
    return None


def _rank_five(cards: tuple[Card, ...]) -> HandRank:
    counts = Counter(card.rank for card in cards)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    ranks = sorted(counts, reverse=True)
    flush = len({card.suit for card in cards}) == 1
    straight = _straight_high(ranks)
    if flush and straight:
        category = (
            HandCategory.ROYAL_FLUSH
            if straight == 14 and min(ranks) == 10
            else HandCategory.STRAIGHT_FLUSH
        )
        tie = (straight,)
    elif groups[0][0] == 4:
        category, tie = HandCategory.FOUR_OF_A_KIND, (groups[0][1], groups[1][1])
    elif groups[0][0] == 3 and groups[1][0] == 2:
        category, tie = HandCategory.FULL_HOUSE, (groups[0][1], groups[1][1])
    elif flush:
        category, tie = HandCategory.FLUSH, tuple(ranks)
    elif straight:
        category, tie = HandCategory.STRAIGHT, (straight,)
    elif groups[0][0] == 3:
        trips = groups[0][1]
        category, tie = (
            HandCategory.THREE_OF_A_KIND,
            (trips, *sorted((r for r in ranks if r != trips), reverse=True)),
        )
    elif groups[0][0] == 2 and groups[1][0] == 2:
        pairs = sorted((groups[0][1], groups[1][1]), reverse=True)
        kicker = next(rank for rank in ranks if rank not in pairs)
        category, tie = HandCategory.TWO_PAIR, (*pairs, kicker)
    elif groups[0][0] == 2:
        pair = groups[0][1]
        category, tie = (
            HandCategory.ONE_PAIR,
            (pair, *sorted((r for r in ranks if r != pair), reverse=True)),
        )
    else:
        category, tie = HandCategory.HIGH_CARD, tuple(ranks)
    return HandRank(category, tie, cards)


def evaluate_hand(cards: Iterable[Card]) -> HandRank:
    """Return the best five-card rank from exactly five, six, or seven cards."""
    hand = tuple(cards)
    if len(hand) not in (5, 6, 7):
        raise ValueError("evaluate_hand requires five, six, or seven cards")
    if len(set(hand)) != len(hand):
        raise ValueError("a hand cannot contain duplicate cards")
    return max(
        (_rank_five(combo) for combo in combinations(hand, 5)),
        key=lambda rank: rank.value,
    )
