"""Leduc hand ranking: a matching rank makes a pair; otherwise high card."""

from pokerlab.research.leduc.cards import Card


def hand_rank(private_card: Card, public_card: Card) -> tuple[int, int]:
    """Return a comparable rank tuple; pair category outranks every high card."""
    if private_card.rank is public_card.rank:
        return (1, int(private_card.rank))
    return (0, int(private_card.rank))


def compare_hands(first: Card, second: Card, public_card: Card) -> int:
    """Return 1 if first wins, -1 if second wins, and 0 for a split pot."""
    first_rank = hand_rank(first, public_card)
    second_rank = hand_rank(second, public_card)
    return (first_rank > second_rank) - (first_rank < second_rank)
