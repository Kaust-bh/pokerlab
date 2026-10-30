import pytest

from pokerlab.engine import Card, Deck


def test_card_is_immutable_and_hashable() -> None:
    card = Card(14, "SPADES")
    assert str(card) == "A of spades"
    assert {card, Card(14, "spades")} == {card}
    with pytest.raises((AttributeError, TypeError)):
        card.rank = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    "rank,suit", [(1, "spades"), (15, "clubs"), (2, "x"), (True, "clubs")]
)
def test_invalid_cards(rank: int, suit: str) -> None:
    with pytest.raises(ValueError):
        Card(rank, suit)


def test_deck_unique_draws_exhaustion_and_reproducibility() -> None:
    a, b = Deck(seed=9), Deck(seed=9)
    assert a.cards == b.cards
    drawn = a.draw(52)
    assert len(set(drawn)) == 52
    assert a.remaining == 0
    with pytest.raises(ValueError):
        a.draw()
