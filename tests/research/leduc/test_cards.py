import pytest

from pokerlab.research.leduc.cards import (
    DECK,
    Card,
    Rank,
    ordered_private_deals,
    remaining_cards,
)


def test_six_physical_cards_have_two_copies_of_each_rank():
    assert len(DECK) == 6
    assert [sum(card.rank is rank for card in DECK) for rank in Rank] == [2, 2, 2]
    assert len(set(DECK)) == 6
    assert len(ordered_private_deals()) == 30


def test_card_validation_and_remaining_cards():
    with pytest.raises(ValueError):
        Card(Rank.JACK, 2)
    with pytest.raises(TypeError):
        Card(9, 0)
    with pytest.raises(ValueError):
        remaining_cards(DECK[0], DECK[0])
    assert len(remaining_cards(DECK[0], DECK[1])) == 4
