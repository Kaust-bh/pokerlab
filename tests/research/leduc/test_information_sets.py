import pytest

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cards import DECK
from pokerlab.research.leduc.game import LeducGame
from pokerlab.research.leduc.information_sets import InformationSet


def test_hidden_opponent_card_is_omitted_and_equivalent_states_group():
    first = LeducGame.initial_state((DECK[0], DECK[2])).act(Action.CHECK)
    second = LeducGame.initial_state((DECK[4], DECK[2])).act(Action.CHECK)
    assert InformationSet.from_state(first) == InformationSet.from_state(second)
    assert DECK[4].label not in InformationSet.from_state(first).identifier


def test_private_card_public_card_and_history_distinguish_observations():
    first = LeducGame.initial_state((DECK[0], DECK[2])).act(Action.CHECK)
    other_card = LeducGame.initial_state((DECK[0], DECK[3])).act(Action.CHECK)
    bet_history = LeducGame.initial_state((DECK[0], DECK[2])).act(Action.BET)
    assert InformationSet.from_state(first) != InformationSet.from_state(other_card)
    assert InformationSet.from_state(first) != InformationSet.from_state(bet_history)

    chance = first.act(Action.CHECK)
    public_a = chance.deal_public(chance.chance_outcomes[0][0]).act(Action.CHECK)
    public_b = chance.deal_public(chance.chance_outcomes[1][0]).act(Action.CHECK)
    assert InformationSet.from_state(public_a) != InformationSet.from_state(public_b)


def test_terminal_and_chance_states_have_no_information_set():
    state = (
        LeducGame.initial_state((DECK[0], DECK[2])).act(Action.CHECK).act(Action.CHECK)
    )
    with pytest.raises(ValueError):
        InformationSet.from_state(state)
