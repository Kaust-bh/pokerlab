from collections import deque

import pytest

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cards import DECK
from pokerlab.research.leduc.game import LeducGame, LeducState


def test_round_order_public_card_and_card_removal():
    state = LeducGame.initial_state((DECK[0], DECK[2]))
    assert state.current_player == 0
    state = state.act(Action.CHECK).act(Action.CHECK)
    assert state.is_chance_node
    assert state.current_player is None
    assert len(state.chance_outcomes) == 4
    for card, probability in state.chance_outcomes:
        assert card not in state.private_cards
        assert probability == 0.25
        post = state.deal_public(card)
        assert post.current_player == 1
        assert post.public_card == card
    with pytest.raises(ValueError):
        state.deal_public(state.private_cards[0])


def test_fold_and_showdown_payoffs_are_net_and_zero_sum():
    state = LeducGame.initial_state((DECK[0], DECK[2]))
    folded = state.act(Action.BET).act(Action.FOLD)
    assert folded.contributions == (3, 1)
    assert folded.utilities() == (1.0, -1.0)

    state = LeducGame.initial_state((DECK[0], DECK[2]))
    state = state.act(Action.CHECK).act(Action.CHECK)
    board = next(card for card, _ in state.chance_outcomes if card.rank is DECK[0].rank)
    state = state.deal_public(board).act(Action.CHECK).act(Action.CHECK)
    assert state.is_terminal
    assert state.utilities() == (1.0, -1.0)


def test_raise_accounting_and_all_terminal_tree_payoffs():
    state = (
        LeducGame.initial_state((DECK[0], DECK[2])).act(Action.CHECK).act(Action.CHECK)
    )
    state = state.deal_public(state.chance_outcomes[0][0])
    state = state.act(Action.BET).act(Action.RAISE).act(Action.CALL)
    assert state.contributions == (9, 9)
    assert sum(state.utilities()) == 0

    terminals = 0
    for deal in LeducGame.deals():
        queue = deque([LeducGame.initial_state(deal)])
        while queue:
            node = queue.popleft()
            if node.is_terminal:
                assert sum(node.utilities()) == 0
                terminals += 1
            elif node.is_chance_node:
                queue.extend(node.deal_public(card) for card, _ in node.chance_outcomes)
            else:
                queue.extend(node.act(action) for action in node.legal_actions)
    assert terminals > 1_000


def test_illegal_action_and_invalid_state_are_rejected():
    state = LeducGame.initial_state((DECK[0], DECK[1]))
    with pytest.raises(ValueError):
        state.act(Action.CALL)
    with pytest.raises(ValueError):
        LeducState((DECK[0], DECK[0]))
    with pytest.raises(ValueError):
        LeducState((DECK[0], DECK[1]), contributions=(8, 1))
