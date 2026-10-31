import pytest

from pokerlab.research.kuhn.game import (
    TERMINAL_HISTORIES,
    KuhnGame,
    KuhnState,
)
from pokerlab.research.kuhn.game import (
    KuhnAction as A,
)
from pokerlab.research.kuhn.game import (
    KuhnCard as C,
)


def play(cards: tuple[C, C], history: tuple[A, ...]) -> KuhnState:
    state = KuhnState(cards)
    for action in history:
        state = state.act(action)
    return state


def test_chance_deals_are_six_ordered_unique_deals() -> None:
    deals = tuple(KuhnGame.deals())
    assert len(deals) == 6
    assert len(set(deals)) == 6
    assert all(first != second for first, second in deals)


def test_nonterminal_turns_and_legal_actions() -> None:
    state = KuhnState((C.JACK, C.KING))
    assert state.current_player == 0
    assert state.legal_actions == (A.CHECK, A.BET)
    state = state.act(A.CHECK)
    assert state.current_player == 1
    assert state.legal_actions == (A.CHECK, A.BET)
    state = state.act(A.BET)
    assert state.current_player == 0
    assert state.legal_actions == (A.FOLD, A.CALL)


@pytest.mark.parametrize("history", sorted(TERMINAL_HISTORIES, key=lambda h: len(h)))
def test_every_terminal_history_has_zero_sum_payoff(history: tuple[A, ...]) -> None:
    for cards in KuhnGame.deals():
        state = play(cards, history)
        assert state.is_terminal
        assert sum(state.utilities()) == 0
        with pytest.raises(ValueError, match="illegal"):
            state.act(A.CHECK)


def test_manual_terminal_payoffs_after_antes() -> None:
    assert play((C.KING, C.JACK), (A.CHECK, A.CHECK)).utilities() == (1, -1)
    assert play((C.JACK, C.KING), (A.CHECK, A.CHECK)).utilities() == (-1, 1)
    assert play((C.JACK, C.KING), (A.BET, A.FOLD)).utilities() == (1, -1)
    assert play((C.JACK, C.KING), (A.CHECK, A.BET, A.FOLD)).utilities() == (-1, 1)
    assert play((C.KING, C.JACK), (A.BET, A.CALL)).utilities() == (2, -2)
    assert play((C.JACK, C.KING), (A.CHECK, A.BET, A.CALL)).utilities() == (-2, 2)


def test_invalid_history_and_action_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid betting history"):
        KuhnState((C.JACK, C.KING), (A.CALL,))
    with pytest.raises(ValueError, match="distinct"):
        KuhnState((C.QUEEN, C.QUEEN))
    with pytest.raises(ValueError, match="illegal"):
        KuhnState((C.JACK, C.KING)).act(A.CALL)
