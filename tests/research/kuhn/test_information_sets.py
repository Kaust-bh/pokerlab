from pokerlab.research.kuhn.game import KuhnAction as A
from pokerlab.research.kuhn.game import KuhnCard as C
from pokerlab.research.kuhn.game import KuhnState
from pokerlab.research.kuhn.information_sets import InformationSet


def test_hidden_opponent_card_does_not_change_information_set() -> None:
    first = InformationSet.from_state(KuhnState((C.JACK, C.QUEEN), (A.CHECK,)))
    second = InformationSet.from_state(KuhnState((C.KING, C.QUEEN), (A.CHECK,)))
    assert first == second
    assert first.identifier == "p1|Q|check"


def test_public_history_and_acting_player_distinguish_information_sets() -> None:
    root = InformationSet.from_state(KuhnState((C.KING, C.JACK)))
    after_check_bet = InformationSet.from_state(
        KuhnState((C.KING, C.JACK), (A.CHECK, A.BET))
    )
    assert root.player == after_check_bet.player == 0
    assert root.private_card == after_check_bet.private_card == C.KING
    assert root != after_check_bet
    assert root.identifier != after_check_bet.identifier


def test_terminal_state_has_no_information_set() -> None:
    state = KuhnState((C.KING, C.JACK), (A.BET, A.FOLD))
    try:
        InformationSet.from_state(state)
    except ValueError as error:
        assert "terminal" in str(error)
    else:
        raise AssertionError("terminal state unexpectedly had an information set")
