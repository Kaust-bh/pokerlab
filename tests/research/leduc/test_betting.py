import pytest

from pokerlab.research.leduc.betting import Action, legal_actions, validate_history


def test_opening_check_and_bet_transitions():
    assert legal_actions(()) == (Action.CHECK, Action.BET)
    assert legal_actions((Action.CHECK,)) == (Action.CHECK, Action.BET)
    assert legal_actions((Action.BET,)) == (Action.FOLD, Action.CALL, Action.RAISE)
    assert legal_actions((Action.CHECK, Action.BET)) == (
        Action.FOLD,
        Action.CALL,
        Action.RAISE,
    )


def test_two_bet_cap_and_round_termination():
    history = (Action.BET, Action.RAISE)
    assert legal_actions(history) == (Action.FOLD, Action.CALL)
    assert legal_actions((Action.CHECK, Action.CHECK)) == ()
    assert legal_actions((*history, Action.CALL)) == ()
    assert validate_history(history)
    assert not validate_history((Action.BET, Action.RAISE, Action.RAISE))
    assert not validate_history((Action.BET, Action.CHECK))
    assert not validate_history((Action.CHECK, Action.CHECK, Action.BET))


def test_invalid_history_raises():
    assert legal_actions((Action.BET, Action.FOLD)) == ()
    with pytest.raises(ValueError):
        legal_actions((Action.BET, Action.CHECK))
