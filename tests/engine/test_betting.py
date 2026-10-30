import pytest

from pokerlab.engine import ActionType, BettingRound


def test_call_check_and_out_of_turn_are_checked_atomically() -> None:
    round_ = BettingRound(
        {"a": 100, "b": 100},
        big_blind=10,
        start_player="a",
        contributions={"a": 5, "b": 10},
    )
    with pytest.raises(ValueError):
        round_.act("b", ActionType.CALL)
    before = (dict(round_.stacks), dict(round_.contributions), round_.current_player)
    with pytest.raises(ValueError):
        round_.act("a", ActionType.CHECK)
    assert before == (round_.stacks, round_.contributions, round_.current_player)
    round_.act("a", ActionType.CALL)
    round_.act("b", ActionType.CHECK)
    assert round_.complete


def test_full_raise_and_short_all_in_do_not_reopen_prior_action() -> None:
    round_ = BettingRound({"a": 100, "b": 100, "c": 15}, big_blind=10, start_player="a")
    round_.act("a", ActionType.BET, 10)
    round_.act("b", ActionType.CALL)
    round_.act("c", ActionType.ALL_IN)
    assert round_.current_bet == 15
    assert round_.current_player == "a"
    with pytest.raises(ValueError, match="reopened"):
        round_.act("a", ActionType.RAISE, 30)
    with pytest.raises(ValueError, match="reopened"):
        round_.act("a", ActionType.ALL_IN)
    round_.act("a", ActionType.CALL)
    round_.act("b", ActionType.CALL)
    assert round_.complete


def test_short_all_in_and_minimum_raise_validation() -> None:
    round_ = BettingRound({"a": 100, "b": 100}, big_blind=10, start_player="a")
    with pytest.raises(ValueError):
        round_.act("a", ActionType.BET, 5)
    round_.act("a", ActionType.BET, 10)
    with pytest.raises(ValueError):
        round_.act("b", ActionType.RAISE, 15)
