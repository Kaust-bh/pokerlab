"""Fixed-limit Leduc betting rules for the selected two-raise-cap variant."""

from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    CHECK = "check"
    BET = "bet"
    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"


# Southey et al.'s two-bet maximum is one opening bet plus one raise.
MAX_RAISES = 1
BET_SIZES = (2, 4)


def is_folded(history: tuple[Action, ...]) -> bool:
    return bool(history) and history[-1] is Action.FOLD


def is_closed(history: tuple[Action, ...]) -> bool:
    if not history or is_folded(history):
        return is_folded(history)
    if history[-1] is Action.CALL:
        return True
    return history == (Action.CHECK, Action.CHECK)


def legal_actions(history: tuple[Action, ...]) -> tuple[Action, ...]:
    """Legal actions after a valid within-round public history."""
    if not isinstance(history, tuple) or not all(
        isinstance(a, Action) for a in history
    ):
        raise ValueError("history must be a tuple of Leduc actions")
    if is_closed(history):
        return ()
    if history == ():
        return (Action.CHECK, Action.BET)
    if history == (Action.CHECK,):
        return (Action.CHECK, Action.BET)
    if history[-1] is Action.BET or history[-1] is Action.RAISE:
        raises = history.count(Action.RAISE)
        if raises < MAX_RAISES:
            return (Action.FOLD, Action.CALL, Action.RAISE)
        return (Action.FOLD, Action.CALL)
    raise ValueError(f"invalid betting history: {history!r}")


def next_player(history: tuple[Action, ...], first_player: int) -> int | None:
    if first_player not in (0, 1):
        raise ValueError("first_player must be 0 or 1")
    if not legal_actions(history):
        return None
    return first_player ^ (len(history) % 2)


def validate_history(history: tuple[Action, ...]) -> bool:
    """Check a complete legal betting sequence, not merely its last action."""
    if not isinstance(history, tuple):
        return False
    prefix: tuple[Action, ...] = ()
    try:
        for action in history:
            if action not in legal_actions(prefix):
                return False
            prefix += (action,)
    except (TypeError, ValueError):
        return False
    return True
