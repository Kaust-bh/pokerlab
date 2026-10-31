"""Abstract action menu and conversion to legal engine contribution targets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pokerlab.agents.action_profiles import ACTION_PROFILES
from pokerlab.engine import Action, ActionType, GameState


class AbstractAction(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET_HALF_POT = "bet_half_pot"
    BET_MIN = "bet_min"
    BET_THIRD_POT = "bet_third_pot"
    BET_THREE_QUARTER_POT = "bet_three_quarter_pot"
    BET_POT = "bet_pot"
    BET_ONE_HALF_POT = "bet_one_half_pot"
    BET_TWO_POT = "bet_two_pot"
    RAISE_MIN = "raise_min"
    RAISE_THIRD_POT = "raise_third_pot"
    RAISE_HALF_POT = "raise_half_pot"
    RAISE_THREE_QUARTER_POT = "raise_three_quarter_pot"
    RAISE_POT = "raise_pot"
    RAISE_ONE_HALF_POT = "raise_one_half_pot"
    RAISE_TWO_POT = "raise_two_pot"
    ALL_IN = "all_in"


@dataclass(frozen=True, slots=True)
class ActionSpec:
    abstract: AbstractAction
    engine: Action


def abstract_action_specs(
    game: GameState, player: str, action_profile: str = "legacy"
) -> tuple[ActionSpec, ...]:
    """Return distinct abstract actions mapped to engine-legal actions.

    Bet increments use current pot size. Raise increments use the pot after
    calling, with the target expressed as the opponent's current street
    contribution plus that increment. The engine receives target street totals.
    """
    round_ = game.betting
    if game.complete or player != round_.current_player or round_.complete:
        return ()
    return _action_specs(
        contribution=round_.contributions[player],
        opponent_contribution=round_.contributions[
            next(opponent for opponent in game.players if opponent != player)
        ],
        stack=round_.stacks[player],
        pot=game.current_pot,
        minimum_raise=round_.min_raise,
        can_raise=player not in round_.acted_since_full_raise,
        action_profile=action_profile,
    )


def abstract_action_specs_from_observation(observation) -> tuple[ActionSpec, ...]:
    """Build the same action menu from player-visible observation fields."""
    contribution = observation.street_contribution
    opponent_contribution = observation.opponent_street_contribution
    can_raise = any(
        action.type in (ActionType.RAISE, ActionType.ALL_IN)
        for action in observation.legal_actions
    )
    specs = _action_specs(
        contribution=contribution,
        opponent_contribution=opponent_contribution,
        stack=observation.stack,
        pot=observation.pot,
        minimum_raise=observation.min_raise,
        can_raise=can_raise,
        action_profile=observation.action_profile,
    )
    return tuple(spec for spec in specs if spec.engine in observation.legal_actions)


def _action_specs(
    *,
    contribution: int,
    opponent_contribution: int,
    stack: int,
    pot: int,
    minimum_raise: int,
    can_raise: bool,
    action_profile: str,
) -> tuple[ActionSpec, ...]:
    if action_profile not in ACTION_PROFILES:
        raise ValueError(f"unknown action profile {action_profile!r}")
    maximum = contribution + stack
    current_bet = max(contribution, opponent_contribution)
    owed = current_bet - contribution
    specs: list[ActionSpec] = []

    if owed > 0:
        specs.extend(
            (
                ActionSpec(AbstractAction.FOLD, Action(ActionType.FOLD)),
                ActionSpec(AbstractAction.CALL, Action(ActionType.CALL)),
            )
        )
    else:
        specs.append(ActionSpec(AbstractAction.CHECK, Action(ActionType.CHECK)))

    candidates: list[tuple[AbstractAction, int]] = []
    if current_bet == 0 and owed == 0 and stack > 0 and can_raise:
        minimum_increment = min(minimum_raise, stack)
        for label, fraction in ACTION_PROFILES[action_profile]["bets"]:
            increment = (
                minimum_increment
                if fraction is None
                else max(minimum_increment, int(pot * fraction))
            )
            action = AbstractAction(label)
            candidates.append((action, min(maximum, contribution + increment)))
    elif current_bet > 0 and can_raise and maximum > current_bet:
        pot_after_call = pot + min(owed, stack)
        for label, fraction in ACTION_PROFILES[action_profile]["raises"]:
            increment = (
                minimum_raise
                if fraction is None
                else max(minimum_raise, int(pot_after_call * fraction))
            )
            action = AbstractAction(label)
            candidates.append((action, min(maximum, current_bet + increment)))

    # If two abstract sizes collapse to one target, retain just one. A sizing
    # action that reaches the stack maximum is represented by the explicit
    # all-in action below.
    seen_targets: set[int] = set()
    for abstract, target in candidates:
        if target >= maximum or target <= current_bet or target in seen_targets:
            continue
        if abstract.value.startswith("bet_"):
            increment = target - contribution
            if increment < minimum_raise:
                continue
            engine_action = Action(ActionType.BET, target)
        else:
            if target - current_bet < minimum_raise:
                continue
            engine_action = Action(ActionType.RAISE, target)
        seen_targets.add(target)
        specs.append(ActionSpec(abstract, engine_action))

    if stack > 0:
        all_in_target = maximum
        all_in_is_short_call = owed > 0 and all_in_target <= current_bet
        all_in_raise_reopened = can_raise or all_in_target <= current_bet
        if not all_in_is_short_call and all_in_raise_reopened:
            specs.append(ActionSpec(AbstractAction.ALL_IN, Action(ActionType.ALL_IN)))
    return tuple(specs)


def engine_action_for(
    abstract: AbstractAction, specs: tuple[ActionSpec, ...]
) -> Action:
    for spec in specs:
        if spec.abstract is abstract:
            return spec.engine
    raise ValueError(f"abstract action {abstract.value!r} is not legal here")
