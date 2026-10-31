"""Agent-facing observations and a shared policy interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from pokerlab.agents.action_profiles import ACTION_PROFILES
from pokerlab.engine import Action, ActionType, GameState
from pokerlab.engine.cards import Card


@dataclass(frozen=True, slots=True)
class ActionRecord:
    player: str
    action: Action
    street: str


@dataclass(frozen=True, slots=True)
class Observation:
    """Only information available to one player; opponent cards are never included."""

    player: str
    hole_cards: tuple[Card, Card]
    community_cards: tuple[Card, ...]
    street: str
    in_position: bool
    stack: int
    opponent_stack: int
    pot: int
    to_call: int
    big_blind: int
    min_raise: int
    legal_actions: tuple[Action, ...]
    history: tuple[ActionRecord, ...] = ()
    street_contribution: int = 0
    opponent_street_contribution: int = 0
    is_button: bool = False
    action_profile: str = "legacy"

    @property
    def effective_stack(self) -> int:
        return min(self.stack, self.opponent_stack)

    @property
    def pot_odds(self) -> float:
        call_amount = min(self.to_call, self.stack)
        denominator = self.pot + call_amount
        return call_amount / denominator if denominator else 0.0


def legal_actions(
    game: GameState, player: str, *, action_profile: str = "legacy"
) -> tuple[Action, ...]:
    """Return a practical no-limit action abstraction accepted by the engine."""
    round_ = game.betting
    if game.complete or player != round_.current_player or round_.complete:
        return ()
    if action_profile not in ACTION_PROFILES:
        raise ValueError(f"unknown action profile {action_profile!r}")
    contribution = round_.contributions[player]
    stack = round_.stacks[player]
    current_bet = round_.current_bet
    owed = current_bet - contribution
    maximum = contribution + stack
    actions: list[Action] = []
    can_raise = player not in round_.acted_since_full_raise

    if action_profile != "legacy":
        sizes = ACTION_PROFILES[action_profile]
        if owed:
            actions.extend((Action(ActionType.FOLD), Action(ActionType.CALL)))
            if can_raise and maximum > current_bet:
                after_call_pot = game.current_pot + min(owed, stack)
                for _label, fraction in sizes["raises"]:
                    increment = (
                        round_.min_raise
                        if fraction is None
                        else max(round_.min_raise, int(after_call_pot * fraction))
                    )
                    target = min(maximum, current_bet + increment)
                    if target <= current_bet or target >= maximum:
                        continue
                    if target - current_bet < round_.min_raise:
                        continue
                    actions.append(Action(ActionType.RAISE, target))
                actions.append(Action(ActionType.ALL_IN))
        else:
            actions.append(Action(ActionType.CHECK))
            if can_raise and stack:
                if current_bet == 0:
                    for _label, fraction in sizes["bets"]:
                        increment = (
                            min(round_.min_raise, stack)
                            if fraction is None
                            else max(
                                round_.min_raise,
                                int(game.current_pot * fraction),
                            )
                        )
                        target = min(maximum, contribution + increment)
                        if target <= contribution or target >= maximum:
                            continue
                        if target - contribution < round_.min_raise:
                            continue
                        actions.append(Action(ActionType.BET, target))
                else:
                    for _label, fraction in sizes["raises"]:
                        increment = (
                            round_.min_raise
                            if fraction is None
                            else max(
                                round_.min_raise,
                                int(game.current_pot * fraction),
                            )
                        )
                        target = min(maximum, current_bet + increment)
                        if target <= current_bet or target >= maximum:
                            continue
                        if target - current_bet < round_.min_raise:
                            continue
                        actions.append(Action(ActionType.RAISE, target))
                actions.append(Action(ActionType.ALL_IN))
        return tuple(dict.fromkeys(actions))

    if owed:
        actions.extend((Action(ActionType.FOLD), Action(ActionType.CALL)))
        if can_raise and maximum > current_bet:
            minimum = current_bet + round_.min_raise
            if maximum >= minimum:
                actions.append(Action(ActionType.RAISE, minimum))
                after_call_pot = game.current_pot + min(owed, stack)
                half_target = min(
                    maximum,
                    current_bet + max(round_.min_raise, int(after_call_pot * 0.5)),
                )
                if (
                    current_bet < half_target < maximum
                    and half_target - current_bet >= round_.min_raise
                ):
                    actions.append(Action(ActionType.RAISE, half_target))
                pot_target = min(
                    maximum,
                    current_bet + max(round_.min_raise, int(after_call_pot)),
                )
                if pot_target > minimum:
                    actions.append(Action(ActionType.RAISE, pot_target))
            # An all-in that only matches the wager duplicates CALL. A jam
            # above the wager is legal only when raising has been reopened.
            if can_raise and maximum > current_bet:
                actions.append(Action(ActionType.ALL_IN))
    else:
        actions.append(Action(ActionType.CHECK))
        if current_bet == 0 and can_raise and stack:
            minimum = min(round_.min_raise, stack)
            actions.append(Action(ActionType.BET, contribution + minimum))
            half_target = min(
                maximum,
                contribution + max(minimum, int(game.current_pot * 0.5)),
            )
            if half_target > contribution + minimum and half_target < maximum:
                actions.append(Action(ActionType.BET, half_target))
            pot_target = min(maximum, contribution + max(minimum, game.current_pot))
            if pot_target > contribution + minimum:
                actions.append(Action(ActionType.BET, pot_target))
            actions.append(Action(ActionType.ALL_IN))
        elif can_raise and maximum > current_bet:
            minimum = current_bet + round_.min_raise
            if maximum >= minimum:
                actions.append(Action(ActionType.RAISE, minimum))
                after_call_pot = game.current_pot
                half_target = min(
                    maximum,
                    current_bet + max(round_.min_raise, int(after_call_pot * 0.5)),
                )
                if (
                    current_bet < half_target < maximum
                    and half_target - current_bet >= round_.min_raise
                ):
                    actions.append(Action(ActionType.RAISE, half_target))
                pot_target = min(
                    maximum,
                    current_bet + max(round_.min_raise, int(after_call_pot)),
                )
                if pot_target > minimum and pot_target < maximum:
                    actions.append(Action(ActionType.RAISE, pot_target))
            actions.append(Action(ActionType.ALL_IN))
    return tuple(dict.fromkeys(actions))


def observe(
    game: GameState,
    player: str,
    history: tuple[ActionRecord, ...] = (),
    *,
    action_profile: str = "legacy",
) -> Observation:
    """Build a player-specific observation without exposing game internals."""
    if player not in game.players:
        raise ValueError("unknown player")
    opponent = next(other for other in game.players if other != player)
    return Observation(
        player=player,
        hole_cards=game.hole_cards[player],
        community_cards=tuple(game.community_cards),
        street=game.street.value,
        in_position=(
            player != game.button
            if game.street.value == "preflop"
            else player == game.button
        ),
        stack=game.stacks[player],
        opponent_stack=game.stacks[opponent],
        pot=game.current_pot,
        to_call=max(0, game.betting.current_bet - game.betting.contributions[player]),
        big_blind=game.big_blind,
        min_raise=game.betting.min_raise,
        legal_actions=legal_actions(game, player, action_profile=action_profile),
        history=history,
        street_contribution=game.betting.contributions[player],
        opponent_street_contribution=game.betting.contributions[opponent],
        is_button=player == game.button,
        action_profile=action_profile,
    )


class Agent(ABC):
    """A poker policy receives only an `Observation` and returns a legal action."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Choose an action from the supplied legal action set."""

    def reset(self, seed: int | None = None) -> None:
        """Reset private random state before an independent match, if applicable."""


AgentFactory = Callable[[], Agent]
