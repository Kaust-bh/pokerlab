"""Agent-facing observations and a shared policy interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

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

    @property
    def effective_stack(self) -> int:
        return min(self.stack, self.opponent_stack)

    @property
    def pot_odds(self) -> float:
        call_amount = min(self.to_call, self.stack)
        denominator = self.pot + call_amount
        return call_amount / denominator if denominator else 0.0


def legal_actions(game: GameState, player: str) -> tuple[Action, ...]:
    """Return a practical no-limit action abstraction accepted by the engine."""
    round_ = game.betting
    if game.complete or player != round_.current_player or round_.complete:
        return ()
    contribution = round_.contributions[player]
    stack = round_.stacks[player]
    current_bet = round_.current_bet
    owed = current_bet - contribution
    maximum = contribution + stack
    actions: list[Action] = []
    can_raise = player not in round_.acted_since_full_raise

    if owed:
        actions.extend((Action(ActionType.FOLD), Action(ActionType.CALL)))
        if can_raise and maximum > current_bet:
            minimum = current_bet + round_.min_raise
            if maximum >= minimum:
                actions.append(Action(ActionType.RAISE, minimum))
                pot_target = min(
                    maximum, current_bet + max(round_.min_raise, game.current_pot)
                )
                if pot_target > minimum:
                    actions.append(Action(ActionType.RAISE, pot_target))
            actions.append(Action(ActionType.ALL_IN))
    else:
        actions.append(Action(ActionType.CHECK))
        if current_bet == 0 and can_raise and stack:
            minimum = min(round_.min_raise, stack)
            actions.append(Action(ActionType.BET, contribution + minimum))
            pot_target = min(maximum, contribution + max(minimum, game.current_pot))
            if pot_target > contribution + minimum:
                actions.append(Action(ActionType.BET, pot_target))
            actions.append(Action(ActionType.ALL_IN))
    return tuple(dict.fromkeys(actions))


def observe(
    game: GameState, player: str, history: tuple[ActionRecord, ...] = ()
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
        legal_actions=legal_actions(game, player),
        history=history,
    )


class Agent(ABC):
    """A poker policy receives only an `Observation` and returns a legal action."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Choose an action from the supplied legal action set."""

    def reset(self, seed: int | None = None) -> None:
        """Reset private random state before an independent match, if applicable."""


AgentFactory = Callable[[], Agent]
