"""Engine-backed abstracted heads-up Hold'em state adapter."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from random import Random

from pokerlab.engine import Action, Deck, GameState
from pokerlab.research.nlhe.actions import (
    AbstractAction,
    abstract_action_specs,
    engine_action_for,
)
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.information_set import HUNLInformationSet

History = tuple[tuple[str, bool, str, int | None], ...]


def _clone_engine(game: GameState) -> GameState:
    """Clone mutable game containers without copying immutable cards/config.

    The deck RNG is only used during the initial shuffle; later transitions
    pop from a branch-local card list. Thus its RNG object can be shared safely.
    """
    child = copy(game)
    child.players = list(game.players)
    child.stacks = dict(game.stacks)
    child.deck = copy(game.deck)
    child.deck._cards = list(game.deck._cards)
    child.hole_cards = dict(game.hole_cards)
    child.community_cards = list(game.community_cards)
    child.pot = copy(game.pot)
    child.pot.contributions = dict(game.pot.contributions)
    child.pot.folded = set(game.pot.folded)
    child.folded = set(game.folded)
    child.all_in = set(game.all_in)
    child.winners = dict(game.winners)
    child.hand_ranks = dict(game.hand_ranks)
    child.betting = copy(game.betting)
    child.betting.players = list(game.betting.players)
    child.betting.stacks = dict(game.betting.stacks)
    child.betting.contributions = dict(game.betting.contributions)
    child.betting.folded = set(game.betting.folded)
    child.betting.all_in = set(game.betting.all_in)
    child.betting.acted_since_full_raise = set(game.betting.acted_since_full_raise)
    child.betting.pending = set(game.betting.pending)
    return child


@dataclass(slots=True)
class AbstractHUNLState:
    engine: GameState
    config: HUNLGameConfig
    button: str
    initial_stacks: tuple[int, int]
    history: History = ()

    @classmethod
    def new_hand(cls, config: HUNLGameConfig, rng: Random, *, button_index: int = 0):
        names = ("p0", "p1")
        deck = Deck(rng=rng)
        engine = GameState(
            {name: config.starting_stack_chips for name in names},
            small_blind=config.small_blind_chips,
            big_blind=config.big_blind_chips,
            button=names[button_index],
            deck=deck,
        )
        return cls(
            engine,
            config,
            names[button_index],
            (config.starting_stack_chips,) * 2,
        )

    @property
    def terminal(self) -> bool:
        return self.engine.complete

    @property
    def current_player(self) -> int:
        return self.engine.players.index(self.engine.current_player)

    @property
    def legal_specs(self):
        return abstract_action_specs(
            self.engine, self.engine.current_player, self.config.action_profile
        )

    def information_set(self, player: int) -> HUNLInformationSet:
        from pokerlab.agents.base import ActionRecord, observe
        from pokerlab.research.nlhe.information_set import (
            information_set_from_observation,
        )

        name = self.engine.players[player]
        button_idx = self.engine.players.index(self.button)
        records = tuple(
            ActionRecord(
                player=self.engine.players[
                    button_idx if button_actor else 1 - button_idx
                ],
                action=Action(action_type, amount),
                street=street,
            )
            for street, button_actor, action_type, amount in self.history
        )
        observation = observe(
            self.engine,
            name,
            records,
            action_profile=self.config.action_profile,
        )
        normalized_history = tuple(
            (street, button_actor, action_type, amount)
            for street, button_actor, action_type, amount in self.history
        )
        return information_set_from_observation(
            observation, normalized_history, self.config
        )

    def act(self, abstract: AbstractAction) -> AbstractHUNLState:
        if self.terminal:
            raise ValueError("hand is terminal")
        actor = self.engine.current_player
        specs = self.legal_specs
        action = engine_action_for(abstract, specs)
        child = copy(self)
        child.engine = _clone_engine(self.engine)
        actor_idx = child.engine.players.index(actor)
        before_street = child.engine.street.value
        child.engine.act(action, player=actor)
        child.history += (
            (
                before_street,
                actor_idx == child.engine.players.index(child.button),
                action.type.value,
                action.amount,
            ),
        )
        return child

    def utility(self, player: int) -> float:
        if not self.terminal:
            raise ValueError("utility requested for nonterminal state")
        name = self.engine.players[player]
        return float(self.engine.stacks[name] - self.initial_stacks[player])
