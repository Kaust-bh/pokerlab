"""Deterministic, player-observable encodings for Kuhn and Leduc Poker."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol, TypeVar

from pokerlab.research.kuhn.game import KuhnAction, KuhnCard, KuhnGame, KuhnState
from pokerlab.research.kuhn.information_sets import (
    InformationSet as KuhnInformationSet,
)
from pokerlab.research.leduc.betting import Action as LeducAction
from pokerlab.research.leduc.cards import DECK, Card
from pokerlab.research.leduc.game import LeducGame, LeducState
from pokerlab.research.leduc.information_sets import (
    InformationSet as LeducInformationSet,
)

State = TypeVar("State")
Information = TypeVar("Information")
Action = TypeVar("Action")


class GameCodec(Protocol[State, Information, Action]):
    """Adapter boundary; future games can provide their own exact encoding."""

    game_name: str
    action_vocabulary: tuple[Action, ...]
    feature_dim: int

    def initial_state(self, rng: random.Random) -> State: ...

    def information_set(self, state: State) -> Information: ...

    def encode(self, info: Information) -> tuple[float, ...]: ...

    def is_terminal(self, state: State) -> bool: ...

    def is_chance_node(self, state: State) -> bool: ...

    def chance_outcomes(self, state: State) -> Sequence[tuple[object, float]]: ...

    def deal(self, state: State, outcome: object) -> State: ...

    def current_player(self, state: State) -> int: ...

    def legal_actions(self, state: State) -> tuple[Action, ...]: ...

    def act(self, state: State, action: Action) -> State: ...

    def utility(self, state: State, player: int) -> float: ...

    def action_index(self, action: Action) -> int: ...


class KuhnCodec:
    game_name = "kuhn"
    action_vocabulary = tuple(KuhnAction)
    feature_dim = 2 + 3 + 3 * len(KuhnAction)

    def initial_state(self, rng: random.Random) -> KuhnState:
        deals = tuple(KuhnGame.deals())
        return KuhnGame.initial_state(rng.choice(deals))

    def information_set(self, state: KuhnState) -> KuhnInformationSet:
        return KuhnInformationSet.from_state(state)

    def encode(self, info: KuhnInformationSet) -> tuple[float, ...]:
        features = [float(info.player == player) for player in (0, 1)]
        features.extend(float(info.private_card == card) for card in KuhnCard)
        for slot in range(3):
            for action in KuhnAction:
                features.append(
                    float(slot < len(info.history) and info.history[slot] is action)
                )
        return tuple(features)

    def is_terminal(self, state: KuhnState) -> bool:
        return state.is_terminal

    def is_chance_node(self, state: KuhnState) -> bool:
        return False

    def chance_outcomes(self, state: KuhnState):
        return ()

    def deal(self, state: KuhnState, outcome: object) -> KuhnState:
        raise ValueError("Kuhn chance is sampled by initial_state")

    def current_player(self, state: KuhnState) -> int:
        player = state.current_player
        if player is None:
            raise ValueError("terminal Kuhn state has no acting player")
        return player

    def legal_actions(self, state: KuhnState) -> tuple[KuhnAction, ...]:
        return state.legal_actions

    def act(self, state: KuhnState, action: KuhnAction) -> KuhnState:
        return state.act(action)

    def utility(self, state: KuhnState, player: int) -> float:
        return float(state.utility(player))

    def action_index(self, action: KuhnAction) -> int:
        return self.action_vocabulary.index(action)


class LeducCodec:
    game_name = "leduc"
    action_vocabulary = tuple(LeducAction)
    _history_slots = 4
    feature_dim = (
        2 + len(DECK) + (len(DECK) + 1) + 2 + 2 * _history_slots * len(LeducAction)
    )

    def initial_state(self, rng: random.Random) -> LeducState:
        deals = LeducGame.deals()
        return LeducGame.initial_state(rng.choice(deals))

    def information_set(self, state: LeducState) -> LeducInformationSet:
        return LeducInformationSet.from_state(state)

    def encode(self, info: LeducInformationSet) -> tuple[float, ...]:
        features = [float(info.player == player) for player in (0, 1)]
        features.extend(float(info.private_card == card) for card in DECK)
        features.extend(
            float(
                info.public_card is None
                if index == 0
                else info.public_card == DECK[index - 1]
            )
            for index in range(len(DECK) + 1)
        )
        features.extend(
            float(info.round_index == round_index) for round_index in (0, 1)
        )
        for history in info.histories:
            if len(history) > self._history_slots:
                raise ValueError("Leduc betting history exceeds encoder capacity")
            for slot in range(self._history_slots):
                for action in LeducAction:
                    features.append(
                        float(slot < len(history) and history[slot] is action)
                    )
        return tuple(features)

    def is_terminal(self, state: LeducState) -> bool:
        return state.is_terminal

    def is_chance_node(self, state: LeducState) -> bool:
        return state.is_chance_node

    def chance_outcomes(self, state: LeducState):
        return state.chance_outcomes

    def deal(self, state: LeducState, outcome: object) -> LeducState:
        if not isinstance(outcome, Card):
            raise TypeError("Leduc chance outcome must be a Card")
        return state.deal_public(outcome)

    def current_player(self, state: LeducState) -> int:
        player = state.current_player
        if player is None:
            raise ValueError("Leduc chance or terminal state has no acting player")
        return player

    def legal_actions(self, state: LeducState) -> tuple[LeducAction, ...]:
        return state.legal_actions

    def act(self, state: LeducState, action: LeducAction) -> LeducState:
        return state.act(action)

    def utility(self, state: LeducState, player: int) -> float:
        return state.utility(player)

    def action_index(self, action: LeducAction) -> int:
        return self.action_vocabulary.index(action)
