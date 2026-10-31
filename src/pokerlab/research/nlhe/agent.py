"""Tournament-compatible policy backed by an MCCFR average strategy."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping

from pokerlab.agents.base import Agent, Observation
from pokerlab.research.nlhe.actions import abstract_action_specs_from_observation
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.information_set import (
    HUNLInformationSet,
    information_set_from_observation,
)


class AbstractHUNLAgent(Agent):
    def __init__(
        self,
        strategy: Mapping[HUNLInformationSet, Mapping],
        config: HUNLGameConfig,
        seed: int = 0,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self.rng = random.Random(seed)
        self.decisions = 0
        self.decision_seconds = 0.0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def act(self, observation: Observation):
        started = time.perf_counter()
        history = tuple(
            (
                record.street,
                record.player == observation.player
                if observation.is_button
                else record.player != observation.player,
                record.action.type.value,
                record.action.amount,
            )
            for record in observation.history
        )
        info = information_set_from_observation(observation, history, self.config)
        probabilities = self.strategy.get(info)
        specs = abstract_action_specs_from_observation(observation)
        if probabilities is None:
            action_spec = self.rng.choice(specs)
        else:
            weights = [
                max(0.0, probabilities.get(spec.abstract, 0.0)) for spec in specs
            ]
            if sum(weights) <= 0:
                action_spec = self.rng.choice(specs)
            else:
                action_spec = self.rng.choices(specs, weights=weights, k=1)[0]
        self.decisions += 1
        self.decision_seconds += time.perf_counter() - started
        return action_spec.engine


def serialize_information_set(info: HUNLInformationSet) -> dict:
    return {
        "player": info.player,
        "street": info.street,
        "private_bucket": info.private_bucket,
        "board": info.board,
        "pot": info.pot,
        "effective_stack": info.effective_stack,
        "street_contributions": info.street_contributions,
        "history": info.history,
        "legal_actions": [action.value for action in info.legal_actions],
    }


def deserialize_information_set(data: dict) -> HUNLInformationSet:
    from pokerlab.research.nlhe.actions import AbstractAction

    return HUNLInformationSet(
        player=data["player"],
        street=data["street"],
        private_bucket=data["private_bucket"],
        board=tuple(tuple(card) for card in data["board"]),
        pot=data["pot"],
        effective_stack=data["effective_stack"],
        street_contributions=tuple(data["street_contributions"]),
        history=tuple(tuple(event) for event in data["history"]),
        legal_actions=tuple(AbstractAction(action) for action in data["legal_actions"]),
    )


def serialize_strategy(strategy: Mapping[HUNLInformationSet, Mapping]) -> dict:
    """Convert stable info keys and action masses to JSON-compatible objects."""
    rows = []
    for info, probabilities in sorted(strategy.items(), key=lambda item: repr(item[0])):
        rows.append(
            {
                "info": serialize_information_set(info),
                "strategy": {
                    action.value: value for action, value in probabilities.items()
                },
            }
        )
    return {"information_sets": rows}


def deserialize_strategy(payload: dict):
    from pokerlab.research.nlhe.actions import AbstractAction

    strategy = {}
    for row in payload["information_sets"]:
        info = deserialize_information_set(row["info"])
        strategy[info] = {
            AbstractAction(action): float(value)
            for action, value in row["strategy"].items()
        }
    return strategy
