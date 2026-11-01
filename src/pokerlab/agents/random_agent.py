"""Uniform random legal-action baseline."""

import random

from pokerlab.agents.base import Agent, Observation
from pokerlab.engine import Action


class RandomAgent(Agent):
    """Choose uniformly from the provided action abstraction using a private RNG."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self._seed = seed if seed is not None else self._seed
        self._rng = random.Random(self._seed)

    def act(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            raise ValueError("no legal actions are available")
        return self._rng.choice(observation.legal_actions)

    def action_probabilities(self, observation: Observation) -> dict[Action, float]:
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions are available")
        prob = 1.0 / len(legal)
        return {action: prob for action in legal}
