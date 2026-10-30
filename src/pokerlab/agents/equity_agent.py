"""Agent that uses Monte Carlo equity with pot odds and conservative sizing."""

import random
from dataclasses import dataclass

from pokerlab.agents.base import Agent, Observation
from pokerlab.algorithms.monte_carlo import EquityEstimate, estimate_equity
from pokerlab.engine import Action, ActionType


@dataclass(frozen=True, slots=True)
class EquityAgentConfig:
    simulations: int = 500
    call_margin: float = 0.06
    value_bet_equity: float = 0.72
    raise_equity: float = 0.84
    position_discount: float = 0.02


class EquityAgent(Agent):
    """Equity-driven baseline; equity is versus uniform holdings, not a range model."""

    def __init__(
        self, config: EquityAgentConfig | None = None, seed: int | None = None
    ) -> None:
        self.config = config or EquityAgentConfig()
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self._seed = seed if seed is not None else self._seed
        self._rng = random.Random(self._seed)

    def estimate(self, observation: Observation) -> EquityEstimate:
        return estimate_equity(
            observation.hole_cards,
            observation.community_cards,
            simulations=self.config.simulations,
            seed=self._rng.randrange(2**32),
            method="monte_carlo",
        )

    def act(self, observation: Observation) -> Action:
        actions = observation.legal_actions
        if not actions:
            raise ValueError("no legal actions are available")
        equity = self.estimate(observation).equity
        margin = self.config.call_margin
        if observation.in_position:
            margin = max(0.0, margin - self.config.position_discount)

        def find(kind: ActionType) -> Action | None:
            return next((action for action in actions if action.type == kind), None)

        if observation.to_call:
            if equity >= self.config.raise_equity:
                raise_action = find(ActionType.RAISE)
                if (
                    raise_action is not None
                    and observation.to_call < observation.effective_stack * 0.5
                ):
                    return raise_action
            if equity >= observation.pot_odds + margin:
                return find(ActionType.CALL) or find(ActionType.CHECK) or actions[0]
            return find(ActionType.FOLD) or find(ActionType.CALL) or actions[0]
        if equity >= self.config.value_bet_equity:
            bet = find(ActionType.BET)
            if bet is not None:
                return bet
        return find(ActionType.CHECK) or actions[0]
