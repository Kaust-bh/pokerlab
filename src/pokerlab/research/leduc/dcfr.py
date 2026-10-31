"""Full-tree Discounted CFR for fixed-limit Leduc Poker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pokerlab.research.cfr_variant_utils import (
    CFRVariantRecord,
    discount_regret_sum,
    discount_strategy_sum,
    train_variant,
    validate_dcfr_parameters,
)
from pokerlab.research.leduc.cfr import CFRConfig, CFRSolver
from pokerlab.research.leduc.evaluation import evaluate_nash
from pokerlab.research.leduc.game import LeducGame
from pokerlab.research.leduc.information_sets import InformationSet


@dataclass(frozen=True, slots=True)
class DCFRConfig:
    iterations: int = 1_000
    seed: int = 0
    evaluation_interval: int = 100
    logging_interval: int = 1_000
    alpha: float = 1.5
    beta: float = 0.0
    gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.evaluation_interval < 1 or self.logging_interval < 1:
            raise ValueError("evaluation and logging intervals must be positive")
        validate_dcfr_parameters(self.alpha, self.beta, self.gamma)


class DCFRSolver(CFRSolver):
    """Simultaneous full-tree DCFR using Brown-Sandholm discount factors."""

    def __init__(self, config: DCFRConfig | None = None) -> None:
        self.config = config or DCFRConfig()
        super().__init__(
            CFRConfig(
                self.config.iterations,
                self.config.seed,
                self.config.evaluation_interval,
                self.config.logging_interval,
            )
        )
        self.config = config or DCFRConfig()
        self.traversals = 0
        self.player_updates = 0

    def run_iteration(self) -> None:
        deals = list(LeducGame.deals())
        self._rng.shuffle(deals)
        regret_delta = {}
        average_delta = {}
        averaged: set[InformationSet] = set()
        node_counter = [0]
        chance = 1.0 / len(deals)
        for deal in deals:
            self._traverse(
                LeducGame.initial_state(deal),
                (1.0, 1.0),
                chance,
                regret_delta,
                average_delta,
                averaged,
                node_counter,
            )
        iteration = self.iterations + 1
        for info, actions in self.regret_sum.items():
            for action in actions:
                self.regret_sum[info][action] = discount_regret_sum(
                    self.regret_sum[info][action],
                    regret_delta.get(info, {}).get(action, 0.0),
                    iteration,
                    self.config.alpha,
                    self.config.beta,
                )
                self.strategy_sum[info][action] = discount_strategy_sum(
                    self.strategy_sum[info][action],
                    average_delta.get(info, {}).get(action, 0.0),
                    iteration,
                    self.config.gamma,
                )
        self.iterations = iteration
        self.game_tree_nodes = node_counter[0]
        self.traversals += 1
        self.player_updates += 2

    def train(
        self,
        iterations: int | None = None,
        *,
        evaluation_interval: int | None = None,
        logging_interval: int | None = None,
        callback: Callable[[CFRVariantRecord], None] | None = None,
    ) -> tuple[CFRVariantRecord, ...]:
        return train_variant(
            self,
            evaluate_nash,
            iterations=iterations,
            evaluation_interval=evaluation_interval,
            logging_interval=logging_interval,
            callback=callback,
        )
