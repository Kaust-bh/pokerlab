"""Simultaneous vanilla full-tree CFR for the six-card Leduc game."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.evaluation import evaluate_nash
from pokerlab.research.leduc.game import LeducGame, LeducState
from pokerlab.research.leduc.information_sets import InformationSet
from pokerlab.research.leduc.strategy import normalize_average_strategy, regret_matching


@dataclass(frozen=True, slots=True)
class CFRConfig:
    iterations: int = 1_000
    seed: int = 0
    evaluation_interval: int = 1_000
    logging_interval: int = 1_000

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.evaluation_interval < 1 or self.logging_interval < 1:
            raise ValueError("evaluation and logging intervals must be positive")


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    iteration: int
    expected_game_value: float
    exploitability: float
    nashconv: float
    information_sets: int
    runtime_seconds: float
    game_tree_nodes: int
    iterations_per_second: float
    best_response_seconds: float


class CFRSolver:
    """CFR with chance/opponent counterfactual reach and delayed deltas.

    At an information set I for player i, regret uses chance reach times the
    opponent's reach, excluding i's own reach. Average strategy uses the acting
    player's own reach. The same behavioral information set may occur under
    several hidden deals; average mass is coalesced once per iteration because
    those compatible states share the same own reach and their multiplicity is
    constant for a fixed observation. All regret changes are delayed until every
    private deal and public-card chance outcome has been traversed.
    """

    def __init__(self, config: CFRConfig | None = None) -> None:
        self.config = config or CFRConfig()
        self._rng = random.Random(self.config.seed)
        self.iterations = 0
        self.regret_sum: dict[InformationSet, dict[Action, float]] = {}
        self.strategy_sum: dict[InformationSet, dict[Action, float]] = {}
        self._legal: dict[InformationSet, tuple[Action, ...]] = {}
        self.records: list[TrainingRecord] = []
        self.game_tree_nodes = 0

    def _ensure(self, info: InformationSet, actions: tuple[Action, ...]) -> None:
        previous = self._legal.get(info)
        if previous is not None and previous != actions:
            raise RuntimeError("an information set produced inconsistent legal actions")
        self._legal[info] = actions
        self.regret_sum.setdefault(info, dict.fromkeys(actions, 0.0))
        self.strategy_sum.setdefault(info, dict.fromkeys(actions, 0.0))

    def current_strategy(self, info: InformationSet) -> dict[Action, float]:
        actions = self._legal.get(info)
        if actions is None:
            from pokerlab.research.leduc.betting import legal_actions

            actions = legal_actions(info.histories[info.round_index])
            if not actions:
                raise ValueError("chance and terminal histories have no strategy")
            self._ensure(info, actions)
        return regret_matching(self.regret_sum[info], actions)

    def average_strategy(self, info: InformationSet) -> dict[Action, float]:
        actions = self._legal.get(info)
        if actions is None:
            from pokerlab.research.leduc.betting import legal_actions

            actions = legal_actions(info.histories[info.round_index])
            if not actions:
                raise ValueError("chance and terminal histories have no strategy")
            self._ensure(info, actions)
        return normalize_average_strategy(self.strategy_sum[info], actions)

    def average_strategy_profile(self) -> dict[InformationSet, dict[Action, float]]:
        return {
            info: self.average_strategy(info)
            for info in sorted(self._legal, key=lambda item: item.identifier)
        }

    def _traverse(
        self,
        state: LeducState,
        reach: tuple[float, float],
        chance_reach: float,
        regrets: dict[InformationSet, dict[Action, float]],
        averages: dict[InformationSet, dict[Action, float]],
        averaged: set[InformationSet],
        node_counter: list[int],
    ) -> float:
        node_counter[0] += 1
        if state.is_terminal:
            return state.utility(0)
        if state.is_chance_node:
            value = 0.0
            for card, probability in state.chance_outcomes:
                value += probability * self._traverse(
                    state.deal_public(card),
                    reach,
                    chance_reach * probability,
                    regrets,
                    averages,
                    averaged,
                    node_counter,
                )
            return value

        player = state.current_player
        if player is None:
            raise RuntimeError("nonterminal state has no acting player")
        info = InformationSet.from_state(state)
        actions = state.legal_actions
        self._ensure(info, actions)
        regrets.setdefault(info, dict.fromkeys(actions, 0.0))
        averages.setdefault(info, dict.fromkeys(actions, 0.0))
        strategy = regret_matching(self.regret_sum[info], actions)
        if info not in averaged:
            for action in actions:
                averages[info][action] = reach[player] * strategy[action]
            averaged.add(info)

        action_values: dict[Action, float] = {}
        node_value = 0.0
        for action in actions:
            next_reach = list(reach)
            next_reach[player] *= strategy[action]
            child_value = self._traverse(
                state.act(action),
                (next_reach[0], next_reach[1]),
                chance_reach,
                regrets,
                averages,
                averaged,
                node_counter,
            )
            action_values[action] = child_value
            node_value += strategy[action] * child_value

        sign = 1.0 if player == 0 else -1.0
        weight = chance_reach * reach[1 - player]
        for action in actions:
            regrets[info][action] += (
                weight * sign * (action_values[action] - node_value)
            )
        return node_value

    def run_iteration(self) -> None:
        deals = list(LeducGame.deals())
        self._rng.shuffle(deals)
        regrets: dict[InformationSet, dict[Action, float]] = {}
        averages: dict[InformationSet, dict[Action, float]] = {}
        averaged: set[InformationSet] = set()
        counter = [0]
        chance = 1.0 / len(deals)
        for deal in deals:
            self._traverse(
                LeducGame.initial_state(deal),
                (1.0, 1.0),
                chance,
                regrets,
                averages,
                averaged,
                counter,
            )
        for info, actions in regrets.items():
            for action, delta in actions.items():
                self.regret_sum[info][action] += delta
        for info, actions in averages.items():
            for action, mass in actions.items():
                self.strategy_sum[info][action] += mass
        self.game_tree_nodes = counter[0]
        self.iterations += 1

    def train(
        self,
        iterations: int | None = None,
        *,
        evaluation_interval: int | None = None,
        logging_interval: int | None = None,
        callback: Callable[[TrainingRecord], None] | None = None,
    ) -> tuple[TrainingRecord, ...]:
        count = self.config.iterations if iterations is None else iterations
        eval_every = (
            self.config.evaluation_interval
            if evaluation_interval is None
            else evaluation_interval
        )
        log_every = (
            self.config.logging_interval
            if logging_interval is None
            else logging_interval
        )
        if count < 1 or eval_every < 1 or log_every < 1:
            raise ValueError("training intervals and iteration count must be positive")
        started = time.perf_counter()
        previous_records = len(self.records)
        for local in range(1, count + 1):
            self.run_iteration()
            final = local == count
            if self.iterations % eval_every == 0 or final:
                evaluation_started = time.perf_counter()
                evaluation = evaluate_nash(self.average_strategy_profile())
                response_time = time.perf_counter() - evaluation_started
                elapsed = time.perf_counter() - started
                record = TrainingRecord(
                    self.iterations,
                    evaluation.expected_value_player0,
                    evaluation.exploitability,
                    evaluation.nashconv,
                    len(self._legal),
                    elapsed,
                    self.game_tree_nodes,
                    local / elapsed if elapsed else float("inf"),
                    response_time,
                )
                self.records.append(record)
                if callback is not None and (self.iterations % log_every == 0 or final):
                    callback(record)
        return tuple(self.records[previous_records:])
