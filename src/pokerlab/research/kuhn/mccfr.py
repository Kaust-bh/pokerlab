"""External-sampling Monte Carlo CFR for Kuhn Poker."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pokerlab.research.kuhn.evaluation import evaluate_nash
from pokerlab.research.kuhn.game import (
    KuhnAction,
    KuhnGame,
    KuhnState,
    player_for_history,
)
from pokerlab.research.kuhn.information_sets import InformationSet
from pokerlab.research.kuhn.strategy import normalize_average_strategy, regret_matching

KuhnProfile = Mapping[InformationSet, Mapping[KuhnAction, float]]


@dataclass(frozen=True, slots=True)
class MCCFRConfig:
    iterations: int = 10_000
    seed: int = 0
    evaluation_interval: int = 100
    logging_interval: int = 1_000

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.evaluation_interval < 1 or self.logging_interval < 1:
            raise ValueError("evaluation and logging intervals must be positive")


@dataclass(frozen=True, slots=True)
class MCCFRRecord:
    iteration: int
    expected_game_value_player0: float
    expected_game_value_player1: float
    nashconv: float
    exploitability: float
    information_sets: int
    training_elapsed_seconds: float
    evaluation_elapsed_seconds: float
    iterations_per_second: float
    seed: int
    sampled_nodes: int


def _catalog() -> dict[InformationSet, tuple[KuhnAction, ...]]:
    found: dict[InformationSet, tuple[KuhnAction, ...]] = {}

    def visit(state: KuhnState) -> None:
        if state.is_terminal:
            return
        info = InformationSet.from_state(state)
        found[info] = state.legal_actions
        for action in state.legal_actions:
            visit(state.act(action))

    for deal in KuhnGame.deals():
        visit(KuhnGame.initial_state(deal))
    return found


def _own_action_sequences(
    information_sets: Mapping[InformationSet, tuple[KuhnAction, ...]],
) -> dict[InformationSet, tuple[tuple[InformationSet, KuhnAction], ...]]:
    sequences = {}
    for info in information_sets:
        own_actions = []
        for index, action in enumerate(info.history):
            prefix = info.history[:index]
            if player_for_history(prefix) == info.player:
                previous = InformationSet(info.player, info.private_card, prefix)
                own_actions.append((previous, action))
        sequences[info] = tuple(own_actions)
    return sequences


class MCCFRSolver:
    """Two-player ES-MCCFR with exact own-reach average-strategy updates.

    Each regret pass samples one ordered deal and samples opponent actions while
    enumerating every traverser action. Conditional on the sampled external
    choices, the action-value difference is the block regret from Lanctot et al.
    Its expectation is the vanilla counterfactual regret: the chance and
    opponent sampling probabilities supply their reach factors, so no extra
    importance multiplier is applied and traverser reach is excluded.

    Average strategy is accumulated separately from sampled regret traversal.
    The game's information sets are catalogued once, and each iteration adds
    pi_i(I) * sigma_i(I,a) using the player's own action sequence. This is the
    exact vanilla average-strategy numerator, including infosets not reached by
    the sampled external trajectory. It avoids chance/opponent visitation bias
    without adding sampling weights to regret estimates.
    """

    def __init__(self, config: MCCFRConfig | None = None) -> None:
        self.config = config or MCCFRConfig()
        self._rng = random.Random(self.config.seed)
        self.iterations = 0
        self.regret_sum: dict[InformationSet, dict[KuhnAction, float]] = {}
        self.strategy_sum: dict[InformationSet, dict[KuhnAction, float]] = {}
        self._legal = _catalog()
        self._own_sequences = _own_action_sequences(self._legal)
        for info, actions in self._legal.items():
            self.regret_sum[info] = dict.fromkeys(actions, 0.0)
            self.strategy_sum[info] = dict.fromkeys(actions, 0.0)
        self.records: list[MCCFRRecord] = []
        self.sampled_nodes = 0

    @property
    def information_sets(self) -> tuple[InformationSet, ...]:
        return tuple(sorted(self._legal, key=lambda item: item.identifier))

    def current_strategy(self, info: InformationSet) -> dict[KuhnAction, float]:
        return regret_matching(self.regret_sum[info], self._legal[info])

    def current_strategy_profile(self) -> dict[InformationSet, dict[KuhnAction, float]]:
        return {info: self.current_strategy(info) for info in self.information_sets}

    def average_strategy(self, info: InformationSet) -> dict[KuhnAction, float]:
        return normalize_average_strategy(self.strategy_sum[info], self._legal[info])

    def average_strategy_profile(self) -> dict[InformationSet, dict[KuhnAction, float]]:
        return {info: self.average_strategy(info) for info in self.information_sets}

    def _traverse_external(
        self,
        state: KuhnState,
        traverser: int,
        profile: KuhnProfile,
        regret_delta: dict[InformationSet, dict[KuhnAction, float]],
        *,
        opponent_choices: Mapping[InformationSet, KuhnAction] | None = None,
        node_counter: list[int] | None = None,
    ) -> float:
        if node_counter is not None:
            node_counter[0] += 1
        if state.is_terminal:
            return float(state.utility(0))
        player = state.current_player
        if player is None:
            raise RuntimeError("nonterminal Kuhn state has no acting player")
        info = InformationSet.from_state(state)
        actions = state.legal_actions
        strategy = profile[info]

        if player == traverser:
            values = {
                action: self._traverse_external(
                    state.act(action),
                    traverser,
                    profile,
                    regret_delta,
                    opponent_choices=opponent_choices,
                    node_counter=node_counter,
                )
                for action in actions
            }
            node_value = sum(strategy[action] * values[action] for action in actions)
            sign = 1.0 if traverser == 0 else -1.0
            updates = regret_delta.setdefault(info, dict.fromkeys(actions, 0.0))
            for action in actions:
                updates[action] += sign * (values[action] - node_value)
            return node_value

        if opponent_choices is not None and info in opponent_choices:
            action = opponent_choices[info]
            if action not in actions:
                raise ValueError(f"sampled illegal action at {info.identifier}")
        else:
            action = self._rng.choices(
                actions, weights=[strategy[item] for item in actions], k=1
            )[0]
        return self._traverse_external(
            state.act(action),
            traverser,
            profile,
            regret_delta,
            opponent_choices=opponent_choices,
            node_counter=node_counter,
        )

    def _average_strategy_delta(
        self, profile: KuhnProfile
    ) -> dict[InformationSet, dict[KuhnAction, float]]:
        delta = {}
        for info in self.information_sets:
            own_reach = 1.0
            for previous, action in self._own_sequences[info]:
                own_reach *= profile[previous][action]
            delta[info] = {
                action: own_reach * profile[info][action]
                for action in self._legal[info]
            }
        return delta

    def run_iteration(self) -> int:
        """Sample one external trajectory per player and commit frozen updates."""
        profile = self.current_strategy_profile()
        regret_delta: dict[InformationSet, dict[KuhnAction, float]] = {}
        node_counter = [0]
        deals = tuple(KuhnGame.deals())
        for traverser in (0, 1):
            deal = self._rng.choice(deals)
            self._traverse_external(
                KuhnGame.initial_state(deal),
                traverser,
                profile,
                regret_delta,
                node_counter=node_counter,
            )
        average_delta = self._average_strategy_delta(profile)
        for info, actions in regret_delta.items():
            for action, value in actions.items():
                self.regret_sum[info][action] += value
        for info, actions in average_delta.items():
            for action, value in actions.items():
                self.strategy_sum[info][action] += value
        self.iterations += 1
        self.sampled_nodes = node_counter[0]
        return self.sampled_nodes

    def train(
        self,
        iterations: int | None = None,
        *,
        evaluation_interval: int | None = None,
        logging_interval: int | None = None,
        callback: Callable[[MCCFRRecord], None] | None = None,
    ) -> tuple[MCCFRRecord, ...]:
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
        previous = len(self.records)
        training_elapsed = 0.0
        for local in range(1, count + 1):
            started = time.perf_counter()
            self.run_iteration()
            training_elapsed += time.perf_counter() - started
            final = local == count
            if self.iterations % eval_every == 0 or final:
                evaluation_started = time.perf_counter()
                result = evaluate_nash(self.average_strategy_profile())
                evaluation_elapsed = time.perf_counter() - evaluation_started
                record = MCCFRRecord(
                    self.iterations,
                    result.expected_value_player0,
                    -result.expected_value_player0,
                    result.nashconv,
                    result.exploitability,
                    len(self._legal),
                    training_elapsed,
                    evaluation_elapsed,
                    local / training_elapsed if training_elapsed else float("inf"),
                    self.config.seed,
                    self.sampled_nodes,
                )
                self.records.append(record)
                if callback is not None and (self.iterations % log_every == 0 or final):
                    callback(record)
        return tuple(self.records[previous:])
