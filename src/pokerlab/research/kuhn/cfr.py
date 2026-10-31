"""Vanilla counterfactual regret minimization for Kuhn Poker."""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from pokerlab.research.kuhn.evaluation import evaluate_nash
from pokerlab.research.kuhn.game import KuhnAction, KuhnGame, KuhnState
from pokerlab.research.kuhn.information_sets import InformationSet
from pokerlab.research.kuhn.strategy import normalize_average_strategy, regret_matching


@dataclass(frozen=True, slots=True)
class CFRConfig:
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
class TrainingRecord:
    iteration: int
    expected_game_value: float
    exploitability: float
    nashconv: float
    information_sets: int
    runtime_seconds: float


class CFRSolver:
    """Full-tree, two-player CFR with exact chance enumeration.

    At an information set I for player i, instantaneous regret is

        r_i(I,a) = sum_h pi_-i(h) [v_i(h,a) - v_i(h)],

    where pi_-i includes chance and the opponent's reach, but excludes player i's
    own reach. The current strategy is regret matching over cumulative positive
    regrets. Average-strategy mass accumulates pi_i(I) * sigma_i(I,a), with chance
    deals coalesced so each information set contributes once per iteration.
    Regret deltas are applied only after all six chance deals, keeping the strategy
    profile fixed for the entire vanilla-CFR iteration.
    """

    def __init__(self, config: CFRConfig | None = None) -> None:
        self.config = config or CFRConfig()
        self._rng = random.Random(self.config.seed)
        self.iterations = 0
        self.regret_sum: dict[InformationSet, dict[KuhnAction, float]] = {}
        self.strategy_sum: dict[InformationSet, dict[KuhnAction, float]] = {}
        self._legal: dict[InformationSet, tuple[KuhnAction, ...]] = {}
        self.records: list[TrainingRecord] = []

    def _ensure_information_set(
        self, info_set: InformationSet, actions: tuple[KuhnAction, ...]
    ) -> None:
        known = self._legal.get(info_set)
        if known is not None and known != actions:
            raise RuntimeError("an information set produced inconsistent legal actions")
        self._legal[info_set] = actions
        self.regret_sum.setdefault(info_set, {action: 0.0 for action in actions})
        self.strategy_sum.setdefault(info_set, {action: 0.0 for action in actions})

    def current_strategy(self, info_set: InformationSet) -> dict[KuhnAction, float]:
        actions = self._legal.get(info_set)
        if actions is None:
            from pokerlab.research.kuhn.game import legal_actions_for_history

            actions = legal_actions_for_history(info_set.history)
            if not actions:
                raise ValueError("terminal histories have no strategy")
            self._ensure_information_set(info_set, actions)
        return regret_matching(self.regret_sum[info_set], actions)

    def average_strategy(self, info_set: InformationSet) -> dict[KuhnAction, float]:
        actions = self._legal.get(info_set)
        if actions is None:
            from pokerlab.research.kuhn.game import legal_actions_for_history

            actions = legal_actions_for_history(info_set.history)
            if not actions:
                raise ValueError("terminal histories have no strategy")
            self._ensure_information_set(info_set, actions)
        return normalize_average_strategy(self.strategy_sum[info_set], actions)

    def average_strategy_profile(
        self,
    ) -> dict[InformationSet, dict[KuhnAction, float]]:
        return {
            info_set: self.average_strategy(info_set)
            for info_set in sorted(self._legal, key=lambda item: item.identifier)
        }

    def strategy_snapshot(
        self,
    ) -> dict[InformationSet, dict[KuhnAction, float]]:
        """Return the current average profile, never the last iteration alone."""
        return self.average_strategy_profile()

    def _traverse(
        self,
        state: KuhnState,
        reach: tuple[float, float],
        chance_reach: float,
        regret_delta: dict[InformationSet, dict[KuhnAction, float]],
        average_delta: dict[InformationSet, dict[KuhnAction, float]],
        averaged_this_iteration: set[InformationSet],
    ) -> float:
        if state.is_terminal:
            return float(state.utility(0))

        player = state.current_player
        if player is None:
            raise RuntimeError("nonterminal state has no acting player")
        info_set = InformationSet.from_state(state)
        actions = state.legal_actions
        self._ensure_information_set(info_set, actions)
        regret_delta.setdefault(info_set, dict.fromkeys(actions, 0.0))
        average_delta.setdefault(info_set, dict.fromkeys(actions, 0.0))
        strategy = regret_matching(self.regret_sum[info_set], actions)

        if info_set not in averaged_this_iteration:
            for action in actions:
                average_delta[info_set][action] = reach[player] * strategy[action]
            averaged_this_iteration.add(info_set)

        action_values: dict[KuhnAction, float] = {}
        node_value = 0.0
        for action in actions:
            next_reach = list(reach)
            next_reach[player] *= strategy[action]
            child_value = self._traverse(
                state.act(action),
                (next_reach[0], next_reach[1]),
                chance_reach,
                regret_delta,
                average_delta,
                averaged_this_iteration,
            )
            action_values[action] = child_value
            node_value += strategy[action] * child_value

        sign = 1.0 if player == 0 else -1.0
        counterfactual_weight = chance_reach * reach[1 - player]
        for action in actions:
            actor_action_value = sign * action_values[action]
            actor_node_value = sign * node_value
            regret_delta[info_set][action] += counterfactual_weight * (
                actor_action_value - actor_node_value
            )
        return node_value

    def run_iteration(self) -> None:
        """Run one simultaneous vanilla-CFR update over all chance outcomes."""
        deals = list(KuhnGame.deals())
        self._rng.shuffle(deals)
        regret_delta: dict[InformationSet, dict[KuhnAction, float]] = {}
        average_delta: dict[InformationSet, dict[KuhnAction, float]] = {}
        averaged_this_iteration: set[InformationSet] = set()
        chance_reach = 1.0 / len(deals)
        for deal in deals:
            self._traverse(
                KuhnGame.initial_state(deal),
                (1.0, 1.0),
                chance_reach,
                regret_delta,
                average_delta,
                averaged_this_iteration,
            )
        for info_set, actions in regret_delta.items():
            for action, delta in actions.items():
                self.regret_sum[info_set][action] += delta
        for info_set, actions in average_delta.items():
            for action, weight in actions.items():
                self.strategy_sum[info_set][action] += weight
        self.iterations += 1

    def train(
        self,
        iterations: int | None = None,
        *,
        evaluation_interval: int | None = None,
        logging_interval: int | None = None,
        callback: Callable[[TrainingRecord], None] | None = None,
    ) -> tuple[TrainingRecord, ...]:
        """Train and periodically evaluate the average strategy profile."""
        count = self.config.iterations if iterations is None else iterations
        evaluate_every = (
            self.config.evaluation_interval
            if evaluation_interval is None
            else evaluation_interval
        )
        log_every = (
            self.config.logging_interval
            if logging_interval is None
            else logging_interval
        )
        if count < 1 or evaluate_every < 1 or log_every < 1:
            raise ValueError("training intervals and iteration count must be positive")
        start_iteration = self.iterations
        started = time.perf_counter()
        initial_records = len(self.records)
        for local_iteration in range(1, count + 1):
            self.run_iteration()
            is_final = local_iteration == count
            if self.iterations % evaluate_every == 0 or is_final:
                profile = self.average_strategy_profile()
                evaluation = evaluate_nash(profile)
                record = TrainingRecord(
                    self.iterations,
                    evaluation.expected_value_player0,
                    evaluation.exploitability,
                    evaluation.nashconv,
                    len(self._legal),
                    time.perf_counter() - started,
                )
                self.records.append(record)
                if callback is not None and (
                    self.iterations % log_every == 0 or is_final
                ):
                    callback(record)
        if self.iterations != start_iteration + count:
            raise RuntimeError("training iteration count changed unexpectedly")
        return tuple(self.records[initial_records:])
