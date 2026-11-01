"""On-demand external-sampling MCCFR for the abstracted HUNL game.

Regret traversals sample a deal and opponent actions from the frozen profile,
while enumerating traverser actions. The average policy is accumulated in a
separate traversal: chance is sampled from its true distribution and opponent
actions use a uniform full-support proposal. Each visited contribution is
weighted by own reach and divided by sampled chance/opponent reach. Its
expectation is the ordinary own-reach average-strategy numerator, up to the
constant number of concrete histories represented by an abstract information
set; that constant cancels when the action masses are normalized.
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field

from pokerlab.research.nlhe.actions import AbstractAction
from pokerlab.research.nlhe.agent import (
    deserialize_information_set,
    serialize_information_set,
)
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.game import AbstractHUNLState
from pokerlab.research.nlhe.information_set import HUNLInformationSet


def average_strategy_importance_weight(
    own_reach: float, chance_reach: float, opponent_proposal_reach: float
) -> float:
    """Importance factor for sampled avg-policy contribution at an infoset."""
    if own_reach < 0 or chance_reach <= 0 or opponent_proposal_reach <= 0:
        raise ValueError(
            "reach probabilities must be nonnegative with positive samples"
        )
    return own_reach / (chance_reach * opponent_proposal_reach)


def average_strategy_reach_weight(
    own_reach: float, opponent_proposal_reach: float
) -> float:
    """Chance-natural, opponent-importance-weighted own-reach factor.

    When chance is sampled from its true prior p_c(h) and opponent actions are
    sampled from a uniform full-support proposal q_{-i}(h), each history h in I
    is visited with probability p_c(h) * q_{-i}(h). Weighting the traverser
    update by own_reach / q_{-i}(h) gives expected increment:

        E[Delta S_i^t(I, a)] = sum_{h in I} p_c(h) * pi_i(h) * sigma_i(I, a)
                             = P_c(I) * pi_i(I) * sigma_i(I, a)

    where P_c(I) = sum_{h in I} p_c(h) > 0 depends only on I (not iteration t
    or action a) and cancels upon action normalization, avoiding both opponent
    strategy reach bias pi_{-i}(I) and 1 / p_c(h) variance explosion.
    """
    if own_reach < 0 or opponent_proposal_reach <= 0:
        raise ValueError(
            "own_reach must be nonnegative and opponent_proposal_reach positive"
        )
    return own_reach / opponent_proposal_reach


@dataclass(frozen=True, slots=True)
class HUNLMCCFRConfig:
    iterations: int = 10
    seed: int = 0
    game: HUNLGameConfig = field(default_factory=HUNLGameConfig)

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")


class ExternalSamplingMCCFR:
    """Two-player, on-demand ES-MCCFR with an unbiased own-reach avg policy."""

    def __init__(self, config: HUNLMCCFRConfig | None = None) -> None:
        self.config = config or HUNLMCCFRConfig()
        self.rng = random.Random(self.config.seed)
        self.regrets: dict[HUNLInformationSet, dict[AbstractAction, float]] = {}
        self._abstract_regrets: dict[tuple, dict[AbstractAction, float]] = {}
        self.strategy_sums: dict[HUNLInformationSet, dict[AbstractAction, float]] = {}
        self.iterations = 0
        self.traversals = 0
        self.sampled_nodes = 0
        self.training_records: list[dict[str, float | int]] = []

    def _board_mode(self) -> str:
        return getattr(self.config.game, "board_abstraction", "exact")

    def _strategy(self, info: HUNLInformationSet) -> dict[AbstractAction, float]:
        actions = info.legal_actions
        values = self.regrets.setdefault(info, dict.fromkeys(actions, 0.0))
        positive = {action: max(values.get(action, 0.0), 0.0) for action in actions}
        total = sum(positive.values())
        if total <= 0 and info.street != "preflop":
            abs_key = info.postflop_abstraction_key(self._board_mode())
            abs_values = self._abstract_regrets.get(abs_key)
            if abs_values is not None:
                positive = {
                    action: max(abs_values.get(action, 0.0), 0.0) for action in actions
                }
                total = sum(positive.values())
        if total <= 0:
            return dict.fromkeys(actions, 1.0 / len(actions))
        return {action: positive[action] / total for action in actions}

    @staticmethod
    def _sample(probabilities: dict[AbstractAction, float], rng: random.Random):
        actions = tuple(probabilities)
        return rng.choices(actions, weights=[probabilities[a] for a in actions], k=1)[0]

    def _profile_snapshot(self):
        return {info: self._strategy(info).copy() for info in self.regrets}

    def _regret_traverse(self, state, traverser, profile, delta, counter) -> float:
        counter[0] += 1
        if state.terminal:
            return state.utility(traverser)
        player = state.current_player
        info = state.information_set(player)
        strategy = profile.get(info)
        if strategy is None:
            strategy = self._strategy(info)
            profile[info] = strategy
        actions = info.legal_actions
        if player == traverser:
            values = {
                action: self._regret_traverse(
                    state.act(action), traverser, profile, delta, counter
                )
                for action in actions
            }
            node_value = sum(strategy[action] * values[action] for action in actions)
            changes = delta.setdefault(info, dict.fromkeys(actions, 0.0))
            for action in actions:
                changes[action] += values[action] - node_value
            return node_value
        action = self._sample(strategy, self.rng)
        return self._regret_traverse(
            state.act(action), traverser, profile, delta, counter
        )

    @staticmethod
    def _chance_reach(state) -> float:
        # Conditional probability of all currently dealt cards under a uniform
        # shuffled deck. Button assignment is fixed for this training game.
        dealt = 52 - state.engine.deck.remaining
        probability = 0.5  # sampled public button assignment
        for remaining in range(52, 52 - dealt, -1):
            probability /= remaining
        return probability

    def _rebuild_abstract_regrets(self) -> None:
        board_mode = self._board_mode()
        abstract: dict[tuple, dict[AbstractAction, float]] = {}
        for info, row in sorted(self.regrets.items(), key=lambda item: repr(item[0])):
            if info.street == "preflop":
                continue
            abs_key = info.postflop_abstraction_key(board_mode)
            target = abstract.setdefault(
                abs_key, dict.fromkeys(info.legal_actions, 0.0)
            )
            for action, value in row.items():
                target[action] = target.get(action, 0.0) + value
        self._abstract_regrets = abstract

    def _average_traverse(
        self,
        state,
        traverser,
        profile,
        delta,
        own_reach,
        opponent_proposal_reach,
        counter,
    ) -> None:
        counter[0] += 1
        if state.terminal:
            return
        player = state.current_player
        info = state.information_set(player)
        strategy = profile.get(info)
        if strategy is None:
            strategy = self._strategy(info)
            profile[info] = strategy
        actions = info.legal_actions
        is_reach_weighted = (
            getattr(self.config.game, "average_strategy_mode", "reach_weighted")
            == "reach_weighted"
        )
        if player == traverser:
            if is_reach_weighted:
                weight = average_strategy_reach_weight(
                    own_reach, opponent_proposal_reach
                )
            else:
                chance_reach = self._chance_reach(state)
                weight = average_strategy_importance_weight(
                    own_reach, chance_reach, opponent_proposal_reach
                )
            sums = delta.setdefault(info, dict.fromkeys(actions, 0.0))
            for action in actions:
                sums[action] += weight * strategy[action]
                self._average_traverse(
                    state.act(action),
                    traverser,
                    profile,
                    delta,
                    own_reach * strategy[action],
                    opponent_proposal_reach,
                    counter,
                )
        else:
            action = self.rng.choice(actions)
            q = 1.0 / len(actions)
            self._average_traverse(
                state.act(action),
                traverser,
                profile,
                delta,
                own_reach,
                opponent_proposal_reach * q,
                counter,
            )

    def run_iteration(self) -> int:
        profile = self._profile_snapshot()
        regret_delta: dict[HUNLInformationSet, dict[AbstractAction, float]] = {}
        average_delta: dict[HUNLInformationSet, dict[AbstractAction, float]] = {}
        counter = [0]
        for player in (0, 1):
            state = AbstractHUNLState.new_hand(
                self.config.game, self.rng, button_index=self.rng.randrange(2)
            )
            # Lazy infosets encountered in either pass get a frozen strategy.
            self._regret_traverse(state, player, profile, regret_delta, counter)
        for info, updates in regret_delta.items():
            target = self.regrets.setdefault(
                info, dict.fromkeys(info.legal_actions, 0.0)
            )
            for action, value in updates.items():
                target[action] = target.get(action, 0.0) + value
        self._rebuild_abstract_regrets()
        post_profile = self._profile_snapshot()
        for player in (0, 1):
            state = AbstractHUNLState.new_hand(
                self.config.game, self.rng, button_index=self.rng.randrange(2)
            )
            self._average_traverse(
                state, player, post_profile, average_delta, 1.0, 1.0, counter
            )
        for info, updates in average_delta.items():
            target = self.strategy_sums.setdefault(
                info, dict.fromkeys(info.legal_actions, 0.0)
            )
            for action, value in updates.items():
                target[action] = target.get(action, 0.0) + value
        self.iterations += 1
        self.traversals += 4
        self.sampled_nodes = counter[0]
        return self.sampled_nodes

    def average_strategy(self, info: HUNLInformationSet) -> dict[AbstractAction, float]:
        actions = info.legal_actions
        masses = self.strategy_sums.get(info, {})
        total = sum(max(0.0, masses.get(action, 0.0)) for action in actions)
        if total <= 0:
            if info in self.regrets:
                return self._strategy(info)
            return dict.fromkeys(actions, 1.0 / len(actions))
        return {action: max(0.0, masses.get(action, 0.0)) / total for action in actions}

    def average_strategy_profile(
        self,
    ) -> dict[HUNLInformationSet, dict[AbstractAction, float]]:
        keys = dict.fromkeys((*self.strategy_sums, *self.regrets))
        return {info: self.average_strategy(info) for info in keys}

    @staticmethod
    def _table_bytes(table) -> int:
        """Approximate resident table size; shared Python objects are overcounted."""
        size = sys.getsizeof(table)
        for info, row in table.items():
            size += sys.getsizeof(info) + sys.getsizeof(row)
            size += sys.getsizeof(info.street) + sys.getsizeof(info.private_bucket)
            size += sys.getsizeof(info.board) + sys.getsizeof(info.history)
            size += sys.getsizeof(info.legal_actions)
            size += sum(sys.getsizeof(event) for event in info.history)
            size += sum(
                sys.getsizeof(action) + sys.getsizeof(value)
                for action, value in row.items()
            )
        return size

    def diagnostics(self) -> dict[str, float | int]:
        regrets = [value for row in self.regrets.values() for value in row.values()]
        positive = sum(max(0.0, value) for value in regrets)
        negative_abs = sum(max(0.0, -value) for value in regrets)
        entropies = []
        for info in self.strategy_sums:
            probabilities = self.average_strategy(info).values()
            entropies.append(-sum(p * math.log(p) for p in probabilities if p > 0))
        return {
            "iterations": self.iterations,
            "traversals": self.traversals,
            "sampled_nodes_last_iteration": self.sampled_nodes,
            "information_sets": len(self.regrets),
            "positive_cumulative_regret": positive,
            "negative_cumulative_regret_abs": negative_abs,
            "cumulative_regret_l1": positive + negative_abs,
            "mean_average_strategy_entropy_nats": (
                sum(entropies) / len(entropies) if entropies else 0.0
            ),
            "regret_table_bytes_estimate": self._table_bytes(self.regrets),
            "strategy_table_bytes_estimate": self._table_bytes(self.strategy_sums),
        }

    def checkpoint_state(self) -> dict:
        def encode_table(table):
            return [
                {
                    "info": serialize_information_set(info),
                    "values": {action.value: value for action, value in row.items()},
                }
                for info, row in sorted(table.items(), key=lambda item: repr(item[0]))
            ]

        return {
            "schema_version": 1,
            "algorithm": "external_sampling_mccfr",
            "seed": self.config.seed,
            "game_and_abstraction": asdict(self.config.game),
            "iterations_completed": self.iterations,
            "traversals": self.traversals,
            "sampled_nodes_last_iteration": self.sampled_nodes,
            "rng_state": self.rng.getstate(),
            "training_records": self.training_records,
            "regrets": encode_table(self.regrets),
            "strategy_sums": encode_table(self.strategy_sums),
        }

    def load_checkpoint(self, payload: dict) -> None:
        """Restore an exact continuation after validating seed and game config."""
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported MCCFR checkpoint schema")
        if payload.get("algorithm") != "external_sampling_mccfr":
            raise ValueError("checkpoint algorithm does not match")
        if payload.get("seed") != self.config.seed:
            raise ValueError("checkpoint seed does not match solver seed")
        if payload.get("game_and_abstraction") != asdict(self.config.game):
            raise ValueError("checkpoint abstraction/game configuration does not match")

        def decode_table(rows):
            decoded = {}
            for row in rows:
                info = deserialize_information_set(row["info"])
                decoded[info] = {
                    AbstractAction(action): float(value)
                    for action, value in row["values"].items()
                }
            return decoded

        def tuple_tree(value):
            return (
                tuple(tuple_tree(item) for item in value)
                if isinstance(value, list)
                else value
            )

        self.regrets = decode_table(payload["regrets"])
        self._rebuild_abstract_regrets()
        self.strategy_sums = decode_table(payload["strategy_sums"])
        self.iterations = int(payload["iterations_completed"])
        self.traversals = int(payload["traversals"])
        self.sampled_nodes = int(payload["sampled_nodes_last_iteration"])
        self.training_records = list(payload["training_records"])
        self.rng.setstate(tuple_tree(payload["rng_state"]))

    def train(self, iterations: int | None = None) -> dict[str, float | int]:
        count = self.config.iterations if iterations is None else iterations
        if count < 1:
            raise ValueError("iterations must be positive")
        started = time.perf_counter()
        for _ in range(count):
            previous_average = {
                info: self.average_strategy(info) for info in self.strategy_sums
            }
            iteration_started = time.perf_counter()
            self.run_iteration()
            average_l1_delta = 0.0
            for info in self.strategy_sums:
                old = previous_average.get(
                    info,
                    dict.fromkeys(info.legal_actions, 1.0 / len(info.legal_actions)),
                )
                new = self.average_strategy(info)
                average_l1_delta += sum(
                    abs(old[action] - new[action]) for action in info.legal_actions
                )
            diagnostics = self.diagnostics()
            self.training_records.append(
                {
                    "iteration": self.iterations,
                    "training_seconds": time.perf_counter() - iteration_started,
                    "sampled_nodes": self.sampled_nodes,
                    "information_sets": diagnostics["information_sets"],
                    "cumulative_regret_l1": diagnostics["cumulative_regret_l1"],
                    "average_strategy_l1_delta": average_l1_delta,
                    "traversals": self.traversals,
                    "positive_cumulative_regret": diagnostics[
                        "positive_cumulative_regret"
                    ],
                    "negative_cumulative_regret_abs": diagnostics[
                        "negative_cumulative_regret_abs"
                    ],
                    "mean_average_strategy_entropy_nats": diagnostics[
                        "mean_average_strategy_entropy_nats"
                    ],
                    "regret_table_bytes_estimate": diagnostics[
                        "regret_table_bytes_estimate"
                    ],
                    "strategy_table_bytes_estimate": diagnostics[
                        "strategy_table_bytes_estimate"
                    ],
                }
            )
        elapsed = time.perf_counter() - started
        return {
            "iterations": self.iterations,
            "training_seconds": elapsed,
            "iterations_per_second": count / elapsed if elapsed else float("inf"),
            "information_sets": len(self.regrets),
            "sampled_nodes_last_iteration": self.sampled_nodes,
        }
