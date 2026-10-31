"""External-sampling CFR traversal used to collect Deep CFR targets."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from pokerlab.algorithms.deep_cfr.encoding import GameCodec
from pokerlab.algorithms.deep_cfr.memory import (
    AdvantageMemory,
    AdvantageSample,
    StrategyMemory,
    StrategySample,
)


@dataclass(slots=True)
class TraversalStats:
    nodes: int = 0
    chance_samples: int = 0
    opponent_samples: int = 0
    traverser_decisions: int = 0
    advantage_samples: int = 0
    strategy_samples: int = 0


PolicyFunction = Callable[[int, tuple[float, ...], tuple[bool, ...]], tuple[float, ...]]


class ExternalSamplingTraverser[State, Information, Action]:
    """One player traversal, following Deep CFR Algorithm 2.

    Chance and opponent actions are sampled from their true distributions;
    every traverser action is expanded. The returned per-history advantages
    are the external-sampling estimator described by Brown et al. (2019).
    """

    def __init__(
        self,
        codec: GameCodec[State, Information, Action],
        policy: PolicyFunction,
        rng: random.Random,
    ) -> None:
        self.codec = codec
        self.policy = policy
        self.rng = rng

    def traverse(
        self,
        state: State,
        traverser: int,
        *,
        iteration: int,
        sample_weight: float,
        advantage_memory: AdvantageMemory,
        strategy_memory: StrategyMemory,
    ) -> tuple[float, TraversalStats]:
        stats = TraversalStats()

        def visit(node: State, reach: tuple[float, float]) -> float:
            stats.nodes += 1
            if self.codec.is_terminal(node):
                return self.codec.utility(node, traverser)
            if self.codec.is_chance_node(node):
                outcomes = tuple(self.codec.chance_outcomes(node))
                if not outcomes or any(
                    not math.isfinite(probability) or probability < 0
                    for _, probability in outcomes
                ):
                    raise ValueError("chance nodes need nonnegative outcomes")
                total = sum(probability for _, probability in outcomes)
                if total <= 0 or abs(total - 1.0) > 1e-9:
                    raise ValueError("chance probabilities must sum to one")
                outcome = self.rng.choices(
                    [item for item, _ in outcomes],
                    weights=[probability for _, probability in outcomes],
                    k=1,
                )[0]
                stats.chance_samples += 1
                return visit(self.codec.deal(node, outcome), reach)

            player = self.codec.current_player(node)
            info = self.codec.information_set(node)
            features = self.codec.encode(info)
            actions = self.codec.legal_actions(node)
            if not actions:
                raise RuntimeError("decision node has no legal actions")
            action_indices = tuple(
                self.codec.action_index(action) for action in actions
            )
            if len(set(action_indices)) != len(action_indices):
                raise RuntimeError("legal actions map to duplicate network outputs")
            mask = tuple(
                index in action_indices
                for index in range(len(self.codec.action_vocabulary))
            )
            probabilities = self.policy(player, features, mask)
            if len(probabilities) != len(mask):
                raise ValueError("policy output does not match action vocabulary")
            if any(not math.isfinite(value) for value in probabilities):
                raise ValueError("policy probabilities must be finite")
            legal_probabilities = [probabilities[index] for index in action_indices]
            if any(value < 0 for value in legal_probabilities):
                raise ValueError("policy returned a negative legal-action probability")
            probability_sum = sum(legal_probabilities)
            if abs(probability_sum - 1.0) > 1e-6:
                raise ValueError(
                    "policy probabilities must normalize over legal actions"
                )

            if player == traverser:
                stats.traverser_decisions += 1
                action_values = {}
                for action, probability in zip(
                    actions, legal_probabilities, strict=True
                ):
                    next_reach = list(reach)
                    next_reach[player] *= probability
                    action_values[action] = visit(
                        self.codec.act(node, action), tuple(next_reach)
                    )
                node_value = sum(
                    probabilities[index] * action_values[action]
                    for action, index in zip(actions, action_indices, strict=True)
                )
                targets = [0.0] * len(mask)
                for action, index in zip(actions, action_indices, strict=True):
                    targets[index] = action_values[action] - node_value
                advantage_memory.add(
                    AdvantageSample(
                        player=player,
                        features=features,
                        targets=tuple(targets),
                        legal_mask=mask,
                        iteration=iteration,
                        sample_weight=sample_weight,
                    )
                )
                stats.advantage_samples += 1
                return node_value

            strategy_sample_weight = sample_weight * reach[traverser]
            if strategy_sample_weight > 0:
                strategy_memory.add(
                    StrategySample(
                        player=player,
                        features=features,
                        targets=tuple(probabilities),
                        legal_mask=mask,
                        iteration=iteration,
                        sample_weight=strategy_sample_weight,
                    )
                )
                stats.strategy_samples += 1
            stats.opponent_samples += 1
            action = self.rng.choices(actions, weights=legal_probabilities, k=1)[0]
            next_reach = list(reach)
            next_reach[player] *= probabilities[self.codec.action_index(action)]
            return visit(self.codec.act(node, action), tuple(next_reach))

        return visit(state, (1.0, 1.0)), stats
