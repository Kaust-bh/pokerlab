"""Experimental private-deal stratification and enumeration for Kuhn MCCFR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pokerlab.research.kuhn.game import KuhnAction, KuhnCard, KuhnGame, KuhnState
from pokerlab.research.kuhn.information_sets import InformationSet
from pokerlab.research.kuhn.mccfr import (
    KuhnProfile,
    MCCFRConfig,
    MCCFRSolver,
)

PrivateDealMode = Literal["sample", "rank_stratified", "enumerate"]


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    iterations: int = 1_000
    seed: int = 0
    evaluation_interval: int = 100
    logging_interval: int = 100
    private_deal_mode: PrivateDealMode = "sample"
    samples_per_stratum: int = 1

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.evaluation_interval < 1 or self.logging_interval < 1:
            raise ValueError("evaluation and logging intervals must be positive")
        if self.private_deal_mode not in ("sample", "rank_stratified", "enumerate"):
            raise ValueError("unsupported private-deal sampling mode")
        if self.samples_per_stratum < 1:
            raise ValueError("samples_per_stratum must be positive")


class SamplingVariantSolver(MCCFRSolver):
    """Kuhn ES-MCCFR with a separately configurable chance-deal estimator.

    Rank stratification divides chance outcomes into P0-King (mass 1/3) and
    P0-J/Q (mass 2/3). One or more deals are sampled conditionally within each
    class, with each traversal weighted by stratum_mass / samples_per_stratum.
    Exact private-deal enumeration instead weights each of the six ordered
    deals by 1/6. Opponent actions remain sampled from the frozen profile.
    """

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.sampling_config = config or SamplingConfig()
        super().__init__(
            MCCFRConfig(
                self.sampling_config.iterations,
                self.sampling_config.seed,
                self.sampling_config.evaluation_interval,
                self.sampling_config.logging_interval,
            )
        )

    @staticmethod
    def _private_strata():
        deals = tuple(KuhnGame.deals())
        king = tuple(deal for deal in deals if deal[0] is KuhnCard.KING)
        other = tuple(deal for deal in deals if deal[0] is not KuhnCard.KING)
        return ((1 / 3, king), (2 / 3, other))

    def _private_deal_samples(self) -> list[tuple[tuple[KuhnCard, KuhnCard], float]]:
        deals = tuple(KuhnGame.deals())
        mode = self.sampling_config.private_deal_mode
        if mode == "sample":
            return [(self._rng.choice(deals), 1.0)]
        if mode == "enumerate":
            return [(deal, 1.0 / len(deals)) for deal in deals]
        samples = self.sampling_config.samples_per_stratum
        return [
            (self._rng.choice(stratum), mass / samples)
            for mass, stratum in self._private_strata()
            for _ in range(samples)
        ]

    def _traverse_variant(
        self,
        state: KuhnState,
        traverser: int,
        profile: KuhnProfile,
        regret_delta: dict[InformationSet, dict[KuhnAction, float]],
        *,
        chance_weight: float,
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
                action: self._traverse_variant(
                    state.act(action),
                    traverser,
                    profile,
                    regret_delta,
                    chance_weight=chance_weight,
                    opponent_choices=opponent_choices,
                    node_counter=node_counter,
                )
                for action in actions
            }
            node_value = sum(strategy[action] * values[action] for action in actions)
            sign = 1.0 if traverser == 0 else -1.0
            updates = regret_delta.setdefault(info, dict.fromkeys(actions, 0.0))
            for action in actions:
                updates[action] += chance_weight * sign * (values[action] - node_value)
            return node_value

        if opponent_choices is not None and info in opponent_choices:
            action = opponent_choices[info]
            if action not in actions:
                raise ValueError(f"sampled illegal action at {info.identifier}")
        else:
            action = self._rng.choices(
                actions, weights=[strategy[item] for item in actions], k=1
            )[0]
        return self._traverse_variant(
            state.act(action),
            traverser,
            profile,
            regret_delta,
            chance_weight=chance_weight,
            opponent_choices=opponent_choices,
            node_counter=node_counter,
        )

    def sample_regret_traversal(
        self, traverser: int, profile: KuhnProfile
    ) -> tuple[dict[InformationSet, dict[KuhnAction, float]], int]:
        """Return one unbiased sampled external-block estimate for a player."""
        if traverser not in (0, 1):
            raise ValueError("traverser must be 0 or 1")
        delta: dict[InformationSet, dict[KuhnAction, float]] = {}
        nodes = [0]
        for deal, weight in self._private_deal_samples():
            self._traverse_variant(
                KuhnGame.initial_state(deal),
                traverser,
                profile,
                delta,
                chance_weight=weight,
                node_counter=nodes,
            )
        return delta, nodes[0]

    def run_iteration(self) -> int:
        profile = self.current_strategy_profile()
        regret_delta: dict[InformationSet, dict[KuhnAction, float]] = {}
        node_count = 0
        for traverser in (0, 1):
            estimate, nodes = self.sample_regret_traversal(traverser, profile)
            node_count += nodes
            for info, actions in estimate.items():
                target = regret_delta.setdefault(
                    info, dict.fromkeys(self._legal[info], 0.0)
                )
                for action, value in actions.items():
                    target[action] += value

        average_delta = self._average_strategy_delta(profile)
        for info, actions in regret_delta.items():
            for action, value in actions.items():
                self.regret_sum[info][action] += value
        for info, actions in average_delta.items():
            for action, value in actions.items():
                self.strategy_sum[info][action] += value
        self.iterations += 1
        self.sampled_nodes = node_count
        return node_count
