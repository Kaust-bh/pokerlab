"""Alternating full-tree CFR+ for Kuhn Poker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pokerlab.research.cfr_variant_utils import (
    CFRVariantRecord,
    regret_matching_plus,
    train_variant,
)
from pokerlab.research.kuhn.cfr import CFRConfig, CFRSolver
from pokerlab.research.kuhn.evaluation import evaluate_nash
from pokerlab.research.kuhn.game import KuhnAction, KuhnGame, KuhnState
from pokerlab.research.kuhn.information_sets import InformationSet

KuhnProfile = Mapping[InformationSet, Mapping[KuhnAction, float]]


@dataclass(frozen=True, slots=True)
class CFRPlusConfig:
    iterations: int = 10_000
    seed: int = 0
    evaluation_interval: int = 100
    logging_interval: int = 1_000

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.evaluation_interval < 1 or self.logging_interval < 1:
            raise ValueError("evaluation and logging intervals must be positive")


class CFRPlusSolver(CFRSolver):
    """CFR+ with alternating player passes and linear strategy averaging.

    One reported iteration consists of a complete Player 0 pass followed by a
    complete Player 1 pass. Player 1's pass uses the strategy after Player 0's
    clipped regrets have been committed. Each pass computes all of that
    player's full-tree regret deltas before applying them. The post-update
    profile contributes iteration-weight ``t`` to the average strategy.
    """

    def __init__(self, config: CFRPlusConfig | None = None) -> None:
        self.config = config or CFRPlusConfig()
        super().__init__(
            CFRConfig(
                self.config.iterations,
                self.config.seed,
                self.config.evaluation_interval,
                self.config.logging_interval,
            )
        )
        self.config = config or CFRPlusConfig()
        self.traversals = 0
        self.player_updates = 0
        self._build_catalog()

    def _build_catalog(self) -> None:
        def visit(state: KuhnState) -> None:
            if state.is_terminal:
                return
            info = InformationSet.from_state(state)
            self._ensure_information_set(info, state.legal_actions)
            for action in state.legal_actions:
                visit(state.act(action))

        for deal in KuhnGame.deals():
            visit(KuhnGame.initial_state(deal))

    def current_strategy(self, info_set: InformationSet) -> dict[KuhnAction, float]:
        return regret_matching_plus(self.regret_sum[info_set], self._legal[info_set])

    def current_strategy_profile(self) -> dict[InformationSet, dict[KuhnAction, float]]:
        return {info: self.current_strategy(info) for info in self._legal}

    def _regret_pass(
        self,
        state: KuhnState,
        traverser: int,
        profile: KuhnProfile,
        reach: tuple[float, float],
        chance_reach: float,
        delta: dict[InformationSet, dict[KuhnAction, float]],
    ) -> float:
        if state.is_terminal:
            return float(state.utility(0))
        player = state.current_player
        if player is None:
            raise RuntimeError("nonterminal Kuhn state has no acting player")
        info = InformationSet.from_state(state)
        actions = state.legal_actions
        strategy = profile[info]
        action_values: dict[KuhnAction, float] = {}
        node_value = 0.0
        for action in actions:
            next_reach = list(reach)
            next_reach[player] *= strategy[action]
            value = self._regret_pass(
                state.act(action),
                traverser,
                profile,
                (next_reach[0], next_reach[1]),
                chance_reach,
                delta,
            )
            action_values[action] = value
            node_value += strategy[action] * value
        if player == traverser:
            sign = 1.0 if player == 0 else -1.0
            updates = delta.setdefault(info, dict.fromkeys(actions, 0.0))
            weight = chance_reach * reach[1 - player]
            for action in actions:
                updates[action] += weight * sign * (action_values[action] - node_value)
        return node_value

    def _average_iteration(
        self, profile: KuhnProfile, weight: float, averaging_player: int
    ) -> None:
        seen: set[InformationSet] = set()

        def visit(state: KuhnState, own_reach: tuple[float, float]) -> None:
            if state.is_terminal:
                return
            actor = state.current_player
            if actor is None:
                raise RuntimeError("nonterminal Kuhn state has no acting player")
            info = InformationSet.from_state(state)
            strategy = profile[info]
            if info.player == averaging_player and info not in seen:
                for action in state.legal_actions:
                    self.strategy_sum[info][action] += (
                        weight * own_reach[averaging_player] * strategy[action]
                    )
                seen.add(info)
            for action in state.legal_actions:
                reach = list(own_reach)
                if actor == averaging_player:
                    reach[averaging_player] *= strategy[action]
                visit(state.act(action), (reach[0], reach[1]))

        for deal in KuhnGame.deals():
            visit(KuhnGame.initial_state(deal), (1.0, 1.0))

    def run_iteration(self) -> None:
        for traverser in (0, 1):
            profile = self.current_strategy_profile()
            delta: dict[InformationSet, dict[KuhnAction, float]] = {}
            deals = list(KuhnGame.deals())
            self._rng.shuffle(deals)
            chance = 1.0 / len(deals)
            for deal in deals:
                self._regret_pass(
                    KuhnGame.initial_state(deal),
                    traverser,
                    profile,
                    (1.0, 1.0),
                    chance,
                    delta,
                )
            for info, actions in delta.items():
                for action, increment in actions.items():
                    self.regret_sum[info][action] = max(
                        0.0, self.regret_sum[info][action] + increment
                    )
            self._average_iteration(profile, self.iterations + 1, traverser)
            self.traversals += 1
            self.player_updates += 1
        self.iterations += 1

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
