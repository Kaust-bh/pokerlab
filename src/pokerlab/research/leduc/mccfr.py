"""External-sampling Monte Carlo CFR for fixed-limit Leduc Poker."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cards import Card
from pokerlab.research.leduc.evaluation import evaluate_nash
from pokerlab.research.leduc.game import LeducGame, LeducState
from pokerlab.research.leduc.information_sets import InformationSet
from pokerlab.research.leduc.strategy import normalize_average_strategy, regret_matching

LeducProfile = Mapping[InformationSet, Mapping[Action, float]]


@dataclass(frozen=True, slots=True)
class MCCFRConfig:
    iterations: int = 1_000
    seed: int = 0
    evaluation_interval: int = 100
    logging_interval: int = 100

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


def _catalog() -> dict[InformationSet, tuple[Action, ...]]:
    found: dict[InformationSet, tuple[Action, ...]] = {}

    def visit(state: LeducState) -> None:
        if state.is_terminal or state.is_chance_node:
            if state.is_chance_node:
                for card, _ in state.chance_outcomes:
                    visit(state.deal_public(card))
            return
        info = InformationSet.from_state(state)
        found[info] = state.legal_actions
        for action in state.legal_actions:
            visit(state.act(action))

    for deal in LeducGame.deals():
        visit(LeducGame.initial_state(deal))
    return found


def _own_action_sequences(
    information_sets: Mapping[InformationSet, tuple[Action, ...]],
) -> dict[InformationSet, tuple[tuple[InformationSet, Action], ...]]:
    sequences = {}
    for info in information_sets:
        own_actions: list[tuple[InformationSet, Action]] = []
        for index, action in enumerate(info.histories[0]):
            if index % 2 == info.player:
                prefix = info.histories[0][:index]
                previous = InformationSet(
                    info.player,
                    info.private_card,
                    None,
                    0,
                    (prefix, ()),
                )
                own_actions.append((previous, action))
        for index, action in enumerate(info.histories[1]):
            actor = 1 ^ (index % 2)
            if actor == info.player:
                prefix = info.histories[1][:index]
                previous = InformationSet(
                    info.player,
                    info.private_card,
                    info.public_card,
                    1,
                    (info.histories[0], prefix),
                )
                own_actions.append((previous, action))
        sequences[info] = tuple(own_actions)
    return sequences


class MCCFRSolver:
    """Two-player ES-MCCFR with frozen strategies and exact average updates.

    A training iteration samples one private deal per traverser. Within each
    traversal it samples public chance and opponent actions, and enumerates
    every traverser action. For a sampled external block, action-value
    differences are its counterfactual regret contribution. Averaging over the
    true chance and opponent sampling distributions supplies exactly the
    chance/opponent reach weights of vanilla CFR; traverser reach is excluded,
    so regret updates receive no extra probability multiplier.

    Average strategy is separate from the sampled regret estimator. The finite
    information-set catalog is built once, and own action sequences provide
    pi_i(I). Each iteration therefore adds the exact vanilla mass
    pi_i(I) * sigma_i(I,a) for every infoset, including ones omitted by that
    iteration's sampled external trajectory. This avoids a chance/opponent
    visitation bias without adding importance weights to regret updates.
    """

    def __init__(self, config: MCCFRConfig | None = None) -> None:
        self.config = config or MCCFRConfig()
        self._rng = random.Random(self.config.seed)
        self.iterations = 0
        self.regret_sum: dict[InformationSet, dict[Action, float]] = {}
        self.strategy_sum: dict[InformationSet, dict[Action, float]] = {}
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

    def current_strategy(self, info: InformationSet) -> dict[Action, float]:
        return regret_matching(self.regret_sum[info], self._legal[info])

    def current_strategy_profile(self) -> dict[InformationSet, dict[Action, float]]:
        return {info: self.current_strategy(info) for info in self.information_sets}

    def average_strategy(self, info: InformationSet) -> dict[Action, float]:
        return normalize_average_strategy(self.strategy_sum[info], self._legal[info])

    def average_strategy_profile(self) -> dict[InformationSet, dict[Action, float]]:
        return {info: self.average_strategy(info) for info in self.information_sets}

    def _traverse_external(
        self,
        state: LeducState,
        traverser: int,
        profile: LeducProfile,
        regret_delta: dict[InformationSet, dict[Action, float]],
        *,
        opponent_choices: Mapping[InformationSet, Action] | None = None,
        chance_choices: Mapping[LeducState, Card] | None = None,
        node_counter: list[int] | None = None,
    ) -> float:
        if node_counter is not None:
            node_counter[0] += 1
        if state.is_terminal:
            return state.utility(0)
        if state.is_chance_node:
            if chance_choices is not None and state in chance_choices:
                card = chance_choices[state]
                if card not in dict(state.chance_outcomes):
                    raise ValueError("sampled unavailable public card")
            else:
                cards, probabilities = zip(*state.chance_outcomes, strict=True)
                card = self._rng.choices(cards, weights=probabilities, k=1)[0]
            return self._traverse_external(
                state.deal_public(card),
                traverser,
                profile,
                regret_delta,
                opponent_choices=opponent_choices,
                chance_choices=chance_choices,
                node_counter=node_counter,
            )

        player = state.current_player
        if player is None:
            raise RuntimeError("nonterminal Leduc state has no acting player")
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
                    chance_choices=chance_choices,
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
            chance_choices=chance_choices,
            node_counter=node_counter,
        )

    def _average_strategy_delta(
        self, profile: LeducProfile
    ) -> dict[InformationSet, dict[Action, float]]:
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
        """Run one sampled regret pass per player using a frozen profile."""
        profile = self.current_strategy_profile()
        regret_delta: dict[InformationSet, dict[Action, float]] = {}
        node_counter = [0]
        deals = LeducGame.deals()
        for traverser in (0, 1):
            deal = self._rng.choice(deals)
            self._traverse_external(
                LeducGame.initial_state(deal),
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
