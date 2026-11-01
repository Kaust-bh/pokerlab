"""Tournament-compatible policy backed by an MCCFR average strategy."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping

from pokerlab.agents.base import Action, ActionType, Agent, Observation
from pokerlab.agents.heuristic_agent import HeuristicAgent
from pokerlab.research.nlhe.actions import (
    ActionSpec,
    abstract_action_specs_from_observation,
)
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.information_set import (
    HUNLInformationSet,
    information_set_from_observation,
)


def _aggregate_index(
    strategy: Mapping[HUNLInformationSet, Mapping],
    key_fn,
    strategy_sums: Mapping[HUNLInformationSet, Mapping] | None = None,
) -> dict[tuple, dict]:
    accum: dict[tuple, dict] = {}
    for info, probs in strategy.items():
        key = key_fn(info)
        if key not in accum:
            accum[key] = {a: 0.0 for a in probs}
        weight = 1.0
        if strategy_sums is not None and info in strategy_sums:
            raw_mass = sum(max(0.0, float(v)) for v in strategy_sums[info].values())
            if raw_mass > 0:
                weight = raw_mass
        for a, p in probs.items():
            accum[key][a] = accum[key].get(a, 0.0) + weight * max(0.0, float(p))
    for key, mass_dict in accum.items():
        total = sum(mass_dict.values())
        if total > 0:
            for a in mass_dict:
                mass_dict[a] /= total
    return accum


class AbstractHUNLAgent(Agent):
    def __init__(
        self,
        strategy: Mapping[HUNLInformationSet, Mapping],
        config: HUNLGameConfig,
        seed: int = 0,
        *,
        strategy_sums: Mapping[HUNLInformationSet, Mapping] | None = None,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self.rng = random.Random(seed)
        self.decisions = 0
        self.exact_hits = 0
        self.generalized_hits = 0
        self.fallback_hits = 0
        self.decision_seconds = 0.0
        board_mode = getattr(config, "board_abstraction", "exact")

        # Hierarchical generalization tables weighted by strategy-sum mass when available:
        # 1. Exact bucket + history across boards
        self._generalized_strategy = _aggregate_index(
            strategy, lambda info: info.generalized_key(), strategy_sums
        )
        # 2. Board-aware postflop bucket + street context
        self._board_generalized_strategy = _aggregate_index(
            strategy,
            lambda info: info.postflop_abstraction_key(board_mode),
            strategy_sums,
        )
        # 3. Coarse board-agnostic postflop bucket + street context
        self._coarse_generalized_strategy = _aggregate_index(
            strategy, lambda info: info.coarse_generalized_key(), strategy_sums
        )

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def lookup_rates(self) -> dict[str, float | int]:
        total = max(self.decisions, 1)
        return {
            "decisions": self.decisions,
            "exact_hits": self.exact_hits,
            "generalized_hits": self.generalized_hits,
            "fallback_hits": self.fallback_hits,
            "exact_rate": self.exact_hits / total,
            "generalized_rate": self.generalized_hits / total,
            "fallback_rate": self.fallback_hits / total,
        }

    def _bucket_decision_strength(
        self, observation: Observation, info: HUNLInformationSet
    ) -> float:
        pos_bonus = 0.03 if observation.in_position else 0.0
        if not observation.community_cards:
            return min(1.0, HeuristicAgent._preflop_strength(observation) + pos_bonus)
        post_heuristic, _has_draw = HeuristicAgent._postflop_strength(observation)
        bucket_pct = post_heuristic
        for token in info.private_bucket.split("|"):
            if token.startswith("equity:"):
                fields = token.split(":")
                if len(fields) == 4:
                    buckets = max(int(fields[1]), 1)
                    idx = int(fields[3])
                    bucket_pct = (idx + 0.5) / buckets
                break
            if token.startswith("relative:"):
                idx = int(token.split(":")[1])
                buckets = max(self.config.postflop_buckets, 1)
                bucket_pct = (idx + 0.5) / buckets
                break
        # Blend abstract bucket rank (70%) with made-hand/draw floor (30%)
        return min(1.0, 0.70 * bucket_pct + 0.30 * post_heuristic + pos_bonus)

    def _tactical_fallback(
        self,
        observation: Observation,
        specs: tuple[ActionSpec, ...],
        info: HUNLInformationSet | None = None,
    ) -> ActionSpec:
        """Card-aware and bucket-aware fallback when an observation is unseen."""
        check_spec = next((s for s in specs if s.engine.type == ActionType.CHECK), None)
        call_spec = next((s for s in specs if s.engine.type == ActionType.CALL), None)
        fold_spec = next((s for s in specs if s.engine.type == ActionType.FOLD), None)
        bet_specs = [
            s for s in specs if s.engine.type in (ActionType.BET, ActionType.RAISE)
        ]

        if info is None:
            info = information_set_from_observation(observation, (), self.config)
        strength = self._bucket_decision_strength(observation, info)

        if observation.to_call == 0:
            if strength >= 0.72 and bet_specs:
                return bet_specs[0]
            return (
                check_spec
                if check_spec is not None
                else (bet_specs[0] if bet_specs else specs[0])
            )

        call_cost = observation.to_call / max(1, observation.effective_stack)
        if strength >= 0.85 and bet_specs and call_cost < 0.50:
            return bet_specs[0]
        if strength >= observation.pot_odds + 0.05 and call_spec is not None:
            return call_spec
        return fold_spec if fold_spec is not None else (call_spec or specs[0])

    def _resolve_policy(
        self, observation: Observation
    ) -> tuple[tuple[ActionSpec, ...], list[float], str]:
        history = tuple(
            (
                record.street,
                record.player == observation.player
                if observation.is_button
                else record.player != observation.player,
                record.action.type.value,
                record.action.amount,
            )
            for record in observation.history
        )
        info = information_set_from_observation(observation, history, self.config)
        specs = abstract_action_specs_from_observation(observation)
        board_mode = getattr(self.config, "board_abstraction", "exact")

        probabilities = self.strategy.get(info)
        source = "exact"
        if probabilities is None:
            source = "generalized"
            probabilities = self._generalized_strategy.get(info.generalized_key())
            if probabilities is None:
                probabilities = self._board_generalized_strategy.get(
                    info.postflop_abstraction_key(board_mode)
                )
            if probabilities is None:
                probabilities = self._coarse_generalized_strategy.get(
                    info.coarse_generalized_key()
                )

        if probabilities is not None:
            weights = [
                max(0.0, float(probabilities.get(spec.abstract, 0.0))) for spec in specs
            ]
            if sum(weights) > 0:
                return specs, weights, source

        fallback_spec = self._tactical_fallback(observation, specs, info)
        weights = [1.0 if spec == fallback_spec else 0.0 for spec in specs]
        return specs, weights, "fallback"

    def action_probabilities(self, observation: Observation) -> dict[Action, float]:
        specs, weights, _source = self._resolve_policy(observation)
        total = sum(weights)
        dist = {action: 0.0 for action in observation.legal_actions}
        if total > 0:
            for spec, w in zip(specs, weights, strict=True):
                dist[spec.engine] = dist.get(spec.engine, 0.0) + w / total
        return dist

    def act(self, observation: Observation):
        started = time.perf_counter()
        specs, weights, source = self._resolve_policy(observation)
        if source == "exact":
            self.exact_hits += 1
        elif source == "generalized":
            self.generalized_hits += 1
        else:
            self.fallback_hits += 1
        action_spec = self.rng.choices(specs, weights=weights, k=1)[0]
        self.decisions += 1
        self.decision_seconds += time.perf_counter() - started
        return action_spec.engine


def serialize_information_set(info: HUNLInformationSet) -> dict:
    return {
        "player": info.player,
        "street": info.street,
        "private_bucket": info.private_bucket,
        "board": info.board,
        "pot": info.pot,
        "effective_stack": info.effective_stack,
        "street_contributions": info.street_contributions,
        "history": info.history,
        "legal_actions": [action.value for action in info.legal_actions],
    }


def deserialize_information_set(data: dict) -> HUNLInformationSet:
    from pokerlab.research.nlhe.actions import AbstractAction

    return HUNLInformationSet(
        player=data["player"],
        street=data["street"],
        private_bucket=data["private_bucket"],
        board=tuple(tuple(card) for card in data["board"]),
        pot=data["pot"],
        effective_stack=data["effective_stack"],
        street_contributions=tuple(data["street_contributions"]),
        history=tuple(tuple(event) for event in data["history"]),
        legal_actions=tuple(AbstractAction(action) for action in data["legal_actions"]),
    )


def serialize_strategy(strategy: Mapping[HUNLInformationSet, Mapping]) -> dict:
    """Convert stable info keys and action masses to JSON-compatible objects."""
    rows = []
    for info, probabilities in sorted(strategy.items(), key=lambda item: repr(item[0])):
        rows.append(
            {
                "info": serialize_information_set(info),
                "strategy": {
                    action.value: value for action, value in probabilities.items()
                },
            }
        )
    return {"information_sets": rows}


def deserialize_strategy(payload: dict):
    from pokerlab.research.nlhe.actions import AbstractAction

    strategy = {}
    for row in payload["information_sets"]:
        info = deserialize_information_set(row["info"])
        strategy[info] = {
            AbstractAction(action): float(value)
            for action, value in row["strategy"].items()
        }
    return strategy
