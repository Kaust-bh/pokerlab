"""Adaptive MCCFR Controller (HarveySpecterAgent) and controlled archetype opponents."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

from pokerlab.agents.base import Action, ActionRecord, ActionType, Agent, Observation
from pokerlab.agents.heuristic_agent import HeuristicAgent
from pokerlab.engine import Card
from pokerlab.opponent_model.exploit import ExploitativeStrategyEngine
from pokerlab.opponent_model.models import (
    ExploitDecisionReport,
    OpponentStatsSnapshot,
)
from pokerlab.opponent_model.profile_store import (
    OpponentProfile,
    OpponentProfileStore,
)
from pokerlab.research.nlhe.agent import AbstractHUNLAgent
from pokerlab.research.nlhe.information_set import (
    HUNLInformationSet,
    information_set_from_observation,
)


def _format_action(action: Action) -> str:
    kind = ActionType(action.type).value
    if kind in (ActionType.BET.value, ActionType.RAISE.value) and action.amount:
        return f"{kind} {action.amount}"
    return kind.replace("_", "-")


def _resolve_subtier_name(
    agent: AbstractHUNLAgent, info: HUNLInformationSet
) -> tuple[str, str]:
    board_mode = getattr(agent.config, "board_abstraction", "exact")
    if info in agent.strategy:
        return "exact", repr(info)
    gen_key = info.generalized_key()
    if gen_key in agent._generalized_strategy:
        return "generalized_key", repr(gen_key)
    board_key = info.postflop_abstraction_key(board_mode)
    if board_key in agent._board_generalized_strategy:
        return "board_abstraction_key", repr(board_key)
    coarse_key = info.coarse_generalized_key()
    if coarse_key in agent._coarse_generalized_strategy:
        return "coarse_generalized_key", repr(coarse_key)
    return "tactical_fallback", repr(coarse_key)


@dataclass(frozen=True, slots=True)
class HarveySpecterConfig:
    """Configuration for HarveySpecter's adaptive MCCFR + exploitative policy mixture."""

    mode: str = "adaptive"  # "adaptive", "fixed", "baseline_only", "pure_exploit"
    fixed_lambda: float = 0.50
    max_lambda: float = 0.80
    min_hands_for_exploit: int = 2
    cold_start_scale_hands: float = 16.0
    recency_decay: float = 0.94
    equity_simulations: int = 44
    preserve_profile_on_reset: bool = True

    def __post_init__(self) -> None:
        valid_modes = {"adaptive", "fixed", "baseline_only", "pure_exploit"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode {self.mode!r}; expected one of {sorted(valid_modes)}"
            )
        if not 0.0 <= self.fixed_lambda <= 1.0:
            raise ValueError("fixed_lambda must be in [0.0, 1.0]")
        if not 0.0 <= self.max_lambda <= 1.0:
            raise ValueError("max_lambda must be in [0.0, 1.0]")


class HarveySpecterAgent(Agent):
    """Adaptive opponent-modeling agent wrapping an `AbstractHUNLAgent` (MCCFR baseline).

    Mixes:
        pi_final(a) = (1 - lambda) * pi_MCCFR(a) + lambda * pi_exploit(a)
    strictly over `observation.legal_actions`.
    """

    def __init__(
        self,
        baseline_agent: AbstractHUNLAgent,
        *,
        config: HarveySpecterConfig | None = None,
        opponent_id: str = "opponent",
        profile_store: OpponentProfileStore | None = None,
        seed: int = 0,
    ) -> None:
        self.baseline_agent = baseline_agent
        self.config = config or HarveySpecterConfig()
        self.default_opponent_id = opponent_id.strip() or "opponent"
        self.profile_store = profile_store or OpponentProfileStore(
            recency_decay=self.config.recency_decay
        )
        self.exploit_engine = ExploitativeStrategyEngine(
            equity_simulations=self.config.equity_simulations
        )
        self.rng = random.Random(seed)
        self._seed = seed

        self.decisions = 0
        self.decision_seconds = 0.0
        self.lambda_history: list[float] = []
        self.last_decision_report: ExploitDecisionReport | None = None

    @property
    def profile(self) -> OpponentProfile:
        return self.profile_store.get_or_create(self.default_opponent_id)

    def get_profile(self, opponent_id: str | None = None) -> OpponentProfile:
        return self.profile_store.get_or_create(opponent_id or self.default_opponent_id)

    def set_opponent(self, opponent_id: str) -> OpponentProfile:
        self.default_opponent_id = opponent_id.strip() or "opponent"
        return self.profile

    def reset_opponent_profile(self, opponent_id: str | None = None) -> OpponentProfile:
        return self.profile_store.reset_profile(opponent_id or self.default_opponent_id)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            self.rng.seed(seed)
            self.baseline_agent.reset(seed)
        if not self.config.preserve_profile_on_reset:
            self.reset_opponent_profile(self.default_opponent_id)

    def start_hand(
        self,
        *,
        opponent_id: str | None = None,
        hand_number: int | None = None,
        opponent_is_button: bool,
        starting_stack: int = 200,
        effective_stack: int | None = None,
        big_blind: int = 2,
        hero_hole: Sequence[Card] = (),
        community_cards: Sequence[Card] = (),
    ) -> None:
        prof = self.get_profile(opponent_id)
        prof.start_hand(
            hand_number=hand_number,
            opponent_is_button=opponent_is_button,
            starting_stack=starting_stack,
            effective_stack=effective_stack,
            big_blind=big_blind,
            hero_hole=hero_hole,
            community_cards=community_cards,
        )

    def finalize_hand(
        self,
        *,
        opponent_id: str | None = None,
        hand_number: int | None = None,
        history: Sequence[ActionRecord] | None = None,
        hero_player: str | None = None,
        community_cards: Sequence[Card] | None = None,
        reached_showdown: bool = False,
        revealed_opponent_cards: tuple[Card, Card] | None = None,
        opponent_net_chips: int = 0,
        opponent_won: bool | None = None,
        final_pot: int | None = None,
    ) -> None:
        prof = self.get_profile(opponent_id)
        prof.finalize_hand(
            hand_number=hand_number,
            history=history,
            hero_player=hero_player,
            community_cards=community_cards,
            reached_showdown=reached_showdown,
            revealed_opponent_cards=revealed_opponent_cards,
            opponent_net_chips=opponent_net_chips,
            opponent_won=opponent_won,
            final_pot=final_pot,
        )

    def compute_adaptive_lambda(
        self,
        stats: OpponentStatsSnapshot,
        archetype_confidence: float,
        archetype_probs: dict[str, float],
        lookup_tier: str,
    ) -> tuple[float, float, float]:
        """Compute `(lambda_value, evidence_confidence, uncertainty)` in [0, 1]."""
        mode = self.config.mode
        n_hands = stats.hands_observed
        w_trials = (
            stats.vpip.weighted_trials
            + 0.5 * stats.aggression_frequency.weighted_trials
        )
        sample_conf = (
            w_trials / (w_trials + self.config.cold_start_scale_hands)
            if w_trials > 0.0
            else 0.0
        )
        stability = stats.stability_score

        # How far opponent tendencies deviate from a balanced baseline
        balanced_mass = archetype_probs.get("balanced_unknown", 0.58)
        exploitability_signal = max(
            0.0,
            min(
                1.0,
                (1.0 - balanced_mass) * 0.75
                + abs(stats.fold_to_bet.smoothed_value - 0.42) * 0.9
                + abs(stats.vpip.smoothed_value - 0.46) * 0.6,
            ),
        )
        evidence_confidence = min(
            0.99,
            sample_conf
            * (0.55 + 0.45 * max(archetype_confidence, exploitability_signal))
            * stability,
        )
        uncertainty = 1.0 - evidence_confidence

        if mode == "baseline_only":
            return 0.0, evidence_confidence, uncertainty
        if mode == "pure_exploit":
            return 1.0, evidence_confidence, uncertainty
        if mode == "fixed":
            return self.config.fixed_lambda, evidence_confidence, uncertainty

        # Adaptive mode:
        # 0 hands -> lambda = 0.0 (pure MCCFR cold start)
        if n_hands == 0 and stats.aggression_frequency.trials == 0:
            return 0.0, 0.0, 1.0
        if (
            n_hands < self.config.min_hands_for_exploit
            and stats.aggression_frequency.trials < 2
        ):
            lam = min(0.10, 0.05 * n_hands) * stability
            return lam, evidence_confidence, uncertainty

        raw_lambda = (
            self.config.max_lambda
            * sample_conf
            * (0.45 + 0.55 * max(archetype_confidence, exploitability_signal))
            * stability
        )
        # If MCCFR had to fall back to tactical heuristic on an unseen infoset and we have
        # meaningful opponent evidence, allow slightly higher reliance on the range-grounded model
        if lookup_tier == "fallback" and sample_conf >= 0.35:
            raw_lambda = min(
                self.config.max_lambda,
                raw_lambda + 0.12 * sample_conf * stability,
            )

        lam = max(0.0, min(self.config.max_lambda, raw_lambda))
        return lam, evidence_confidence, uncertainty

    def evaluate_decision(
        self,
        observation: Observation,
        *,
        opponent_id: str | None = None,
        hand_number: int | None = None,
        ingest: bool = True,
        rng_for_equity: random.Random | None = None,
    ) -> tuple[dict[Action, float], ExploitDecisionReport]:
        """Compute `pi_MCCFR`, `pi_exploit`, `pi_final`, and full telemetry report."""
        legal = observation.legal_actions
        if not legal:
            raise ValueError("Observation has no legal actions")

        prof = self.get_profile(opponent_id)
        if ingest:
            record, _ = prof.ingest_observation(observation, hand_number=hand_number)
            current_hand_num = record.hand_number
        else:
            prof.range_model.condition_on_cards(
                hero_hole=observation.hole_cards,
                community_cards=observation.community_cards,
            )
            current_hand_num = hand_number or prof.last_hand_number

        # 1. Baseline MCCFR policy over `observation.legal_actions` (single infoset construction)
        history_tuples = tuple(
            (
                rec.street,
                rec.player == observation.player
                if observation.is_button
                else rec.player != observation.player,
                ActionType(rec.action.type).value,
                rec.action.amount,
            )
            for rec in observation.history
        )
        info = information_set_from_observation(
            observation, history_tuples, self.baseline_agent.config
        )
        subtier, _ = _resolve_subtier_name(self.baseline_agent, info)
        from pokerlab.research.nlhe.actions import (
            abstract_action_specs_from_observation,
        )

        specs = abstract_action_specs_from_observation(observation)
        board_mode = getattr(self.baseline_agent.config, "board_abstraction", "exact")
        probabilities = self.baseline_agent.strategy.get(info)
        lookup_tier = "exact"
        if probabilities is None:
            lookup_tier = "generalized"
            probabilities = self.baseline_agent._generalized_strategy.get(
                info.generalized_key()
            )
            if probabilities is None:
                probabilities = self.baseline_agent._board_generalized_strategy.get(
                    info.postflop_abstraction_key(board_mode)
                )
            if probabilities is None:
                probabilities = self.baseline_agent._coarse_generalized_strategy.get(
                    info.coarse_generalized_key()
                )
        weights: list[float] = []
        if probabilities is not None:
            weights = [
                max(0.0, float(probabilities.get(spec.abstract, 0.0))) for spec in specs
            ]
        if not weights or sum(weights) <= 0:
            fallback_spec = self.baseline_agent._tactical_fallback(
                observation, specs, info
            )
            weights = [1.0 if spec == fallback_spec else 0.0 for spec in specs]
            lookup_tier = "fallback"
        total_w = sum(weights)
        mccfr_dist: dict[Action, float] = {act: 0.0 for act in legal}
        if total_w > 0:
            for spec, w in zip(specs, weights, strict=True):
                if spec.engine in mccfr_dist:
                    mccfr_dist[spec.engine] += w / total_w
        mccfr_sum = sum(mccfr_dist.values())
        if mccfr_sum > 0:
            for act in legal:
                mccfr_dist[act] /= mccfr_sum
        else:
            for act in legal:
                mccfr_dist[act] = 1.0 / len(legal)

        # 2. Opponent statistics, archetype classification, and Bayesian range snapshot
        stats = prof.stats_snapshot()
        archetype = prof.archetype_distribution()
        range_snap = prof.range_snapshot()

        # 3. Exploitative strategy distribution over `observation.legal_actions`
        eval_rng = rng_for_equity or random.Random(
            (self._seed * 1_000_003)
            ^ (current_hand_num * 97_531)
            ^ (len(observation.history) * 7_919)
        )
        exploit_eval = self.exploit_engine.evaluate(
            observation,
            stats,
            archetype,
            prof.range_model,
            range_snap,
            rng=eval_rng,
        )
        exploit_dist: dict[Action, float] = {
            act: max(0.0, float(exploit_eval.probabilities.get(act, 0.0)))
            for act in legal
        }
        exp_sum = sum(exploit_dist.values())
        if exp_sum > 0:
            for act in legal:
                exploit_dist[act] /= exp_sum
        else:
            for act in legal:
                exploit_dist[act] = 1.0 / len(legal)

        # 4. Adaptive exploitation coefficient lambda and final mixture
        lam, ev_conf, uncertainty = self.compute_adaptive_lambda(
            stats,
            archetype.confidence,
            archetype.probabilities,
            lookup_tier,
        )

        final_dist: dict[Action, float] = {}
        for act in legal:
            final_dist[act] = (1.0 - lam) * mccfr_dist[act] + lam * exploit_dist[act]
        final_sum = sum(final_dist.values())
        if final_sum > 0:
            for act in legal:
                final_dist[act] /= final_sum
        else:
            for act in legal:
                final_dist[act] = 1.0 / len(legal)

        mccfr_rows = [
            {
                "action_type": ActionType(act.type).value,
                "amount": act.amount,
                "label": _format_action(act),
                "probability": round(mccfr_dist[act], 6),
            }
            for act in legal
        ]
        exploit_rows = [
            {
                "action_type": ActionType(act.type).value,
                "amount": act.amount,
                "label": _format_action(act),
                "probability": round(exploit_dist[act], 6),
            }
            for act in legal
        ]
        final_rows = [
            {
                "action_type": ActionType(act.type).value,
                "amount": act.amount,
                "label": _format_action(act),
                "probability": round(final_dist[act], 6),
            }
            for act in legal
        ]

        report = ExploitDecisionReport(
            opponent_id=prof.opponent_id,
            hand_number=current_hand_num,
            street=observation.street,
            lambda_value=lam,
            lambda_mode=self.config.mode,
            max_lambda=self.config.max_lambda,
            evidence_confidence=ev_conf,
            uncertainty=uncertainty,
            stability_score=stats.stability_score,
            lookup_tier=lookup_tier,
            lookup_subtier=subtier,
            hero_range_equity=exploit_eval.hero_range_equity,
            hero_uniform_equity=exploit_eval.hero_uniform_equity,
            estimated_fold_prob=exploit_eval.estimated_fold_prob,
            pot_odds=exploit_eval.pot_odds,
            mccfr_probabilities=mccfr_rows,
            exploit_probabilities=exploit_rows,
            final_probabilities=final_rows,
            action_expected_values=exploit_eval.action_evs,
            adjustment_reasons=exploit_eval.adjustment_reasons,
            explanation=exploit_eval.explanation,
            opponent_stats=stats,
            archetype=archetype,
            range_snapshot=range_snap,
        )
        return final_dist, report

    def action_probabilities(self, observation: Observation) -> dict[Action, float]:
        """Return non-mutating final action probability distribution over `legal_actions`."""
        dist, _ = self.evaluate_decision(observation, ingest=False)
        return dist

    def act(self, observation: Observation) -> Action:
        """Observe public history, compute mixed strategy `pi_final`, and sample a legal action."""
        started = time.perf_counter()
        final_dist, report = self.evaluate_decision(observation, ingest=True)
        legal = observation.legal_actions

        # In baseline_only mode (lambda == 0.0), delegate directly to baseline_agent
        # or sample from the identical distribution using self.rng
        weights = [max(0.0, final_dist[act]) for act in legal]
        if sum(weights) <= 0.0:
            chosen = legal[0]
        else:
            chosen = self.rng.choices(legal, weights=weights, k=1)[0]

        for row in report.final_probabilities:
            row["is_chosen"] = (
                row["action_type"] == ActionType(chosen.type).value
                and row["amount"] == chosen.amount
            )

        self.lambda_history.append(report.lambda_value)
        self.last_decision_report = report
        self.decisions += 1
        self.decision_seconds += time.perf_counter() - started
        return chosen


# ============================================================================
# Controlled Archetype Opponent Agents for Evaluation & Benchmarking
# ============================================================================


def _find_action(actions: Sequence[Action], kind: ActionType) -> Action | None:
    return next((a for a in actions if ActionType(a.type) == kind), None)


def _find_all_actions(
    actions: Sequence[Action], kinds: set[ActionType]
) -> list[Action]:
    return [a for a in actions if ActionType(a.type) in kinds]


class NitAgent(Agent):
    """Ultra-tight opponent: folds most hands preflop and postflop unless holding premium strength."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._seed = seed

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            self.rng.seed(seed)

    def act(self, observation: Observation) -> Action:
        legal = observation.legal_actions
        if not legal:
            raise ValueError("No legal actions")
        strength = (
            HeuristicAgent._preflop_strength(observation)
            if not observation.community_cards
            else HeuristicAgent._postflop_strength(observation)[0]
        )
        fold = _find_action(legal, ActionType.FOLD)
        check = _find_action(legal, ActionType.CHECK)
        call = _find_action(legal, ActionType.CALL)
        bets = _find_all_actions(legal, {ActionType.BET, ActionType.RAISE})

        if observation.to_call > 0:
            if strength >= 0.86 and bets:
                return bets[0]
            if strength >= max(0.68, observation.pot_odds + 0.22) and call:
                return call
            return fold or call or check or legal[0]

        if strength >= 0.78 and bets:
            return bets[0]
        return check or legal[0]


class TightPassiveAgent(Agent):
    """Tight-passive (Rock) opponent: selective preflop, rarely bets/raises, calls with decent hands."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._seed = seed

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            self.rng.seed(seed)

    def act(self, observation: Observation) -> Action:
        legal = observation.legal_actions
        if not legal:
            raise ValueError("No legal actions")
        strength = (
            HeuristicAgent._preflop_strength(observation)
            if not observation.community_cards
            else HeuristicAgent._postflop_strength(observation)[0]
        )
        fold = _find_action(legal, ActionType.FOLD)
        check = _find_action(legal, ActionType.CHECK)
        call = _find_action(legal, ActionType.CALL)
        bets = _find_all_actions(legal, {ActionType.BET, ActionType.RAISE})

        if observation.to_call > 0:
            if strength >= 0.92 and bets:
                return bets[0]
            threshold = (
                0.56
                if not observation.community_cards
                else max(0.50, observation.pot_odds + 0.12)
            )
            if strength >= threshold and call:
                return call
            return fold or call or check or legal[0]

        if strength >= 0.86 and bets:
            return bets[0]
        return check or legal[0]


class CallingStationAgent(Agent):
    """Loose-passive Calling Station: high VPIP, low PFR, calls down with almost anything, rarely folds."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._seed = seed

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            self.rng.seed(seed)

    def act(self, observation: Observation) -> Action:
        legal = observation.legal_actions
        if not legal:
            raise ValueError("No legal actions")
        strength = (
            HeuristicAgent._preflop_strength(observation)
            if not observation.community_cards
            else HeuristicAgent._postflop_strength(observation)[0]
        )
        fold = _find_action(legal, ActionType.FOLD)
        check = _find_action(legal, ActionType.CHECK)
        call = _find_action(legal, ActionType.CALL)
        bets = _find_all_actions(legal, {ActionType.BET, ActionType.RAISE})

        if observation.to_call > 0:
            if strength >= 0.94 and bets:
                return bets[0]
            # Calls with ~85% of hands preflop and any pair / draw / high card postflop
            if strength >= 0.24 and call is not None:
                return call
            if self.rng.random() < 0.70 and call is not None:
                return call
            return fold or call or check or legal[0]

        if strength >= 0.88 and bets:
            return bets[0]
        return check or legal[0]


class LooseAggressiveAgent(Agent):
    """Loose-Aggressive (LAG) opponent: wide VPIP/PFR, frequent c-bets and barrels."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._seed = seed

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            self.rng.seed(seed)

    def act(self, observation: Observation) -> Action:
        legal = observation.legal_actions
        if not legal:
            raise ValueError("No legal actions")
        strength, has_draw = (
            (HeuristicAgent._preflop_strength(observation), False)
            if not observation.community_cards
            else HeuristicAgent._postflop_strength(observation)
        )
        fold = _find_action(legal, ActionType.FOLD)
        check = _find_action(legal, ActionType.CHECK)
        call = _find_action(legal, ActionType.CALL)
        bets = _find_all_actions(legal, {ActionType.BET, ActionType.RAISE})

        if observation.to_call > 0:
            if (
                (strength >= 0.62 or has_draw)
                and bets
                and observation.to_call <= observation.effective_stack * 0.4
            ):
                return bets[min(1, len(bets) - 1)]
            if (
                strength >= max(0.36, observation.pot_odds - 0.04) or has_draw
            ) and call:
                return call
            return fold or call or check or legal[0]

        if (strength >= 0.42 or has_draw or self.rng.random() < 0.55) and bets:
            return bets[min(1, len(bets) - 1)]
        return check or legal[0]


class ManiacAgent(Agent):
    """Hyper-aggressive Maniac: raises and bets relentlessly with a very wide range."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._seed = seed

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            self.rng.seed(seed)

    def act(self, observation: Observation) -> Action:
        legal = observation.legal_actions
        if not legal:
            raise ValueError("No legal actions")
        strength = (
            HeuristicAgent._preflop_strength(observation)
            if not observation.community_cards
            else HeuristicAgent._postflop_strength(observation)[0]
        )
        fold = _find_action(legal, ActionType.FOLD)
        check = _find_action(legal, ActionType.CHECK)
        call = _find_action(legal, ActionType.CALL)
        aggressive = _find_all_actions(
            legal, {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
        )
        non_jam_agg = _find_all_actions(legal, {ActionType.BET, ActionType.RAISE})

        if aggressive and (strength >= 0.26 or self.rng.random() < 0.78):
            if non_jam_agg:
                return non_jam_agg[-1]
            return aggressive[-1]
        if observation.to_call > 0:
            if (strength >= 0.25 or self.rng.random() < 0.65) and call:
                return call
            return fold or call or check or legal[0]
        return check or legal[0]
