"""Non-mutating Policy Inspector telemetry and public equity helpers for PokerLab Arena."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pokerlab.agents.base import Agent, Observation
from pokerlab.agents.equity_agent import EquityAgent
from pokerlab.agents.heuristic_agent import HeuristicAgent
from pokerlab.agents.random_agent import RandomAgent
from pokerlab.algorithms.monte_carlo import estimate_equity
from pokerlab.engine import Action, ActionType, Card
from pokerlab.opponent_model.adaptive_agent import HarveySpecterAgent
from pokerlab.research.nlhe.agent import AbstractHUNLAgent
from pokerlab.research.nlhe.information_set import (
    HUNLInformationSet,
    information_set_from_observation,
)


def compute_shannon_entropy(probabilities: list[float]) -> dict[str, float]:
    """Compute exact Shannon entropy (bits and nats) from a probability distribution."""
    positive = [p for p in probabilities if p > 0.0]
    total = sum(positive)
    if total <= 0.0 or not positive:
        return {
            "entropy_bits": 0.0,
            "entropy_nats": 0.0,
            "max_entropy_bits": 0.0,
            "normalized_entropy": 0.0,
        }
    normed = [p / total for p in positive]
    bits = -sum(p * math.log2(p) for p in normed if p > 0.0)
    nats = -sum(p * math.log(p) for p in normed if p > 0.0)
    n_actions = len(probabilities)
    max_bits = math.log2(n_actions) if n_actions > 1 else 0.0
    normalized = (bits / max_bits) if max_bits > 0.0 else 0.0
    return {
        "entropy_bits": round(bits, 6),
        "entropy_nats": round(nats, 6),
        "max_entropy_bits": round(max_bits, 6),
        "normalized_entropy": round(min(1.0, max(0.0, normalized)), 6),
    }


def format_action_label(action: Action) -> str:
    """Format an engine Action into a concise human-readable label."""
    kind = ActionType(action.type).value
    if kind in {ActionType.BET.value, ActionType.RAISE.value} and action.amount:
        return f"{kind} {action.amount}"
    return kind.replace("_", "-")


def serialize_engine_action_choice(action: Action) -> dict[str, Any]:
    """Serialize an engine Action candidate for telemetry display."""
    kind = ActionType(action.type).value
    return {
        "action_type": kind,
        "amount": action.amount,
        "label": format_action_label(action),
    }


def _observation_history_tuples(
    observation: Observation,
) -> tuple[tuple[str, bool, str, int | None], ...]:
    return tuple(
        (
            record.street,
            record.player == observation.player
            if observation.is_button
            else record.player != observation.player,
            ActionType(record.action.type).value,
            record.action.amount,
        )
        for record in observation.history
    )


def _resolve_mccfr_subtier(
    agent: AbstractHUNLAgent,
    info: HUNLInformationSet,
) -> tuple[str, str]:
    """Identify the exact lookup tier, subtier, and matched key string for MCCFR."""
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


def inspect_and_execute_agent(
    agent: Agent,
    agent_id: str,
    observation: Observation,
) -> tuple[Action, dict[str, Any]]:
    """Capture genuine pre-decision telemetry without altering agent RNG, then call `act`."""
    if isinstance(agent, HarveySpecterAgent):
        chosen_action = agent.act(observation)
        report = agent.last_decision_report
        assert report is not None
        report_dict = report.to_dict()

        history = _observation_history_tuples(observation)
        info = information_set_from_observation(
            observation, history, agent.baseline_agent.config
        )
        _, matched_key_repr = _resolve_mccfr_subtier(agent.baseline_agent, info)
        board_mode = getattr(agent.baseline_agent.config, "board_abstraction", "exact")
        public_info_summary = (
            f"role={'BTN' if info.player == 0 else 'BB'} | street={info.street} | "
            f"board={list(info.board)} | pot={info.pot} | eff_stack={info.effective_stack}"
        )

        mccfr_rows = [
            {
                **row,
                "is_chosen": (
                    row["action_type"] == ActionType(chosen_action.type).value
                    and row["amount"] == chosen_action.amount
                ),
            }
            for row in report_dict["mccfr_probabilities"]
        ]
        exploit_rows = [
            {
                **row,
                "is_chosen": (
                    row["action_type"] == ActionType(chosen_action.type).value
                    and row["amount"] == chosen_action.amount
                ),
            }
            for row in report_dict["exploit_probabilities"]
        ]
        final_rows = [
            {
                **row,
                "is_chosen": (
                    row["action_type"] == ActionType(chosen_action.type).value
                    and row["amount"] == chosen_action.amount
                ),
            }
            for row in report_dict["final_probabilities"]
        ]

        entropy_stats = compute_shannon_entropy(
            [float(r["probability"]) for r in final_rows]
        )

        telemetry: dict[str, Any] = {
            "agent_type": "HarveySpecterAgent",
            "agent_id": agent_id,
            "lookup_tier": report.lookup_tier,
            "lookup_subtier": report.lookup_subtier,
            "action_probabilities": final_rows,
            "mccfr_probabilities": mccfr_rows,
            "exploit_probabilities": exploit_rows,
            "final_probabilities": final_rows,
            "lambda_value": round(report.lambda_value, 6),
            "lambda_mode": report.lambda_mode,
            "max_lambda": round(report.max_lambda, 6),
            "evidence_confidence": round(report.evidence_confidence, 6),
            "uncertainty": round(report.uncertainty, 6),
            "stability_score": round(report.stability_score, 6),
            "hero_range_equity": round(report.hero_range_equity, 6),
            "hero_uniform_equity": round(report.hero_uniform_equity, 6),
            "estimated_fold_prob": round(report.estimated_fold_prob, 6),
            "pot_odds": round(report.pot_odds, 6),
            "action_expected_values": report_dict["action_expected_values"],
            "adjustment_reasons": report_dict["adjustment_reasons"],
            "explanation": report.explanation,
            "opponent_profile": report_dict["opponent_profile"],
            "archetype_classification": report_dict["archetype_classification"],
            "bayesian_range": report_dict["bayesian_range"],
            "strategy_entropy": entropy_stats,
            "public_information_set_summary": public_info_summary,
            "private_bucket": info.private_bucket,
            "information_set_key": {
                "player_role": info.player,
                "street": info.street,
                "private_bucket": info.private_bucket,
                "board": [list(c) for c in info.board],
                "pot": info.pot,
                "effective_stack": info.effective_stack,
                "street_contributions": list(info.street_contributions),
                "history": [list(h) for h in info.history],
                "legal_actions": [a.value for a in info.legal_actions],
                "board_abstraction": board_mode,
            },
            "matched_abstraction_key": matched_key_repr,
            "direct_telemetry": [
                "mccfr_probabilities",
                "exploit_probabilities",
                "final_probabilities",
                "lambda_value",
                "evidence_confidence",
                "uncertainty",
                "opponent_profile",
                "archetype_classification",
                "bayesian_range",
                "explanation",
            ],
            "derived_analysis": [
                "strategy_entropy",
                "action_expected_values",
            ],
            "unavailable_fields": [],
        }
        return chosen_action, telemetry

    if isinstance(agent, AbstractHUNLAgent):
        history = _observation_history_tuples(observation)
        info = information_set_from_observation(observation, history, agent.config)
        specs, weights, lookup_tier = agent._resolve_policy(observation)
        subtier, matched_key_repr = _resolve_mccfr_subtier(agent, info)
        total_w = sum(weights)
        norm_weights = (
            [w / total_w for w in weights]
            if total_w > 0
            else [1.0 / max(len(specs), 1) for _ in specs]
        )

        # Execute the real agent decision once (advances agent.rng by 1 draw)
        chosen_action = agent.act(observation)

        prob_rows: list[dict[str, Any]] = []
        for spec, prob in zip(specs, norm_weights, strict=True):
            prob_rows.append(
                {
                    "abstract_action": spec.abstract.value,
                    "action_type": ActionType(spec.engine.type).value,
                    "amount": spec.engine.amount,
                    "label": f"{spec.abstract.value} ({format_action_label(spec.engine)})",
                    "probability": round(prob, 6),
                    "is_chosen": spec.engine == chosen_action,
                }
            )

        engine_dist = agent.action_probabilities(observation)
        engine_prob_rows = [
            {
                "action_type": ActionType(act.type).value,
                "amount": act.amount,
                "label": format_action_label(act),
                "probability": round(float(prob), 6),
                "is_chosen": act == chosen_action,
            }
            for act, prob in engine_dist.items()
        ]
        entropy_stats = compute_shannon_entropy(
            [float(p) for p in engine_dist.values()]
        )
        board_mode = getattr(agent.config, "board_abstraction", "exact")
        public_info_summary = (
            f"role={'BTN' if info.player == 0 else 'BB'} | street={info.street} | "
            f"board={list(info.board)} | pot={info.pot} | eff_stack={info.effective_stack}"
        )

        telemetry: dict[str, Any] = {
            "agent_type": "AbstractHUNLAgent",
            "agent_id": agent_id,
            "lookup_tier": lookup_tier,
            "lookup_subtier": subtier,
            "action_probabilities": engine_prob_rows,
            "abstract_action_probabilities": prob_rows,
            "strategy_entropy": entropy_stats,
            "public_information_set_summary": public_info_summary,
            "private_bucket": info.private_bucket,
            "information_set_key": {
                "player_role": info.player,
                "street": info.street,
                "private_bucket": info.private_bucket,
                "board": [list(c) for c in info.board],
                "pot": info.pot,
                "effective_stack": info.effective_stack,
                "street_contributions": list(info.street_contributions),
                "history": [list(h) for h in info.history],
                "legal_actions": [a.value for a in info.legal_actions],
                "board_abstraction": board_mode,
            },
            "matched_abstraction_key": matched_key_repr,
            "cumulative_lookup_rates": agent.lookup_rates(),
            "direct_telemetry": [
                "lookup_tier",
                "lookup_subtier",
                "action_probabilities",
                "abstract_action_probabilities",
                "information_set_key",
                "matched_abstraction_key",
                "cumulative_lookup_rates",
            ],
            "derived_analysis": [
                "strategy_entropy",
            ],
            "unavailable_fields": [
                "hand_equity_estimate (AbstractHUNLAgent uses MCCFR strategy tables over abstract buckets, not runtime Monte Carlo equity rollouts)",
                "action_expected_values (MCCFR average strategy stores normalized action probabilities, not chip EVs)",
            ],
        }
        return chosen_action, telemetry

    if isinstance(agent, EquityAgent):
        # Peek the exact seed that EquityAgent.estimate() will draw from agent._rng
        rng_state = agent._rng.getstate()
        shadow_rng = random.Random()
        shadow_rng.setstate(rng_state)
        rollout_seed = shadow_rng.randrange(2**32)
        est = estimate_equity(
            observation.hole_cards,
            observation.community_cards,
            simulations=agent.config.simulations,
            seed=rollout_seed,
            method="monte_carlo",
        )

        # Execute the real agent decision (consumes the same rollout_seed from agent._rng)
        chosen_action = agent.act(observation)

        margin = agent.config.call_margin
        if observation.in_position:
            margin = max(0.0, margin - agent.config.position_discount)

        call_amount = min(observation.to_call, observation.stack)
        static_evs: dict[str, float | None] = {}
        for legal in observation.legal_actions:
            kind = ActionType(legal.type).value
            if kind == ActionType.FOLD.value:
                static_evs["fold"] = 0.0
            elif kind == ActionType.CHECK.value and observation.to_call == 0:
                static_evs["check"] = round(est.equity * observation.pot, 4)
            elif kind == ActionType.CALL.value and observation.to_call > 0:
                static_evs["call"] = round(
                    est.equity * (observation.pot + call_amount) - call_amount,
                    4,
                )

        action_probs = [
            {
                "action_type": ActionType(act.type).value,
                "amount": act.amount,
                "label": format_action_label(act),
                "probability": 1.0 if act == chosen_action else 0.0,
                "is_chosen": act == chosen_action,
            }
            for act in observation.legal_actions
        ]

        telemetry = {
            "agent_type": "EquityAgent",
            "agent_id": agent_id,
            "equity_estimate": {
                "equity": round(est.equity, 6),
                "win_probability": round(est.win_probability, 6),
                "tie_probability": round(est.tie_probability, 6),
                "loss_probability": round(est.loss_probability, 6),
                "simulations": est.simulations,
                "confidence_interval": [
                    round(est.confidence_interval[0], 6),
                    round(est.confidence_interval[1], 6),
                ],
                "method": est.method,
                "is_exact": False,
                "rollout_seed": rollout_seed,
                "agent_seed": agent._seed,
            },
            "thresholds": {
                "pot_odds": round(observation.pot_odds, 6),
                "effective_call_margin": round(margin, 6),
                "required_call_equity": round(observation.pot_odds + margin, 6)
                if observation.to_call > 0
                else 0.0,
                "value_bet_equity": agent.config.value_bet_equity,
                "raise_equity": agent.config.raise_equity,
            },
            "static_pot_odds_ev_chips": static_evs,
            "action_probabilities": action_probs,
            "direct_telemetry": [
                "equity_estimate",
                "thresholds",
            ],
            "derived_analysis": [
                "static_pot_odds_ev_chips (1-step showdown EV for fold/check/call assuming no further betting)",
            ],
            "unavailable_fields": [
                "bet_and_raise_ev_estimates (EquityAgent uses fixed equity thresholds rather than modeling opponent fold/call frequencies)",
                "opponent_hand_range_posterior (EquityAgent samples uniform random opponent hole cards)",
            ],
        }
        return chosen_action, telemetry

    if isinstance(agent, HeuristicAgent):
        raw_strength, has_draw = agent.strength(observation)
        pos_adj = agent.config.position_bonus if observation.in_position else 0.0
        base_decision_strength = min(1.0, raw_strength + pos_adj)
        recent_pressure = 0
        if observation.to_call:
            recent_pressure = sum(
                record.player != observation.player
                and record.street == observation.street
                and record.action.type
                in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
                for record in observation.history
            )
            decision_strength = max(
                0.0, base_decision_strength - 0.04 * recent_pressure
            )
        else:
            decision_strength = base_decision_strength

        chosen_action = agent.act(observation)
        action_probs = [
            {
                "action_type": ActionType(act.type).value,
                "amount": act.amount,
                "label": format_action_label(act),
                "probability": 1.0 if act == chosen_action else 0.0,
                "is_chosen": act == chosen_action,
            }
            for act in observation.legal_actions
        ]

        telemetry = {
            "agent_type": "HeuristicAgent",
            "agent_id": agent_id,
            "heuristic_signals": {
                "raw_strength": round(raw_strength, 6),
                "has_draw": bool(has_draw),
                "in_position": bool(observation.in_position),
                "position_adjustment": round(pos_adj, 6),
                "recent_pressure_actions": int(recent_pressure),
                "decision_strength": round(decision_strength, 6),
                "pot_odds": round(observation.pot_odds, 6),
                "required_call_strength": round(
                    observation.pot_odds + agent.config.call_margin, 6
                )
                if observation.to_call > 0
                else 0.0,
                "value_bet_strength": agent.config.value_bet_strength,
                "strong_raise_strength": agent.config.strong_raise_strength,
            },
            "action_probabilities": action_probs,
            "direct_telemetry": [
                "heuristic_signals",
            ],
            "derived_analysis": [],
            "unavailable_fields": [
                "equity_estimate (HeuristicAgent uses static hand-category/preflop strength bands, not equity simulations)",
                "action_expected_values (rule-based threshold policy without EV computation)",
            ],
        }
        return chosen_action, telemetry

    if isinstance(agent, RandomAgent):
        dist = agent.action_probabilities(observation)
        chosen_action = agent.act(observation)
        action_probs = [
            {
                "action_type": ActionType(act.type).value,
                "amount": act.amount,
                "label": format_action_label(act),
                "probability": round(float(prob), 6),
                "is_chosen": act == chosen_action,
            }
            for act, prob in dist.items()
        ]
        entropy_stats = compute_shannon_entropy([float(p) for p in dist.values()])
        telemetry = {
            "agent_type": "RandomAgent",
            "agent_id": agent_id,
            "sampling_info": {
                "distribution": "uniform_over_legal_action_abstraction",
                "legal_action_count": len(observation.legal_actions),
                "uniform_probability": round(
                    1.0 / max(len(observation.legal_actions), 1), 6
                ),
                "agent_seed": agent._seed,
            },
            "action_probabilities": action_probs,
            "strategy_entropy": entropy_stats,
            "direct_telemetry": [
                "sampling_info",
                "action_probabilities",
            ],
            "derived_analysis": [
                "strategy_entropy",
            ],
            "unavailable_fields": [
                "hand_strength_or_equity (RandomAgent ignores hole cards, community cards, and pot odds)",
                "action_expected_values (uniform random baseline)",
            ],
        }
        return chosen_action, telemetry

    # Generic fallback for any custom Agent subclass used in tests
    dist = agent.action_probabilities(observation)
    chosen_action = agent.act(observation)
    action_probs = [
        {
            "action_type": ActionType(act.type).value,
            "amount": act.amount,
            "label": format_action_label(act),
            "probability": round(float(dist.get(act, 0.0)), 6),
            "is_chosen": act == chosen_action,
        }
        for act in observation.legal_actions
    ]
    telemetry = {
        "agent_type": type(agent).__name__,
        "agent_id": agent_id,
        "action_probabilities": action_probs,
        "direct_telemetry": ["action_probabilities"],
        "derived_analysis": [],
        "unavailable_fields": [
            "internal_agent_telemetry (custom agent class does not expose additional telemetry)"
        ],
    }
    return chosen_action, telemetry


def estimate_human_public_equity(
    hole_cards: tuple[Card, Card],
    community_cards: tuple[Card, ...],
    *,
    simulations: int = 250,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute an approximate public Monte Carlo equity estimate for the human seat.

    Strictly uses only the human's own `hole_cards` and public `community_cards`,
    sampling unknown opponent cards uniformly without replacement.
    """
    est = estimate_equity(
        hole_cards,
        community_cards,
        simulations=max(25, min(int(simulations), 2000)),
        seed=seed,
        method="monte_carlo",
    )
    return {
        "enabled": True,
        "is_exact": False,
        "label": "Approximate Monte Carlo Equity vs. Uniform Random Opponent Range",
        "equity": round(est.equity, 6),
        "win_probability": round(est.win_probability, 6),
        "tie_probability": round(est.tie_probability, 6),
        "loss_probability": round(est.loss_probability, 6),
        "simulations": est.simulations,
        "confidence_interval": [
            round(est.confidence_interval[0], 6),
            round(est.confidence_interval[1], 6),
        ],
        "method": est.method,
        "seed": seed,
    }


@dataclass(slots=True)
class DecisionRecord:
    """Immutable snapshot of a single human or AI decision in a hand."""

    decision_id: int
    hand_number: int
    step_index: int
    timestamp_iso: str
    street: str
    player: str
    agent_id: str
    agent_name: str
    is_human: bool
    chosen_action: dict[str, Any]
    public_context: dict[str, Any]
    telemetry: dict[str, Any]

    def serialize(self) -> dict[str, Any]:
        """Return an immutable JSON-safe copy of this decision snapshot."""
        import copy

        return {
            "decision_id": self.decision_id,
            "hand_number": self.hand_number,
            "step_index": self.step_index,
            "timestamp_iso": self.timestamp_iso,
            "street": self.street,
            "player": self.player,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "is_human": self.is_human,
            "chosen_action": copy.deepcopy(self.chosen_action),
            "public_context": copy.deepcopy(self.public_context),
            "telemetry": copy.deepcopy(self.telemetry),
        }


def current_utc_iso() -> str:
    return datetime.now(UTC).isoformat()
