"""Data models for HarveySpecter opponent intelligence and exploitative strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pokerlab.engine.cards import Card

MODEL_VERSION = "1.0.0"

ARCHETYPE_NAMES: tuple[str, ...] = (
    "nit",
    "tight_passive",
    "calling_station",
    "tight_aggressive",
    "loose_aggressive",
    "maniac",
    "balanced_unknown",
)

ARCHETYPE_LABELS: dict[str, str] = {
    "nit": "Nit (Ultra-Tight)",
    "tight_passive": "Tight-Passive (Rock)",
    "calling_station": "Loose-Passive / Calling Station",
    "tight_aggressive": "Tight-Aggressive (TAG)",
    "loose_aggressive": "Loose-Aggressive (LAG)",
    "maniac": "Maniac (Hyper-Aggressive)",
    "balanced_unknown": "Balanced / Unknown",
}


@dataclass(frozen=True, slots=True)
class ObservedActionEvent:
    """Single publicly observed action within a hand."""

    hand_number: int
    event_index: int
    opponent_id: str
    actor: str
    is_opponent: bool
    street: str
    action_type: str
    amount: int | None
    paid_amount: int
    pot_before: int
    to_call_before: int
    stack_before: int
    effective_stack_before: int
    position: str  # "button" or "big_blind"
    in_position: bool
    pot_ratio: float
    facing_bet: bool
    facing_raise_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_number": self.hand_number,
            "event_index": self.event_index,
            "opponent_id": self.opponent_id,
            "actor": self.actor,
            "is_opponent": self.is_opponent,
            "street": self.street,
            "action_type": self.action_type,
            "amount": self.amount,
            "paid_amount": self.paid_amount,
            "pot_before": self.pot_before,
            "to_call_before": self.to_call_before,
            "stack_before": self.stack_before,
            "effective_stack_before": self.effective_stack_before,
            "position": self.position,
            "in_position": self.in_position,
            "pot_ratio": round(self.pot_ratio, 6),
            "facing_bet": self.facing_bet,
            "facing_raise_count": self.facing_raise_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservedActionEvent:
        return cls(
            hand_number=int(data["hand_number"]),
            event_index=int(data["event_index"]),
            opponent_id=str(data["opponent_id"]),
            actor=str(data["actor"]),
            is_opponent=bool(data["is_opponent"]),
            street=str(data["street"]),
            action_type=str(data["action_type"]),
            amount=int(data["amount"]) if data.get("amount") is not None else None,
            paid_amount=int(data.get("paid_amount", 0)),
            pot_before=int(data.get("pot_before", 0)),
            to_call_before=int(data.get("to_call_before", 0)),
            stack_before=int(data.get("stack_before", 0)),
            effective_stack_before=int(data.get("effective_stack_before", 0)),
            position=str(data.get("position", "big_blind")),
            in_position=bool(data.get("in_position", False)),
            pot_ratio=float(data.get("pot_ratio", 0.0)),
            facing_bet=bool(data.get("facing_bet", False)),
            facing_raise_count=int(data.get("facing_raise_count", 0)),
        )


@dataclass(slots=True)
class HandObservationRecord:
    """Complete or partial public record of a single hand against an opponent."""

    opponent_id: str
    hand_number: int
    opponent_position: str  # "button" or "big_blind"
    opponent_is_button: bool
    starting_stack: int
    effective_stack: int
    big_blind: int
    actions: list[ObservedActionEvent] = field(default_factory=list)
    community_cards: tuple[Card, ...] = ()
    saw_flop: bool = False
    reached_showdown: bool = False
    opponent_folded: bool = False
    hero_folded: bool = False
    revealed_hole_cards: tuple[Card, Card] | None = None
    final_pot: int = 0
    opponent_net_chips: int = 0
    opponent_won: bool = False
    completed: bool = False
    processed_history_len: int = 0

    @property
    def preflop_actions(self) -> tuple[ObservedActionEvent, ...]:
        return tuple(a for a in self.actions if a.street == "preflop")

    @property
    def flop_actions(self) -> tuple[ObservedActionEvent, ...]:
        return tuple(a for a in self.actions if a.street == "flop")

    @property
    def turn_actions(self) -> tuple[ObservedActionEvent, ...]:
        return tuple(a for a in self.actions if a.street == "turn")

    @property
    def river_actions(self) -> tuple[ObservedActionEvent, ...]:
        return tuple(a for a in self.actions if a.street == "river")

    @property
    def opponent_actions(self) -> tuple[ObservedActionEvent, ...]:
        return tuple(a for a in self.actions if a.is_opponent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "hand_number": self.hand_number,
            "opponent_position": self.opponent_position,
            "opponent_is_button": self.opponent_is_button,
            "starting_stack": self.starting_stack,
            "effective_stack": self.effective_stack,
            "big_blind": self.big_blind,
            "actions": [a.to_dict() for a in self.actions],
            "community_cards": [[c.rank, c.suit] for c in self.community_cards],
            "saw_flop": self.saw_flop,
            "reached_showdown": self.reached_showdown,
            "opponent_folded": self.opponent_folded,
            "hero_folded": self.hero_folded,
            "revealed_hole_cards": (
                [[c.rank, c.suit] for c in self.revealed_hole_cards]
                if self.revealed_hole_cards is not None and self.reached_showdown
                else None
            ),
            "final_pot": self.final_pot,
            "opponent_net_chips": self.opponent_net_chips,
            "opponent_won": self.opponent_won,
            "completed": self.completed,
            "processed_history_len": self.processed_history_len,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandObservationRecord:
        reached_sd = bool(data.get("reached_showdown", False))
        raw_revealed = data.get("revealed_hole_cards")
        revealed: tuple[Card, Card] | None = None
        if reached_sd and raw_revealed and len(raw_revealed) == 2:
            revealed = (
                Card(int(raw_revealed[0][0]), str(raw_revealed[0][1])),
                Card(int(raw_revealed[1][0]), str(raw_revealed[1][1])),
            )
        comm = tuple(
            Card(int(c[0]), str(c[1])) for c in data.get("community_cards", [])
        )
        return cls(
            opponent_id=str(data["opponent_id"]),
            hand_number=int(data["hand_number"]),
            opponent_position=str(data.get("opponent_position", "big_blind")),
            opponent_is_button=bool(data.get("opponent_is_button", False)),
            starting_stack=int(data.get("starting_stack", 200)),
            effective_stack=int(data.get("effective_stack", 200)),
            big_blind=int(data.get("big_blind", 2)),
            actions=[ObservedActionEvent.from_dict(a) for a in data.get("actions", [])],
            community_cards=comm,
            saw_flop=bool(data.get("saw_flop", False)),
            reached_showdown=reached_sd,
            opponent_folded=bool(data.get("opponent_folded", False)),
            hero_folded=bool(data.get("hero_folded", False)),
            revealed_hole_cards=revealed,
            final_pot=int(data.get("final_pot", 0)),
            opponent_net_chips=int(data.get("opponent_net_chips", 0)),
            opponent_won=bool(data.get("opponent_won", False)),
            completed=bool(data.get("completed", False)),
            processed_history_len=int(data.get("processed_history_len", 0)),
        )


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """Statistically smoothed behavioral metric with sample count and uncertainty."""

    name: str
    successes: int
    trials: int
    weighted_successes: float
    weighted_trials: float
    raw_value: float | None
    smoothed_value: float
    prior_mean: float
    prior_strength: float
    ci95_low: float
    ci95_high: float
    uncertainty: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "successes": self.successes,
            "trials": self.trials,
            "weighted_successes": round(self.weighted_successes, 4),
            "weighted_trials": round(self.weighted_trials, 4),
            "raw_value": round(self.raw_value, 6)
            if self.raw_value is not None
            else None,
            "smoothed_value": round(self.smoothed_value, 6),
            "prior_mean": round(self.prior_mean, 6),
            "prior_strength": round(self.prior_strength, 4),
            "ci95_low": round(self.ci95_low, 6),
            "ci95_high": round(self.ci95_high, 6),
            "uncertainty": round(self.uncertainty, 6),
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True, slots=True)
class OpponentStatsSnapshot:
    """Complete snapshot of opponent behavioral metrics and sample counts."""

    opponent_id: str
    hands_observed: int
    completed_hands: int
    showdowns_observed: int
    recency_decay: float
    stability_score: float
    vpip: MetricEstimate
    pfr: MetricEstimate
    vpip_minus_pfr: float
    three_bet_freq: MetricEstimate
    fold_to_three_bet: MetricEstimate
    cbet_freq: MetricEstimate
    fold_to_cbet: MetricEstimate
    aggression_frequency: MetricEstimate
    aggression_factor: float
    raw_aggression_factor: float | None
    fold_to_bet: MetricEstimate
    fold_to_river_bet: MetricEstimate
    showdown_frequency: MetricEstimate
    wtsd_when_saw_flop: MetricEstimate
    won_at_showdown: MetricEstimate
    avg_bet_size_pot_ratio: MetricEstimate
    avg_raise_size_pot_ratio: MetricEstimate
    vpip_button: MetricEstimate
    vpip_big_blind: MetricEstimate
    pfr_button: MetricEstimate
    pfr_big_blind: MetricEstimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "hands_observed": self.hands_observed,
            "completed_hands": self.completed_hands,
            "showdowns_observed": self.showdowns_observed,
            "recency_decay": round(self.recency_decay, 6),
            "stability_score": round(self.stability_score, 6),
            "vpip": self.vpip.to_dict(),
            "pfr": self.pfr.to_dict(),
            "vpip_minus_pfr": round(self.vpip_minus_pfr, 6),
            "three_bet_freq": self.three_bet_freq.to_dict(),
            "fold_to_three_bet": self.fold_to_three_bet.to_dict(),
            "cbet_freq": self.cbet_freq.to_dict(),
            "fold_to_cbet": self.fold_to_cbet.to_dict(),
            "aggression_frequency": self.aggression_frequency.to_dict(),
            "aggression_factor": round(self.aggression_factor, 6),
            "raw_aggression_factor": (
                round(self.raw_aggression_factor, 6)
                if self.raw_aggression_factor is not None
                else None
            ),
            "fold_to_bet": self.fold_to_bet.to_dict(),
            "fold_to_river_bet": self.fold_to_river_bet.to_dict(),
            "showdown_frequency": self.showdown_frequency.to_dict(),
            "wtsd_when_saw_flop": self.wtsd_when_saw_flop.to_dict(),
            "won_at_showdown": self.won_at_showdown.to_dict(),
            "avg_bet_size_pot_ratio": self.avg_bet_size_pot_ratio.to_dict(),
            "avg_raise_size_pot_ratio": self.avg_raise_size_pot_ratio.to_dict(),
            "vpip_button": self.vpip_button.to_dict(),
            "vpip_big_blind": self.vpip_big_blind.to_dict(),
            "pfr_button": self.pfr_button.to_dict(),
            "pfr_big_blind": self.pfr_big_blind.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ArchetypeDistribution:
    """Probabilistic classification across opponent archetypes."""

    probabilities: dict[str, float]
    most_likely: str
    most_likely_label: str
    confidence: float
    evidence_weight: float
    entropy_bits: float
    sample_hands: int
    stability_score: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
            "labels": dict(ARCHETYPE_LABELS),
            "most_likely": self.most_likely,
            "most_likely_label": self.most_likely_label,
            "confidence": round(self.confidence, 6),
            "evidence_weight": round(self.evidence_weight, 6),
            "entropy_bits": round(self.entropy_bits, 6),
            "sample_hands": self.sample_hands,
            "stability_score": round(self.stability_score, 6),
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class RangeClassEstimate:
    """Prior and posterior probability mass for a canonical 169 starting hand class."""

    hand_class: str
    combo_count: int
    prior_probability: float
    posterior_probability: float
    probability_delta: float
    mean_strength: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_class": self.hand_class,
            "combo_count": self.combo_count,
            "prior_probability": round(self.prior_probability, 6),
            "posterior_probability": round(self.posterior_probability, 6),
            "probability_delta": round(self.probability_delta, 6),
            "mean_strength": round(self.mean_strength, 6),
        }


@dataclass(frozen=True, slots=True)
class RangeComboEstimate:
    """Prior and posterior probability for a specific 2-card holding."""

    cards: tuple[str, str]
    hand_class: str
    prior_probability: float
    posterior_probability: float
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cards": list(self.cards),
            "combo_label": "".join(self.cards),
            "hand_class": self.hand_class,
            "prior_probability": round(self.prior_probability, 6),
            "posterior_probability": round(self.posterior_probability, 6),
            "strength": round(self.strength, 6),
        }


@dataclass(frozen=True, slots=True)
class RangeUpdateStep:
    """Summary of how a single observed opponent action shifted the Bayesian range."""

    step_index: int
    street: str
    action_type: str
    amount: int | None
    pot_ratio: float
    prior_mean_strength: float
    posterior_mean_strength: float
    l1_shift: float
    top_classes_after: tuple[tuple[str, float], ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "street": self.street,
            "action_type": self.action_type,
            "amount": self.amount,
            "pot_ratio": round(self.pot_ratio, 6),
            "prior_mean_strength": round(self.prior_mean_strength, 6),
            "posterior_mean_strength": round(self.posterior_mean_strength, 6),
            "l1_shift": round(self.l1_shift, 6),
            "top_classes_after": [
                {"hand_class": c, "probability": round(p, 6)}
                for c, p in self.top_classes_after
            ],
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class RangeSnapshot:
    """Serializable summary of the Bayesian opponent hand range."""

    street: str
    active_combo_count: int
    eliminated_combo_count: int
    is_folded: bool
    prior_mean_strength: float
    posterior_mean_strength: float
    l1_shift_from_prior: float
    entropy_bits: float
    max_entropy_bits: float
    concentration: float
    top_classes: tuple[RangeClassEstimate, ...]
    top_combos: tuple[RangeComboEstimate, ...]
    tier_breakdown: dict[str, dict[str, float]]
    action_updates: tuple[RangeUpdateStep, ...]
    showdown_calibration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "street": self.street,
            "active_combo_count": self.active_combo_count,
            "eliminated_combo_count": self.eliminated_combo_count,
            "is_folded": self.is_folded,
            "prior_mean_strength": round(self.prior_mean_strength, 6),
            "posterior_mean_strength": round(self.posterior_mean_strength, 6),
            "l1_shift_from_prior": round(self.l1_shift_from_prior, 6),
            "entropy_bits": round(self.entropy_bits, 6),
            "max_entropy_bits": round(self.max_entropy_bits, 6),
            "concentration": round(self.concentration, 6),
            "top_classes": [c.to_dict() for c in self.top_classes],
            "top_combos": [c.to_dict() for c in self.top_combos],
            "tier_breakdown": {
                k: {sub_k: round(sub_v, 6) for sub_k, sub_v in v.items()}
                for k, v in self.tier_breakdown.items()
            },
            "action_updates": [u.to_dict() for u in self.action_updates],
            "showdown_calibration": dict(self.showdown_calibration),
        }


@dataclass(frozen=True, slots=True)
class ExploitDecisionReport:
    """Complete decision telemetry for HarveySpecter Policy Inspector integration."""

    opponent_id: str
    hand_number: int
    street: str
    lambda_value: float
    lambda_mode: str
    max_lambda: float
    evidence_confidence: float
    uncertainty: float
    stability_score: float
    lookup_tier: str
    lookup_subtier: str
    hero_range_equity: float
    hero_uniform_equity: float
    estimated_fold_prob: float
    pot_odds: float
    mccfr_probabilities: list[dict[str, Any]]
    exploit_probabilities: list[dict[str, Any]]
    final_probabilities: list[dict[str, Any]]
    action_expected_values: dict[str, float]
    adjustment_reasons: tuple[str, ...]
    explanation: str
    opponent_stats: OpponentStatsSnapshot
    archetype: ArchetypeDistribution
    range_snapshot: RangeSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent_id": self.opponent_id,
            "hand_number": self.hand_number,
            "street": self.street,
            "lambda_value": round(self.lambda_value, 6),
            "lambda_mode": self.lambda_mode,
            "max_lambda": round(self.max_lambda, 6),
            "evidence_confidence": round(self.evidence_confidence, 6),
            "uncertainty": round(self.uncertainty, 6),
            "stability_score": round(self.stability_score, 6),
            "lookup_tier": self.lookup_tier,
            "lookup_subtier": self.lookup_subtier,
            "hero_range_equity": round(self.hero_range_equity, 6),
            "hero_uniform_equity": round(self.hero_uniform_equity, 6),
            "estimated_fold_prob": round(self.estimated_fold_prob, 6),
            "pot_odds": round(self.pot_odds, 6),
            "mccfr_probabilities": self.mccfr_probabilities,
            "exploit_probabilities": self.exploit_probabilities,
            "final_probabilities": self.final_probabilities,
            "action_expected_values": {
                k: round(v, 4) for k, v in self.action_expected_values.items()
            },
            "adjustment_reasons": list(self.adjustment_reasons),
            "explanation": self.explanation,
            "opponent_profile": self.opponent_stats.to_dict(),
            "archetype_classification": self.archetype.to_dict(),
            "bayesian_range": self.range_snapshot.to_dict(),
        }
