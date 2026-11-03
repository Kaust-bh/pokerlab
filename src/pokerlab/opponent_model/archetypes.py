"""Probabilistic Opponent Archetype Classifier for Heads-Up No-Limit Hold'em.

Supported Archetypes:
- `nit`: Ultra-tight preflop (very low VPIP & PFR), high fold-to-bet, low WTSD.
- `tight_passive`: Low-to-moderate VPIP, very low PFR, low aggression frequency, moderate fold-to-bet.
- `calling_station` (Loose-Passive): High VPIP, low PFR, large (VPIP - PFR) gap, low AFq, very low fold-to-bet, high WTSD.
- `tight_aggressive` (TAG): Selective VPIP (~32-44% HU), high PFR/VPIP ratio, solid AFq (~42-55%), balanced fold-to-bet.
- `loose_aggressive` (LAG): Wide VPIP (~54-72% HU), high PFR (~40-58%), high AFq (~52-68%), active 3-betting.
- `maniac`: Extreme VPIP (>75%), extreme PFR (>62%), extreme AFq (>70%), very low fold-to-bet.
- `balanced_unknown`: Equilibrium/prior baseline dominant when observations are sparse or near-GTO balanced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pokerlab.opponent_model.models import (
    ARCHETYPE_LABELS,
    ARCHETYPE_NAMES,
    ArchetypeDistribution,
    OpponentStatsSnapshot,
)


@dataclass(frozen=True, slots=True)
class ArchetypePrototype:
    """Gaussian feature prototype in (vpip, pfr, gap, afq, fold_to_bet, wtsd) space."""

    name: str
    vpip: tuple[float, float]  # (mean, std)
    pfr: tuple[float, float]
    gap: tuple[float, float]  # vpip - pfr
    afq: tuple[float, float]
    fold_to_bet: tuple[float, float]
    wtsd: tuple[float, float]


PROTOTYPES: dict[str, ArchetypePrototype] = {
    "nit": ArchetypePrototype(
        name="nit",
        vpip=(0.16, 0.09),
        pfr=(0.11, 0.08),
        gap=(0.05, 0.07),
        afq=(0.26, 0.12),
        fold_to_bet=(0.68, 0.12),
        wtsd=(0.24, 0.12),
    ),
    "tight_passive": ArchetypePrototype(
        name="tight_passive",
        vpip=(0.28, 0.09),
        pfr=(0.08, 0.07),
        gap=(0.20, 0.09),
        afq=(0.18, 0.10),
        fold_to_bet=(0.52, 0.12),
        wtsd=(0.38, 0.13),
    ),
    "calling_station": ArchetypePrototype(
        name="calling_station",
        vpip=(0.74, 0.13),
        pfr=(0.10, 0.09),
        gap=(0.62, 0.14),
        afq=(0.14, 0.10),
        fold_to_bet=(0.16, 0.11),
        wtsd=(0.72, 0.14),
    ),
    "tight_aggressive": ArchetypePrototype(
        name="tight_aggressive",
        vpip=(0.36, 0.09),
        pfr=(0.29, 0.08),
        gap=(0.07, 0.07),
        afq=(0.48, 0.11),
        fold_to_bet=(0.46, 0.11),
        wtsd=(0.42, 0.12),
    ),
    "loose_aggressive": ArchetypePrototype(
        name="loose_aggressive",
        vpip=(0.62, 0.10),
        pfr=(0.46, 0.10),
        gap=(0.16, 0.09),
        afq=(0.58, 0.11),
        fold_to_bet=(0.32, 0.11),
        wtsd=(0.52, 0.13),
    ),
    "maniac": ArchetypePrototype(
        name="maniac",
        vpip=(0.86, 0.11),
        pfr=(0.74, 0.13),
        gap=(0.12, 0.11),
        afq=(0.78, 0.12),
        fold_to_bet=(0.14, 0.11),
        wtsd=(0.66, 0.15),
    ),
    "balanced_unknown": ArchetypePrototype(
        name="balanced_unknown",
        vpip=(0.46, 0.12),
        pfr=(0.30, 0.10),
        gap=(0.16, 0.09),
        afq=(0.39, 0.11),
        fold_to_bet=(0.43, 0.11),
        wtsd=(0.46, 0.12),
    ),
}


class ArchetypeClassifier:
    """Computes a smooth posterior probability distribution over poker archetypes."""

    def __init__(self, *, cold_start_hands: float = 8.0) -> None:
        self.cold_start_hands = max(2.0, float(cold_start_hands))

    def classify(self, stats: OpponentStatsSnapshot) -> ArchetypeDistribution:
        """Compute archetype probabilities and confidence from `stats`."""
        n_hands = stats.hands_observed
        w_trials = stats.vpip.weighted_trials
        action_trials = stats.aggression_frequency.weighted_trials

        # Effective evidence weight scales in [0, 1) with sample count and action volume
        effective_hands = max(w_trials, min(float(n_hands), w_trials * 1.35))
        effective_n = 0.75 * effective_hands + 0.25 * min(30.0, action_trials / 2.0)
        evidence_weight = (
            effective_n / (effective_n + self.cold_start_hands)
            if effective_n > 0.0
            else 0.0
        )

        # Use recency-weighted empirical rate blended with smoothed estimate so regime changes register cleanly
        vpip_w_rate = (
            stats.vpip.weighted_successes / stats.vpip.weighted_trials
            if stats.vpip.weighted_trials > 0
            else stats.vpip.smoothed_value
        )
        pfr_w_rate = (
            stats.pfr.weighted_successes / stats.pfr.weighted_trials
            if stats.pfr.weighted_trials > 0
            else stats.pfr.smoothed_value
        )
        afq_w_rate = (
            stats.aggression_frequency.weighted_successes
            / stats.aggression_frequency.weighted_trials
            if stats.aggression_frequency.weighted_trials > 0
            else stats.aggression_frequency.smoothed_value
        )
        vpip_val = (
            0.70 * vpip_w_rate + 0.30 * stats.vpip.smoothed_value
            if stats.vpip.weighted_trials >= 4.0
            else stats.vpip.smoothed_value
        )
        pfr_val = (
            0.70 * pfr_w_rate + 0.30 * stats.pfr.smoothed_value
            if stats.pfr.weighted_trials >= 4.0
            else stats.pfr.smoothed_value
        )
        gap_val = max(0.0, vpip_val - pfr_val)
        afq_val = (
            0.70 * afq_w_rate + 0.30 * stats.aggression_frequency.smoothed_value
            if stats.aggression_frequency.weighted_trials >= 4.0
            else stats.aggression_frequency.smoothed_value
        )
        fold_val = stats.fold_to_bet.smoothed_value
        wtsd_val = stats.wtsd_when_saw_flop.smoothed_value

        # Confidence of each individual feature (zero when no postflop trials exist)
        c_vpip = max(0.20, stats.vpip.confidence)
        c_pfr = max(0.20, stats.pfr.confidence)
        c_afq = max(0.15, stats.aggression_frequency.confidence)
        c_fold = stats.fold_to_bet.confidence
        c_wtsd = stats.wtsd_when_saw_flop.confidence

        log_scores: dict[str, float] = {}
        for name in ARCHETYPE_NAMES:
            proto = PROTOTYPES[name]
            d_vpip = ((vpip_val - proto.vpip[0]) / proto.vpip[1]) ** 2
            d_pfr = ((pfr_val - proto.pfr[0]) / proto.pfr[1]) ** 2
            d_gap = ((gap_val - proto.gap[0]) / proto.gap[1]) ** 2
            d_afq = ((afq_val - proto.afq[0]) / proto.afq[1]) ** 2
            d_fold = ((fold_val - proto.fold_to_bet[0]) / proto.fold_to_bet[1]) ** 2
            d_wtsd = ((wtsd_val - proto.wtsd[0]) / proto.wtsd[1]) ** 2

            weighted_sq_dist = (
                1.45 * c_vpip * d_vpip
                + 1.35 * c_pfr * d_pfr
                + 1.25 * min(c_vpip, c_pfr) * d_gap
                + 1.35 * c_afq * d_afq
                + 1.15 * c_fold * d_fold
                + 0.80 * c_wtsd * d_wtsd
            )
            # Temper likelihood by evidence_weight so cold-start stays flat
            temper = 0.35 + 1.85 * evidence_weight
            prior_boost = (
                0.75 * (1.0 - evidence_weight) if name == "balanced_unknown" else 0.0
            )
            log_scores[name] = -0.5 * temper * weighted_sq_dist + prior_boost

        max_log = max(log_scores.values())
        exp_scores = {
            k: math.exp(max(-30.0, v - max_log)) for k, v in log_scores.items()
        }
        total_exp = sum(exp_scores.values())
        likelihood_probs = {k: v / total_exp for k, v in exp_scores.items()}

        # Cold-start prior distribution centered on `balanced_unknown`
        cold_prior: dict[str, float] = {
            "balanced_unknown": 0.56,
            "tight_aggressive": 0.09,
            "loose_aggressive": 0.09,
            "tight_passive": 0.07,
            "calling_station": 0.07,
            "nit": 0.06,
            "maniac": 0.06,
        }

        # Mix cold-start prior and data likelihood posterior by evidence_weight * stability
        effective_mix = min(
            0.95,
            evidence_weight * (0.60 + 0.40 * stats.stability_score),
        )
        final_probs: dict[str, float] = {}
        for name in ARCHETYPE_NAMES:
            final_probs[name] = (1.0 - effective_mix) * cold_prior[
                name
            ] + effective_mix * likelihood_probs[name]

        norm = sum(final_probs.values())
        if norm > 0:
            for name in final_probs:
                final_probs[name] /= norm

        most_likely = max(
            ARCHETYPE_NAMES, key=lambda k: (final_probs[k], k == "balanced_unknown")
        )
        entropy_bits = -sum(p * math.log2(p) for p in final_probs.values() if p > 0.0)
        max_entropy = math.log2(len(ARCHETYPE_NAMES))
        peakedness = (
            max(0.0, 1.0 - entropy_bits / max_entropy) if max_entropy > 0 else 0.0
        )
        confidence = min(
            0.98,
            evidence_weight * (0.45 + 0.55 * peakedness) * stats.stability_score,
        )

        label = ARCHETYPE_LABELS[most_likely]
        if n_hands < 5:
            summary = (
                f"Insufficient sample ({n_hands} hands observed); defaulting toward "
                f"{ARCHETYPE_LABELS['balanced_unknown']} (confidence {confidence:.0%})."
            )
        else:
            runner_up = sorted(
                ARCHETYPE_NAMES, key=lambda k: final_probs[k], reverse=True
            )[1]
            summary = (
                f"Primary profile: {label} ({final_probs[most_likely]:.0%}), "
                f"secondary: {ARCHETYPE_LABELS[runner_up]} ({final_probs[runner_up]:.0%}) "
                f"across {n_hands} hands (confidence {confidence:.0%}, stability {stats.stability_score:.0%})."
            )

        return ArchetypeDistribution(
            probabilities=final_probs,
            most_likely=most_likely,
            most_likely_label=label,
            confidence=confidence,
            evidence_weight=evidence_weight,
            entropy_bits=entropy_bits,
            sample_hands=n_hands,
            stability_score=stats.stability_score,
            summary=summary,
        )
