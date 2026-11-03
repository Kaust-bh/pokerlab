"""Behavioural Profiler for Heads-Up No-Limit Hold'em opponents.

Mathematical definitions:
- Rate metrics use a Beta-Binomial conjugate model:
    Prior: p ~ Beta(alpha_0, beta_0), with prior mean mu_0 = alpha_0 / kappa_0
    and pseudocount strength kappa_0 = alpha_0 + beta_0.
    Given recency-decayed successes S_w and trials N_w:
        p_hat = (alpha_0 + S_w) / (kappa_0 + N_w)
    Posterior variance:
        Var(p | D) = (p_hat * (1 - p_hat)) / (kappa_0 + N_w + 1)
    95% Credible Interval (clamped to [0, 1]):
        p_hat +/- 1.96 * sqrt(Var(p | D))
    Sample confidence:
        c(N) = N_w / (N_w + kappa_0) in [0, 1)
- Continuous sizing ratios (bet/raise size relative to pot) use Normal-Normal
  shrinkage toward prior mean mu_0 with pseudocount kappa_0:
        r_hat = (kappa_0 * mu_0 + sum_i w_i * r_i) / (kappa_0 + sum_i w_i)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pokerlab.engine import ActionType
from pokerlab.opponent_model.models import (
    HandObservationRecord,
    MetricEstimate,
    OpponentStatsSnapshot,
)

PRIORS: dict[str, tuple[float, float]] = {
    # metric_name: (prior_mean, prior_strength_kappa)
    "vpip": (0.45, 6.0),
    "pfr": (0.28, 6.0),
    "three_bet_freq": (0.10, 6.0),
    "fold_to_three_bet": (0.50, 6.0),
    "cbet_freq": (0.60, 5.0),
    "fold_to_cbet": (0.45, 5.0),
    "aggression_frequency": (0.38, 8.0),
    "fold_to_bet": (0.42, 6.0),
    "fold_to_river_bet": (0.45, 5.0),
    "showdown_frequency": (0.32, 6.0),
    "wtsd_when_saw_flop": (0.48, 5.0),
    "won_at_showdown": (0.50, 5.0),
    "avg_bet_size_pot_ratio": (0.65, 4.0),
    "avg_raise_size_pot_ratio": (0.75, 4.0),
    "vpip_button": (0.55, 5.0),
    "vpip_big_blind": (0.35, 5.0),
    "pfr_button": (0.38, 5.0),
    "pfr_big_blind": (0.18, 5.0),
}


def compute_beta_metric(
    name: str,
    successes: int,
    trials: int,
    weighted_successes: float,
    weighted_trials: float,
    *,
    prior_mean: float | None = None,
    prior_strength: float | None = None,
) -> MetricEstimate:
    """Compute a Beta-smoothed proportion metric with 95% credible bounds."""
    default_mean, default_kappa = PRIORS.get(name, (0.5, 5.0))
    mu0 = default_mean if prior_mean is None else float(prior_mean)
    kappa0 = (
        default_kappa if prior_strength is None else max(0.5, float(prior_strength))
    )

    alpha0 = mu0 * kappa0
    beta0 = (1.0 - mu0) * kappa0

    raw_val = (successes / trials) if trials > 0 else None
    post_alpha = alpha0 + max(0.0, weighted_successes)
    post_beta = beta0 + max(0.0, weighted_trials - weighted_successes)
    post_total = post_alpha + post_beta

    smoothed = post_alpha / post_total if post_total > 0 else mu0
    variance = (smoothed * (1.0 - smoothed)) / max(1.0, post_total + 1.0)
    std_err = math.sqrt(max(0.0, variance))
    ci_low = max(0.0, smoothed - 1.96 * std_err)
    ci_high = min(1.0, smoothed + 1.96 * std_err)

    confidence = (
        weighted_trials / (weighted_trials + kappa0) if weighted_trials > 0 else 0.0
    )
    uncertainty = 1.0 - confidence

    return MetricEstimate(
        name=name,
        successes=int(successes),
        trials=int(trials),
        weighted_successes=float(weighted_successes),
        weighted_trials=float(weighted_trials),
        raw_value=raw_val,
        smoothed_value=smoothed,
        prior_mean=mu0,
        prior_strength=kappa0,
        ci95_low=ci_low,
        ci95_high=ci_high,
        uncertainty=uncertainty,
        confidence=confidence,
    )


def compute_continuous_metric(
    name: str,
    count: int,
    raw_sum: float,
    weighted_count: float,
    weighted_sum: float,
    weighted_sq_sum: float,
    *,
    prior_mean: float | None = None,
    prior_strength: float | None = None,
) -> MetricEstimate:
    """Compute a shrinkage-smoothed continuous ratio metric (e.g. bet size / pot)."""
    default_mean, default_kappa = PRIORS.get(name, (0.65, 4.0))
    mu0 = default_mean if prior_mean is None else float(prior_mean)
    kappa0 = (
        default_kappa if prior_strength is None else max(0.5, float(prior_strength))
    )

    raw_val = (raw_sum / count) if count > 0 else None
    denom = kappa0 + max(0.0, weighted_count)
    smoothed = (kappa0 * mu0 + max(0.0, weighted_sum)) / denom if denom > 0 else mu0

    if weighted_count > 1.0 and raw_val is not None:
        mean_w = weighted_sum / weighted_count
        sample_var = max(0.04, (weighted_sq_sum / weighted_count) - mean_w * mean_w)
    else:
        sample_var = 0.16
    std_err = math.sqrt(sample_var / max(1.0, denom))
    ci_low = max(0.0, smoothed - 1.96 * std_err)
    ci_high = max(ci_low, smoothed + 1.96 * std_err)

    confidence = (
        weighted_count / (weighted_count + kappa0) if weighted_count > 0 else 0.0
    )
    uncertainty = 1.0 - confidence

    return MetricEstimate(
        name=name,
        successes=round(raw_sum * 100),
        trials=int(count),
        weighted_successes=float(weighted_sum),
        weighted_trials=float(weighted_count),
        raw_value=raw_val,
        smoothed_value=smoothed,
        prior_mean=mu0,
        prior_strength=kappa0,
        ci95_low=ci_low,
        ci95_high=ci_high,
        uncertainty=uncertainty,
        confidence=confidence,
    )


class BehaviouralProfiler:
    """Computes smoothed opponent poker statistics with recency weighting."""

    def __init__(self, *, recency_decay: float = 0.94) -> None:
        if not 0.0 < recency_decay <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        self.recency_decay = float(recency_decay)
        self._feature_cache: dict[
            tuple[str, int, int, bool, bool], dict[str, int | float | bool]
        ] = {}

    @staticmethod
    def _hand_features(hand: HandObservationRecord) -> dict[str, int | float | bool]:
        """Extract single-hand indicator counts from a `HandObservationRecord`."""
        pre = hand.preflop_actions
        opp_pre = [a for a in pre if a.is_opponent]

        has_pre_opp = bool(opp_pre) or hand.completed or len(hand.actions) > 0
        vpip_hit = any(
            a.action_type
            in (
                ActionType.CALL.value,
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
            for a in opp_pre
        )
        pfr_hit = any(
            a.action_type
            in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
            for a in opp_pre
        )

        # 3-bet opportunity: opponent acts preflop facing exactly 1 prior raise
        three_bet_opp = False
        three_bet_hit = False
        fold_to_3bet_opp = False
        fold_to_3bet_hit = False

        opp_raised_pre = False
        for a in pre:
            if a.is_opponent:
                if a.facing_raise_count == 1 and not opp_raised_pre:
                    three_bet_opp = True
                    if a.action_type in (
                        ActionType.RAISE.value,
                        ActionType.ALL_IN.value,
                    ):
                        three_bet_hit = True
                if opp_raised_pre and a.facing_raise_count >= 2 and a.facing_bet:
                    fold_to_3bet_opp = True
                    if a.action_type == ActionType.FOLD.value:
                        fold_to_3bet_hit = True
                if a.action_type in (
                    ActionType.BET.value,
                    ActionType.RAISE.value,
                    ActionType.ALL_IN.value,
                ):
                    opp_raised_pre = True

        # Last preflop aggressor for C-Bet tracking
        pre_aggressors = [
            a
            for a in pre
            if a.action_type
            in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
        ]
        last_pre_aggressor_is_opp = (
            pre_aggressors[-1].is_opponent if pre_aggressors else None
        )

        flop = hand.flop_actions
        cbet_opp = False
        cbet_hit = False
        fold_to_cbet_opp = False
        fold_to_cbet_hit = False

        if flop and last_pre_aggressor_is_opp is True:
            # Opponent was preflop aggressor; check first flop decision if not facing a donk bet
            opp_flop = [a for a in flop if a.is_opponent]
            if opp_flop and not opp_flop[0].facing_bet:
                cbet_opp = True
                if opp_flop[0].action_type in (
                    ActionType.BET.value,
                    ActionType.ALL_IN.value,
                ):
                    cbet_hit = True
        elif flop and last_pre_aggressor_is_opp is False:
            # Hero was preflop aggressor; did hero c-bet and opponent face it?
            hero_flop_bets = [
                a
                for a in flop
                if not a.is_opponent
                and not a.facing_bet
                and a.action_type in (ActionType.BET.value, ActionType.ALL_IN.value)
            ]
            if hero_flop_bets:
                opp_responses = [
                    a
                    for a in flop
                    if a.is_opponent
                    and a.event_index > hero_flop_bets[0].event_index
                    and a.facing_bet
                ]
                if opp_responses:
                    fold_to_cbet_opp = True
                    if opp_responses[0].action_type == ActionType.FOLD.value:
                        fold_to_cbet_hit = True

        # Action counts across all streets for aggression and fold metrics
        opp_all = hand.opponent_actions
        n_agg = sum(
            1
            for a in opp_all
            if a.action_type
            in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
        )
        n_call = sum(1 for a in opp_all if a.action_type == ActionType.CALL.value)
        n_fold = sum(1 for a in opp_all if a.action_type == ActionType.FOLD.value)
        n_afq_denom = n_agg + n_call + n_fold

        facing_bet_actions = [a for a in opp_all if a.facing_bet]
        n_face_bet = len(facing_bet_actions)
        n_fold_to_bet = sum(
            1 for a in facing_bet_actions if a.action_type == ActionType.FOLD.value
        )

        river_facing_bet = [a for a in facing_bet_actions if a.street == "river"]
        n_face_river_bet = len(river_facing_bet)
        n_fold_river_bet = sum(
            1 for a in river_facing_bet if a.action_type == ActionType.FOLD.value
        )

        bet_ratios = [
            min(5.0, max(0.05, a.pot_ratio))
            for a in opp_all
            if a.action_type == ActionType.BET.value and a.pot_ratio > 0
        ]
        raise_ratios = [
            min(5.0, max(0.05, a.pot_ratio))
            for a in opp_all
            if a.action_type in (ActionType.RAISE.value, ActionType.ALL_IN.value)
            and a.pot_ratio > 0
        ]

        return {
            "has_pre_opp": has_pre_opp,
            "vpip_hit": int(vpip_hit),
            "pfr_hit": int(pfr_hit),
            "is_button": hand.opponent_is_button,
            "three_bet_opp": int(three_bet_opp),
            "three_bet_hit": int(three_bet_hit),
            "fold_to_3bet_opp": int(fold_to_3bet_opp),
            "fold_to_3bet_hit": int(fold_to_3bet_hit),
            "cbet_opp": int(cbet_opp),
            "cbet_hit": int(cbet_hit),
            "fold_to_cbet_opp": int(fold_to_cbet_opp),
            "fold_to_cbet_hit": int(fold_to_cbet_hit),
            "n_agg": n_agg,
            "n_call": n_call,
            "n_afq_denom": n_afq_denom,
            "n_face_bet": n_face_bet,
            "n_fold_to_bet": n_fold_to_bet,
            "n_face_river_bet": n_face_river_bet,
            "n_fold_river_bet": n_fold_river_bet,
            "bet_ratios": bet_ratios,
            "raise_ratios": raise_ratios,
        }

    def compute_snapshot(
        self,
        hands: Sequence[HandObservationRecord],
        *,
        opponent_id: str = "opponent",
    ) -> OpponentStatsSnapshot:
        """Compute a complete `OpponentStatsSnapshot` over `hands`."""
        n_hands = len(hands)
        completed_hands = sum(1 for h in hands if h.completed)
        showdown_hands = sum(1 for h in hands if h.completed and h.reached_showdown)

        # Raw integer counters
        vpip_s = vpip_n = 0
        pfr_s = pfr_n = 0
        vpip_btn_s = vpip_btn_n = 0
        vpip_bb_s = vpip_bb_n = 0
        pfr_btn_s = pfr_btn_n = 0
        pfr_bb_s = pfr_bb_n = 0

        tbet_s = tbet_n = 0
        f3bet_s = f3bet_n = 0
        cbet_s = cbet_n = 0
        fcbet_s = fcbet_n = 0

        afq_s = afq_n = 0
        raw_agg_total = raw_call_total = 0

        fbet_s = fbet_n = 0
        friv_s = friv_n = 0

        wtsd_s = wtsd_n = 0
        wtsd_flop_s = wtsd_flop_n = 0
        wsd_s = wsd_n = 0

        bet_cnt = 0
        bet_sum = 0.0
        raise_cnt = 0
        raise_sum = 0.0

        # Recency-weighted counters
        w_vpip_s = w_vpip_n = 0.0
        w_pfr_s = w_pfr_n = 0.0
        w_vpip_btn_s = w_vpip_btn_n = 0.0
        w_vpip_bb_s = w_vpip_bb_n = 0.0
        w_pfr_btn_s = w_pfr_btn_n = 0.0
        w_pfr_bb_s = w_pfr_bb_n = 0.0

        w_tbet_s = w_tbet_n = 0.0
        w_f3bet_s = w_f3bet_n = 0.0
        w_cbet_s = w_cbet_n = 0.0
        w_fcbet_s = w_fcbet_n = 0.0

        w_afq_s = w_afq_n = 0.0
        w_agg_total = w_call_total = 0.0

        w_fbet_s = w_fbet_n = 0.0
        w_friv_s = w_friv_n = 0.0

        w_wtsd_s = w_wtsd_n = 0.0
        w_wtsd_flop_s = w_wtsd_flop_n = 0.0
        w_wsd_s = w_wsd_n = 0.0

        w_bet_cnt = w_bet_sum = w_bet_sq = 0.0
        w_raise_cnt = w_raise_sum = w_raise_sq = 0.0

        hand_feats: list[dict[str, int | float | bool]] = []
        for hand in hands:
            cache_key = (
                hand.opponent_id,
                hand.hand_number,
                len(hand.actions),
                hand.completed,
                hand.reached_showdown,
            )
            feat = self._feature_cache.get(cache_key)
            if feat is None:
                feat = self._hand_features(hand)
                if hand.completed:
                    self._feature_cache[cache_key] = feat
            hand_feats.append(feat)

        stability, changepoint_idx, drift_mag = self._analyze_drift(hand_feats)
        # When a strong behavioral shift is detected at `changepoint_idx`, downweight
        # pre-shift observations more aggressively so stale regime evidence does not dominate.
        pre_shift_penalty = (
            max(0.18, 1.0 - 1.45 * max(0.0, drift_mag - 0.16))
            if changepoint_idx > 0
            else 1.0
        )

        for idx, hand in enumerate(hands):
            age = max(0, n_hands - 1 - idx)
            w = self.recency_decay**age
            if idx < changepoint_idx:
                w *= pre_shift_penalty
            feat = hand_feats[idx]

            if feat["has_pre_opp"]:
                vpip_hit = int(feat["vpip_hit"])
                pfr_hit = int(feat["pfr_hit"])
                vpip_n += 1
                vpip_s += vpip_hit
                pfr_n += 1
                pfr_s += pfr_hit

                w_vpip_n += w
                w_vpip_s += w * vpip_hit
                w_pfr_n += w
                w_pfr_s += w * pfr_hit

                if feat["is_button"]:
                    vpip_btn_n += 1
                    vpip_btn_s += vpip_hit
                    pfr_btn_n += 1
                    pfr_btn_s += pfr_hit
                    w_vpip_btn_n += w
                    w_vpip_btn_s += w * vpip_hit
                    w_pfr_btn_n += w
                    w_pfr_btn_s += w * pfr_hit
                else:
                    vpip_bb_n += 1
                    vpip_bb_s += vpip_hit
                    pfr_bb_n += 1
                    pfr_bb_s += pfr_hit
                    w_vpip_bb_n += w
                    w_vpip_bb_s += w * vpip_hit
                    w_pfr_bb_n += w
                    w_pfr_bb_s += w * pfr_hit

            if feat["three_bet_opp"]:
                hit = int(feat["three_bet_hit"])
                tbet_n += 1
                tbet_s += hit
                w_tbet_n += w
                w_tbet_s += w * hit

            if feat["fold_to_3bet_opp"]:
                hit = int(feat["fold_to_3bet_hit"])
                f3bet_n += 1
                f3bet_s += hit
                w_f3bet_n += w
                w_f3bet_s += w * hit

            if feat["cbet_opp"]:
                hit = int(feat["cbet_hit"])
                cbet_n += 1
                cbet_s += hit
                w_cbet_n += w
                w_cbet_s += w * hit

            if feat["fold_to_cbet_opp"]:
                hit = int(feat["fold_to_cbet_hit"])
                fcbet_n += 1
                fcbet_s += hit
                w_fcbet_n += w
                w_fcbet_s += w * hit

            n_agg = int(feat["n_agg"])
            n_call = int(feat["n_call"])
            n_afq = int(feat["n_afq_denom"])
            raw_agg_total += n_agg
            raw_call_total += n_call
            w_agg_total += w * n_agg
            w_call_total += w * n_call

            afq_n += n_afq
            afq_s += n_agg
            w_afq_n += w * n_afq
            w_afq_s += w * n_agg

            n_fb = int(feat["n_face_bet"])
            n_flb = int(feat["n_fold_to_bet"])
            fbet_n += n_fb
            fbet_s += n_flb
            w_fbet_n += w * n_fb
            w_fbet_s += w * n_flb

            n_frb = int(feat["n_face_river_bet"])
            n_flrb = int(feat["n_fold_river_bet"])
            friv_n += n_frb
            friv_s += n_flrb
            w_friv_n += w * n_frb
            w_friv_s += w * n_flrb

            for r in feat["bet_ratios"]:  # type: ignore[union-attr]
                val = float(r)
                bet_cnt += 1
                bet_sum += val
                w_bet_cnt += w
                w_bet_sum += w * val
                w_bet_sq += w * val * val

            for r in feat["raise_ratios"]:  # type: ignore[union-attr]
                val = float(r)
                raise_cnt += 1
                raise_sum += val
                w_raise_cnt += w
                w_raise_sum += w * val
                w_raise_sq += w * val * val

            if hand.completed:
                sd_hit = 1 if hand.reached_showdown else 0
                wtsd_n += 1
                wtsd_s += sd_hit
                w_wtsd_n += w
                w_wtsd_s += w * sd_hit

                if hand.saw_flop:
                    wtsd_flop_n += 1
                    wtsd_flop_s += sd_hit
                    w_wtsd_flop_n += w
                    w_wtsd_flop_s += w * sd_hit

                if hand.reached_showdown:
                    won_hit = 1 if hand.opponent_won else 0
                    wsd_n += 1
                    wsd_s += won_hit
                    w_wsd_n += w
                    w_wsd_s += w * won_hit

        vpip_est = compute_beta_metric("vpip", vpip_s, vpip_n, w_vpip_s, w_vpip_n)
        pfr_est = compute_beta_metric("pfr", pfr_s, pfr_n, w_pfr_s, w_pfr_n)
        vpip_minus_pfr = max(0.0, vpip_est.smoothed_value - pfr_est.smoothed_value)

        three_bet_est = compute_beta_metric(
            "three_bet_freq", tbet_s, tbet_n, w_tbet_s, w_tbet_n
        )
        fold_3bet_est = compute_beta_metric(
            "fold_to_three_bet", f3bet_s, f3bet_n, w_f3bet_s, w_f3bet_n
        )
        cbet_est = compute_beta_metric("cbet_freq", cbet_s, cbet_n, w_cbet_s, w_cbet_n)
        fold_cbet_est = compute_beta_metric(
            "fold_to_cbet", fcbet_s, fcbet_n, w_fcbet_s, w_fcbet_n
        )
        afq_est = compute_beta_metric(
            "aggression_frequency", afq_s, afq_n, w_afq_s, w_afq_n
        )

        # Aggression Factor: smoothed with prior (3.0 agg / 3.0 call -> prior AF = 1.0)
        smoothed_af = (3.0 + w_agg_total) / (3.0 + w_call_total)
        raw_af: float | None = None
        if raw_call_total > 0:
            raw_af = raw_agg_total / raw_call_total
        elif raw_agg_total > 0:
            raw_af = float(raw_agg_total)

        fold_bet_est = compute_beta_metric(
            "fold_to_bet", fbet_s, fbet_n, w_fbet_s, w_fbet_n
        )
        fold_river_est = compute_beta_metric(
            "fold_to_river_bet", friv_s, friv_n, w_friv_s, w_friv_n
        )
        wtsd_est = compute_beta_metric(
            "showdown_frequency", wtsd_s, wtsd_n, w_wtsd_s, w_wtsd_n
        )
        wtsd_flop_est = compute_beta_metric(
            "wtsd_when_saw_flop",
            wtsd_flop_s,
            wtsd_flop_n,
            w_wtsd_flop_s,
            w_wtsd_flop_n,
        )
        wsd_est = compute_beta_metric("won_at_showdown", wsd_s, wsd_n, w_wsd_s, w_wsd_n)

        bet_size_est = compute_continuous_metric(
            "avg_bet_size_pot_ratio",
            bet_cnt,
            bet_sum,
            w_bet_cnt,
            w_bet_sum,
            w_bet_sq,
        )
        raise_size_est = compute_continuous_metric(
            "avg_raise_size_pot_ratio",
            raise_cnt,
            raise_sum,
            w_raise_cnt,
            w_raise_sum,
            w_raise_sq,
        )

        vpip_btn_est = compute_beta_metric(
            "vpip_button", vpip_btn_s, vpip_btn_n, w_vpip_btn_s, w_vpip_btn_n
        )
        vpip_bb_est = compute_beta_metric(
            "vpip_big_blind", vpip_bb_s, vpip_bb_n, w_vpip_bb_s, w_vpip_bb_n
        )
        pfr_btn_est = compute_beta_metric(
            "pfr_button", pfr_btn_s, pfr_btn_n, w_pfr_btn_s, w_pfr_btn_n
        )
        pfr_bb_est = compute_beta_metric(
            "pfr_big_blind", pfr_bb_s, pfr_bb_n, w_pfr_bb_s, w_pfr_bb_n
        )

        return OpponentStatsSnapshot(
            opponent_id=opponent_id,
            hands_observed=n_hands,
            completed_hands=completed_hands,
            showdowns_observed=showdown_hands,
            recency_decay=self.recency_decay,
            stability_score=stability,
            vpip=vpip_est,
            pfr=pfr_est,
            vpip_minus_pfr=vpip_minus_pfr,
            three_bet_freq=three_bet_est,
            fold_to_three_bet=fold_3bet_est,
            cbet_freq=cbet_est,
            fold_to_cbet=fold_cbet_est,
            aggression_frequency=afq_est,
            aggression_factor=smoothed_af,
            raw_aggression_factor=raw_af,
            fold_to_bet=fold_bet_est,
            fold_to_river_bet=fold_river_est,
            showdown_frequency=wtsd_est,
            wtsd_when_saw_flop=wtsd_flop_est,
            won_at_showdown=wsd_est,
            avg_bet_size_pot_ratio=bet_size_est,
            avg_raise_size_pot_ratio=raise_size_est,
            vpip_button=vpip_btn_est,
            vpip_big_blind=vpip_bb_est,
            pfr_button=pfr_btn_est,
            pfr_big_blind=pfr_bb_est,
        )

    @staticmethod
    def _window_stats(
        window: Sequence[dict[str, int | float | bool]],
    ) -> tuple[float, float, float, float]:
        n_pre = max(1, sum(1 for f in window if f["has_pre_opp"]))
        v = sum(int(f["vpip_hit"]) for f in window if f["has_pre_opp"]) / n_pre
        p = sum(int(f["pfr_hit"]) for f in window if f["has_pre_opp"]) / n_pre
        af_d = sum(int(f["n_afq_denom"]) for f in window)
        a = sum(int(f["n_agg"]) for f in window) / af_d if af_d > 0 else 0.38
        fb_d = sum(int(f["n_face_bet"]) for f in window)
        fb = sum(int(f["n_fold_to_bet"]) for f in window) / fb_d if fb_d > 0 else 0.42
        return v, p, a, fb

    @classmethod
    def _analyze_drift(
        cls,
        hand_feats: Sequence[dict[str, int | float | bool]],
    ) -> tuple[float, int, float]:
        """Detect behavioral drift between recent and preceding windows.

        Returns `(stability_score, changepoint_idx, max_drift)`.
        """
        n = len(hand_feats)
        if n < 10:
            return 1.0, 0, 0.0

        # Evaluate candidate split points in the last 6..18 hands against the preceding 10..20 hands
        best_cp = 0
        max_drift = 0.0
        min_rec = max(5, min(8, n // 3))
        max_rec = min(16, n - 5)

        for rec_len in range(min_rec, max_rec + 1, 2):
            cp = n - rec_len
            prev_start = max(0, cp - 18)
            prev_win = hand_feats[prev_start:cp]
            rec_win = hand_feats[cp:]
            if len(prev_win) < 5 or len(rec_win) < 5:
                continue
            v_old, p_old, a_old, f_old = cls._window_stats(prev_win)
            v_rec, p_rec, a_rec, f_rec = cls._window_stats(rec_win)
            drift = (
                0.30 * abs(v_rec - v_old)
                + 0.30 * abs(p_rec - p_old)
                + 0.25 * abs(a_rec - a_old)
                + 0.15 * abs(f_rec - f_old)
            )
            if drift > max_drift:
                max_drift = drift
                best_cp = cp

        # Immediate stability uses the most recent 8-hand window vs preceding 12-hand window
        imm_rec_len = min(8, n // 2)
        imm_cp = n - imm_rec_len
        imm_prev = hand_feats[max(0, imm_cp - 12) : imm_cp]
        imm_rec = hand_feats[imm_cp:]
        v_o, p_o, a_o, f_o = cls._window_stats(imm_prev)
        v_r, p_r, a_r, f_r = cls._window_stats(imm_rec)
        imm_drift = (
            0.30 * abs(v_r - v_o)
            + 0.30 * abs(p_r - p_o)
            + 0.25 * abs(a_r - a_o)
            + 0.15 * abs(f_r - f_o)
        )

        effective_drift = 0.65 * imm_drift + 0.35 * max_drift
        stability = max(0.20, min(1.0, 1.0 - 1.85 * max(0.0, effective_drift - 0.08)))
        changepoint = best_cp if max_drift >= 0.18 else 0
        return stability, changepoint, max_drift

    @classmethod
    def _compute_stability_score(
        cls,
        hand_feats: Sequence[dict[str, int | float | bool]],
    ) -> float:
        """Measure behavioral consistency between earlier and recent hands in [0.2, 1.0]."""
        stability, _, _ = cls._analyze_drift(hand_feats)
        return stability
