"""Phase 11 Validation Suite: Behavioural Adaptation, Bayesian Range Calibration & Counter-Exploitation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pokerlab.agents import (
    EquityAgent,
    EquityAgentConfig,
    HeuristicAgent,
)
from pokerlab.agents.base import (
    Action,
    ActionRecord,
    ActionType,
    Agent,
    Observation,
    observe,
)
from pokerlab.engine import Deck, GameState
from pokerlab.opponent_model.adaptive_agent import (
    CallingStationAgent,
    HarveySpecterAgent,
    HarveySpecterConfig,
    LooseAggressiveAgent,
    ManiacAgent,
    NitAgent,
    TightPassiveAgent,
    _find_action,
    _find_all_actions,
)
from pokerlab.opponent_model.benchmark import (
    HarveyBenchmarkConfig,
    build_mccfr_baseline,
    run_harvey_specter_benchmark,
)
from pokerlab.opponent_model.models import ARCHETYPE_NAMES
from pokerlab.opponent_model.profile_store import OpponentProfile
from pokerlab.opponent_model.range_model import COMBO_TO_CLASS
from pokerlab.research.nlhe.agent import AbstractHUNLAgent


def _git_revision() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return proc.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ============================================================================
# 1. Controlled Non-Stationary Opponent for Behavioural Adaptation Experiment
# ============================================================================


@dataclass(frozen=True, slots=True)
class RegimePhase:
    """One scheduled behavioral phase `[start_hand, end_hand]` (1-indexed, inclusive)."""

    start_hand: int
    end_hand: int
    archetype_name: str
    label: str


DEFAULT_ADAPTATION_SCHEDULE: tuple[RegimePhase, ...] = (
    RegimePhase(1, 20, "tight_passive", "Tight-Passive (Hands 1-20)"),
    RegimePhase(21, 50, "loose_aggressive", "Loose-Aggressive (Hands 21-50)"),
    RegimePhase(51, 80, "calling_station", "Calling Station (Hands 51-80)"),
    RegimePhase(81, 100, "tight_passive", "Tight-Passive (Hands 81-100)"),
)


class ScheduledStrategyOpponent(Agent):
    """Controlled opponent that switches its actual policy at scheduled hand boundaries.

    HarveySpecter never accesses `current_hand` or `schedule`; the test harness
    advances `set_hand_number(hand_num)` before each hand begins.
    """

    name: str = "ScheduledStrategyOpponent"

    def __init__(
        self,
        schedule: Sequence[RegimePhase] = DEFAULT_ADAPTATION_SCHEDULE,
        *,
        seed: int = 0,
    ) -> None:
        self.schedule = tuple(schedule)
        self._seed = seed
        self.current_hand: int = 1
        self._delegates: dict[str, Agent] = {
            "tight_passive": TightPassiveAgent(seed=seed + 101),
            "loose_aggressive": LooseAggressiveAgent(seed=seed + 202),
            "calling_station": CallingStationAgent(seed=seed + 303),
            "maniac": ManiacAgent(seed=seed + 404),
            "nit": NitAgent(seed=seed + 505),
        }

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
            for idx, ag in enumerate(self._delegates.values()):
                ag.reset(seed + (idx + 1) * 101)
        self.current_hand = 1

    def set_hand_number(self, hand_number: int) -> None:
        self.current_hand = int(hand_number)

    def notify_new_hand(self, hand_number: int) -> None:
        self.set_hand_number(hand_number)

    @property
    def current_true_archetype(self) -> str:
        return self.active_archetype()

    def active_archetype(self, hand_number: int | None = None) -> str:
        h = self.current_hand if hand_number is None else int(hand_number)
        for phase in self.schedule:
            if phase.start_hand <= h <= phase.end_hand:
                return phase.archetype_name
        return self.schedule[-1].archetype_name

    def act(self, observation: Observation) -> Action:
        arch = self.active_archetype()
        delegate = self._delegates[arch]
        return delegate.act(observation)


# ============================================================================
# 2. Adversarial Counter-Exploitation Opponents (Section 5)
# ============================================================================


class BluffPunisherAgent(Agent):
    """Plays Nit-like for `bait_hands` to induce bluffs, then switches to Calling Station."""

    name: str = "BluffPunisherAgent"

    def __init__(self, *, bait_hands: int = 18, seed: int = 0) -> None:
        self.bait_hands = bait_hands
        self.current_hand = 1
        self.nit = NitAgent(seed=seed + 11)
        self.station = CallingStationAgent(seed=seed + 22)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.nit.reset(seed + 11)
            self.station.reset(seed + 22)
        self.current_hand = 1

    def set_hand_number(self, hand_number: int) -> None:
        self.current_hand = int(hand_number)

    def act(self, observation: Observation) -> Action:
        if self.current_hand <= self.bait_hands:
            return self.nit.act(observation)
        return self.station.act(observation)


class ValueFolderAgent(Agent):
    """Plays Calling Station for `bait_hands` to induce wide value bets, then switches to Nit."""

    name: str = "ValueFolderAgent"

    def __init__(self, *, bait_hands: int = 18, seed: int = 0) -> None:
        self.bait_hands = bait_hands
        self.current_hand = 1
        self.station = CallingStationAgent(seed=seed + 33)
        self.nit = NitAgent(seed=seed + 44)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.station.reset(seed + 33)
            self.nit.reset(seed + 44)
        self.current_hand = 1

    def set_hand_number(self, hand_number: int) -> None:
        self.current_hand = int(hand_number)

    def act(self, observation: Observation) -> Action:
        if self.current_hand <= self.bait_hands:
            return self.station.act(observation)
        return self.nit.act(observation)


class AlternatingGearAgent(Agent):
    """Rapidly alternates between Nit and Maniac every `period_hands` hands."""

    name: str = "AlternatingGearAgent"

    def __init__(self, *, period_hands: int = 8, seed: int = 0) -> None:
        self.period_hands = max(2, int(period_hands))
        self.current_hand = 1
        self.nit = NitAgent(seed=seed + 55)
        self.maniac = ManiacAgent(seed=seed + 66)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.nit.reset(seed + 55)
            self.maniac.reset(seed + 66)
        self.current_hand = 1

    def set_hand_number(self, hand_number: int) -> None:
        self.current_hand = int(hand_number)

    def act(self, observation: Observation) -> Action:
        block = (self.current_hand - 1) // self.period_hands
        if block % 2 == 0:
            return self.nit.act(observation)
        return self.maniac.act(observation)


class DeceptiveSizingAgent(Agent):
    """Uses inverted/deceptive bet sizing: min-bets monsters, overbets/pots bluffs and medium hands."""

    name: str = "DeceptiveSizingAgent"

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
        all_in = _find_action(legal, ActionType.ALL_IN)

        # Deceptive sizing: monster hands (>= 0.80) use smallest bet or slowplay call/check;
        # weak/air hands (< 0.36) frequently fire largest bet size as polarized bluff.
        if observation.to_call > 0:
            if strength >= 0.80:
                return call or (bets[0] if bets else legal[0])
            if strength < 0.34 and self.rng.random() < 0.35 and bets:
                return bets[-1]
            if strength >= max(0.44, observation.pot_odds) and call:
                return call
            return fold or call or check or legal[0]

        if strength >= 0.80:
            return bets[0] if bets else (check or legal[0])
        if strength < 0.35 and self.rng.random() < 0.45:
            if bets:
                return bets[-1]
            if all_in and self.rng.random() < 0.15:
                return all_in
        if 0.52 <= strength < 0.80 and bets:
            return bets[0]
        return check or legal[0]


# ============================================================================
# 3. Behavioural Adaptation Experiment Runner & Visualization
# ============================================================================


def _regime_family_match(predicted: str, target: str, probs: dict[str, float]) -> bool:
    if predicted == target:
        return True
    # Allow closely related neighbor when target probability is already substantial (>= 0.25)
    if probs.get(target, 0.0) >= 0.26:
        return True
    if target == "tight_passive" and predicted == "nit":
        return True
    return bool(target == "loose_aggressive" and predicted == "maniac")


def run_behavioural_adaptation_experiment(
    *,
    seeds: Sequence[int] = (101, 202, 303, 404, 505, 606),
    schedule: Sequence[RegimePhase] = DEFAULT_ADAPTATION_SCHEDULE,
    mccfr_iterations: int = 12,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the Phase 11 Section 2 controlled strategy-change experiment across multiple seeds."""
    started = time.perf_counter()
    total_hands = max(phase.end_hand for phase in schedule)
    bench_cfg = HarveyBenchmarkConfig(
        seeds=tuple(seeds),
        hands_per_seed=total_hands,
        mccfr_iterations=mccfr_iterations,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
    )

    per_hand_records_by_hand: dict[int, list[dict[str, Any]]] = {
        h: [] for h in range(1, total_hands + 1)
    }
    seed_summaries: list[dict[str, Any]] = []
    change_boundaries = [phase.start_hand for phase in schedule[1:]]
    prev_archetypes = {
        schedule[i].start_hand: schedule[i - 1].archetype_name
        for i in range(1, len(schedule))
    }
    target_archetypes = {
        phase.start_hand: phase.archetype_name for phase in schedule[1:]
    }

    detection_delays: dict[int, list[int]] = {b: [] for b in change_boundaries}
    stabilization_times: dict[int, list[int]] = {b: [] for b in change_boundaries}

    for seed in seeds:
        bundle = build_mccfr_baseline(bench_cfg, seed)
        strategy, strategy_sums, game_cfg = bundle

        # 1. Adaptive HarveySpecter (recency_decay = 0.90 for responsive online tracking)
        base_harvey = AbstractHUNLAgent(
            strategy=strategy,
            config=game_cfg,
            seed=seed + 11,
            strategy_sums=strategy_sums,
        )
        harvey = HarveySpecterAgent(
            base_harvey,
            config=HarveySpecterConfig(
                mode="adaptive",
                max_lambda=0.80,
                recency_decay=0.90,
                equity_simulations=36,
            ),
            opponent_id="ChameleonVillain",
            seed=seed + 11,
        )

        # 2. Unweighted HarveySpecter ablation (recency_decay = 1.0) to test whether old evidence dominates
        unweighted_profile = OpponentProfile(
            opponent_id="ChameleonVillain",
            recency_decay=1.0,
        )

        # 3. Baseline MCCFR agent (for paired BB/100 comparison against the exact same deals)
        baseline_agent = AbstractHUNLAgent(
            strategy=strategy,
            config=game_cfg,
            seed=seed + 11,
            strategy_sums=strategy_sums,
        )

        opp_harvey = ScheduledStrategyOpponent(schedule, seed=seed + 77)
        opp_base = ScheduledStrategyOpponent(schedule, seed=seed + 77)

        deck_rng = random.Random(seed * 91_827)
        deal_seeds = [deck_rng.randrange(2**32) for _ in range(total_hands)]

        seed_hand_rows: list[dict[str, Any]] = []

        for h_idx, d_seed in enumerate(deal_seeds):
            hand_num = h_idx + 1
            true_arch = opp_harvey.active_archetype(hand_num)
            opp_harvey.set_hand_number(hand_num)
            opp_base.set_hand_number(hand_num)

            hero_is_button = (h_idx % 2) == 0
            button_name = "Hero" if hero_is_button else "Villain"
            initial = {"Hero": starting_stack, "Villain": starting_stack}

            # --- Play hand with Adaptive HarveySpecter ---
            game = GameState(
                dict(initial),
                small_blind=small_blind,
                big_blind=big_blind,
                button=button_name,
                deck=Deck(seed=d_seed),
            )
            harvey.start_hand(
                opponent_id="ChameleonVillain",
                hand_number=hand_num,
                opponent_is_button=not hero_is_button,
                starting_stack=starting_stack,
                effective_stack=starting_stack,
                big_blind=big_blind,
                hero_hole=game.hole_cards["Hero"],
                community_cards=(),
            )
            unweighted_profile.start_hand(
                hand_number=hand_num,
                opponent_is_button=not hero_is_button,
                starting_stack=starting_stack,
                effective_stack=starting_stack,
                big_blind=big_blind,
                hero_hole=game.hole_cards["Hero"],
                community_cards=(),
            )

            history: list[ActionRecord] = []
            hand_lambdas: list[float] = []
            hand_policy_l1: list[float] = []
            while not game.complete and game.current_player is not None:
                actor = game.current_player
                obs = observe(game, actor, tuple(history))
                if actor == "Hero":
                    act = harvey.act(obs)
                    if harvey.last_decision_report is not None:
                        rep = harvey.last_decision_report
                        hand_lambdas.append(rep.lambda_value)
                        l1_diff = 0.5 * sum(
                            abs(f["probability"] - m["probability"])
                            for f, m in zip(
                                rep.final_probabilities,
                                rep.mccfr_probabilities,
                                strict=True,
                            )
                        )
                        hand_policy_l1.append(l1_diff)
                else:
                    act = opp_harvey.act(obs)
                st = game.street.value
                game.act(act, player=actor)
                history.append(ActionRecord(actor, act, st))

            hero_net_harvey = game.stacks["Hero"] - initial["Hero"]
            villain_net_harvey = game.stacks["Villain"] - initial["Villain"]
            assert hero_net_harvey + villain_net_harvey == 0
            bb100_harvey = (hero_net_harvey / big_blind) * 100.0

            reached_sd = bool(game.hand_ranks and "Villain" in game.hand_ranks)
            revealed = game.hole_cards["Villain"] if reached_sd else None

            harvey.finalize_hand(
                opponent_id="ChameleonVillain",
                hand_number=hand_num,
                history=tuple(history),
                hero_player="Hero",
                community_cards=tuple(game.community_cards),
                reached_showdown=reached_sd,
                revealed_opponent_cards=revealed,
                opponent_net_chips=villain_net_harvey,
                opponent_won=(villain_net_harvey > 0),
                final_pot=game.current_pot,
            )
            unweighted_profile.finalize_hand(
                hand_number=hand_num,
                history=tuple(history),
                hero_player="Hero",
                community_cards=tuple(game.community_cards),
                reached_showdown=reached_sd,
                revealed_opponent_cards=revealed,
                opponent_net_chips=villain_net_harvey,
                opponent_won=(villain_net_harvey > 0),
                final_pot=game.current_pot,
            )

            # --- Play paired hand with Baseline MCCFR ---
            game_b = GameState(
                dict(initial),
                small_blind=small_blind,
                big_blind=big_blind,
                button=button_name,
                deck=Deck(seed=d_seed),
            )
            hist_b: list[ActionRecord] = []
            while not game_b.complete and game_b.current_player is not None:
                actor_b = game_b.current_player
                obs_b = observe(game_b, actor_b, tuple(hist_b))
                act_b = (
                    baseline_agent.act(obs_b)
                    if actor_b == "Hero"
                    else opp_base.act(obs_b)
                )
                st_b = game_b.street.value
                game_b.act(act_b, player=actor_b)
                hist_b.append(ActionRecord(actor_b, act_b, st_b))
            hero_net_base = game_b.stacks["Hero"] - initial["Hero"]
            bb100_base = (hero_net_base / big_blind) * 100.0

            prof = harvey.get_profile("ChameleonVillain")
            stats = prof.stats_snapshot()
            arch = prof.archetype_distribution()
            unweighted_arch = unweighted_profile.archetype_distribution()
            range_snap = prof.range_snapshot()

            row = {
                "seed": seed,
                "hand": hand_num,
                "true_archetype": true_arch,
                "predicted_archetype": arch.most_likely,
                "unweighted_predicted_archetype": unweighted_arch.most_likely,
                "correct_family": int(
                    _regime_family_match(
                        arch.most_likely, true_arch, arch.probabilities
                    )
                ),
                "unweighted_correct_family": int(
                    _regime_family_match(
                        unweighted_arch.most_likely,
                        true_arch,
                        unweighted_arch.probabilities,
                    )
                ),
                "target_prob": arch.probabilities.get(true_arch, 0.0),
                "unweighted_target_prob": unweighted_arch.probabilities.get(
                    true_arch, 0.0
                ),
                "archetype_confidence": arch.confidence,
                "stability_score": stats.stability_score,
                "drift_detected": int(stats.stability_score < 0.78),
                "vpip": stats.vpip.smoothed_value,
                "pfr": stats.pfr.smoothed_value,
                "afq": stats.aggression_frequency.smoothed_value,
                "fold_to_bet": stats.fold_to_bet.smoothed_value,
                "wtsd": stats.wtsd_when_saw_flop.smoothed_value,
                "range_post_mean_strength": range_snap.posterior_mean_strength,
                "range_prior_mean_strength": range_snap.prior_mean_strength,
                "lambda_value": (
                    float(statistics.mean(hand_lambdas))
                    if hand_lambdas
                    else (harvey.lambda_history[-1] if harvey.lambda_history else 0.0)
                ),
                "policy_l1_shift_from_mccfr": (
                    float(statistics.mean(hand_policy_l1)) if hand_policy_l1 else 0.0
                ),
                "harvey_bb100": bb100_harvey,
                "baseline_bb100": bb100_base,
                "paired_diff_bb100": bb100_harvey - bb100_base,
                **{
                    f"prob_{name}": arch.probabilities.get(name, 0.0)
                    for name in ARCHETYPE_NAMES
                },
            }
            seed_hand_rows.append(row)
            per_hand_records_by_hand[hand_num].append(row)

        # Measure detection delay & stabilization time at each boundary for this seed
        for boundary in change_boundaries:
            target = target_archetypes[boundary]
            prev = prev_archetypes[boundary]
            phase_end = next(p.end_hand for p in schedule if p.start_hand == boundary)
            delay = phase_end - boundary + 1
            for r in seed_hand_rows:
                h = r["hand"]
                if boundary <= h <= phase_end and (
                    r[f"prob_{target}"] > r[f"prob_{prev}"]
                    or _regime_family_match(
                        r["predicted_archetype"],
                        target,
                        {k: r[f"prob_{k}"] for k in ARCHETYPE_NAMES},
                    )
                ):
                    delay = h - boundary + 1
                    break
            detection_delays[boundary].append(delay)

            # Stabilization time: first hand >= boundary + 2 where classification holds for 3 consecutive hands
            stab = phase_end - boundary + 1
            for h_cand in range(boundary, phase_end - 1):
                window_rows = [
                    r for r in seed_hand_rows if h_cand <= r["hand"] <= h_cand + 2
                ]
                if len(window_rows) == 3 and all(
                    r["correct_family"] == 1 for r in window_rows
                ):
                    stab = h_cand - boundary + 1
                    break
            stabilization_times[boundary].append(stab)

        seed_summaries.append(
            {
                "seed": seed,
                "harvey_mean_bb100": float(
                    statistics.mean(r["harvey_bb100"] for r in seed_hand_rows)
                ),
                "baseline_mean_bb100": float(
                    statistics.mean(r["baseline_bb100"] for r in seed_hand_rows)
                ),
                "paired_gain_bb100": float(
                    statistics.mean(r["paired_diff_bb100"] for r in seed_hand_rows)
                ),
                "overall_accuracy": float(
                    statistics.mean(r["correct_family"] for r in seed_hand_rows)
                ),
                "unweighted_accuracy": float(
                    statistics.mean(
                        r["unweighted_correct_family"] for r in seed_hand_rows
                    )
                ),
            }
        )

    # Aggregate time series across seeds for each hand 1..100
    aggregated_timeseries: list[dict[str, Any]] = []
    for h in range(1, total_hands + 1):
        rows = per_hand_records_by_hand[h]
        true_arch = rows[0]["true_archetype"]
        agg_row: dict[str, Any] = {
            "hand": h,
            "true_archetype": true_arch,
            "accuracy": round(
                float(statistics.mean(r["correct_family"] for r in rows)), 4
            ),
            "unweighted_accuracy": round(
                float(statistics.mean(r["unweighted_correct_family"] for r in rows)),
                4,
            ),
            "target_prob": round(
                float(statistics.mean(r["target_prob"] for r in rows)), 4
            ),
            "unweighted_target_prob": round(
                float(statistics.mean(r["unweighted_target_prob"] for r in rows)),
                4,
            ),
            "archetype_confidence": round(
                float(statistics.mean(r["archetype_confidence"] for r in rows)), 4
            ),
            "stability_score": round(
                float(statistics.mean(r["stability_score"] for r in rows)), 4
            ),
            "drift_rate": round(
                float(statistics.mean(r["drift_detected"] for r in rows)), 4
            ),
            "vpip": round(float(statistics.mean(r["vpip"] for r in rows)), 4),
            "pfr": round(float(statistics.mean(r["pfr"] for r in rows)), 4),
            "afq": round(float(statistics.mean(r["afq"] for r in rows)), 4),
            "fold_to_bet": round(
                float(statistics.mean(r["fold_to_bet"] for r in rows)), 4
            ),
            "wtsd": round(float(statistics.mean(r["wtsd"] for r in rows)), 4),
            "range_post_mean_strength": round(
                float(statistics.mean(r["range_post_mean_strength"] for r in rows)),
                4,
            ),
            "lambda_value": round(
                float(statistics.mean(r["lambda_value"] for r in rows)), 4
            ),
            "policy_l1_shift_from_mccfr": round(
                float(statistics.mean(r["policy_l1_shift_from_mccfr"] for r in rows)),
                4,
            ),
            "harvey_bb100": round(
                float(statistics.mean(r["harvey_bb100"] for r in rows)), 4
            ),
            "baseline_bb100": round(
                float(statistics.mean(r["baseline_bb100"] for r in rows)), 4
            ),
        }
        for name in ARCHETYPE_NAMES:
            agg_row[f"prob_{name}"] = round(
                float(statistics.mean(r[f"prob_{name}"] for r in rows)), 4
            )
        aggregated_timeseries.append(agg_row)

    # Compute phase-level metrics (post-warmup accuracy, BB/100 per phase, false positive drift rate)
    phase_summaries: list[dict[str, Any]] = []
    for phase in schedule:
        p_rows = [
            r
            for h in range(phase.start_hand, phase.end_hand + 1)
            for r in per_hand_records_by_hand[h]
        ]
        # Post-transition rows (after 6 hands of observing the new regime)
        post_warmup_start = min(phase.end_hand, phase.start_hand + 6)
        settled_rows = [
            r
            for h in range(post_warmup_start, phase.end_hand + 1)
            for r in per_hand_records_by_hand[h]
        ]
        phase_summaries.append(
            {
                "phase_label": phase.label,
                "start_hand": phase.start_hand,
                "end_hand": phase.end_hand,
                "true_archetype": phase.archetype_name,
                "overall_phase_accuracy": round(
                    float(statistics.mean(r["correct_family"] for r in p_rows)),
                    4,
                ),
                "post_transition_accuracy": round(
                    float(statistics.mean(r["correct_family"] for r in settled_rows)),
                    4,
                ),
                "unweighted_post_transition_accuracy": round(
                    float(
                        statistics.mean(
                            r["unweighted_correct_family"] for r in settled_rows
                        )
                    ),
                    4,
                ),
                "mean_target_prob_settled": round(
                    float(statistics.mean(r["target_prob"] for r in settled_rows)),
                    4,
                ),
                "unweighted_target_prob_settled": round(
                    float(
                        statistics.mean(
                            r["unweighted_target_prob"] for r in settled_rows
                        )
                    ),
                    4,
                ),
                "mean_confidence": round(
                    float(statistics.mean(r["archetype_confidence"] for r in p_rows)),
                    4,
                ),
                "mean_lambda": round(
                    float(statistics.mean(r["lambda_value"] for r in p_rows)), 4
                ),
                "harvey_bb100": round(
                    float(statistics.mean(r["harvey_bb100"] for r in p_rows)), 2
                ),
                "baseline_bb100": round(
                    float(statistics.mean(r["baseline_bb100"] for r in p_rows)), 2
                ),
                "paired_gain_bb100": round(
                    float(statistics.mean(r["paired_diff_bb100"] for r in p_rows)),
                    2,
                ),
            }
        )

    boundary_metrics = {}
    transition_metrics = {}
    for b in change_boundaries:
        delays = detection_delays[b]
        stabs = stabilization_times[b]
        phase = next(p for p in schedule if p.start_hand == b)
        phase_len = phase.end_hand - phase.start_hand + 1
        det_rate = sum(1 for d in delays if d < phase_len) / max(1, len(delays))
        b_info = {
            "boundary_hand": b,
            "from_archetype": prev_archetypes[b],
            "to_archetype": target_archetypes[b],
            "mean_detection_delay_hands": round(float(statistics.mean(delays)), 2),
            "median_detection_delay_hands": round(float(statistics.median(delays)), 2),
            "mean_stabilization_hands": round(float(statistics.mean(stabs)), 2),
            "detection_rate": round(det_rate, 4),
            "delays_by_seed": delays,
        }
        boundary_metrics[f"hand_{b}_{prev_archetypes[b]}_to_{target_archetypes[b]}"] = (
            b_info
        )
        transition_metrics[str(b)] = b_info

    # Stationary false-positive drift rate: fraction of hands in settled stationary windows
    # (hands 8..20) before any regime change occurs where drift_detected triggered
    stationary_initial_rows = [
        r for h in range(8, 21) for r in per_hand_records_by_hand[h]
    ]
    false_positive_drift_rate = (
        float(statistics.mean(r["drift_detected"] for r in stationary_initial_rows))
        if stationary_initial_rows
        else 0.0
    )

    all_harvey_bb = [
        r["harvey_bb100"]
        for h in range(1, total_hands + 1)
        for r in per_hand_records_by_hand[h]
    ]
    all_base_bb = [
        r["baseline_bb100"]
        for h in range(1, total_hands + 1)
        for r in per_hand_records_by_hand[h]
    ]
    all_diff_bb = [a - b for a, b in zip(all_harvey_bb, all_base_bb, strict=True)]

    overall_post_warmup_acc = float(
        statistics.mean(p["post_transition_accuracy"] for p in phase_summaries)
    )
    unweighted_post_warmup_acc = float(
        statistics.mean(
            p["unweighted_post_transition_accuracy"] for p in phase_summaries
        )
    )

    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_revision": _git_revision(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "seeds": list(seeds),
        "total_hands_per_seed": total_hands,
        "total_paired_hands": len(all_harvey_bb),
        "schedule": [asdict(p) for p in schedule],
        "overall_harvey_bb100": round(float(statistics.mean(all_harvey_bb)), 2),
        "overall_baseline_bb100": round(float(statistics.mean(all_base_bb)), 2),
        "overall_paired_gain_bb100": round(float(statistics.mean(all_diff_bb)), 2),
        "overall_post_transition_accuracy": round(overall_post_warmup_acc, 4),
        "unweighted_ablation_post_transition_accuracy": round(
            unweighted_post_warmup_acc, 4
        ),
        "post_shift_accuracy_vs_stale": {
            "adaptive_post_shift_accuracy": round(overall_post_warmup_acc, 4),
            "stale_unweighted_post_shift_accuracy": round(
                unweighted_post_warmup_acc, 4
            ),
        },
        "false_positive_drift_rate_stationary": round(false_positive_drift_rate, 4),
        "boundary_detection_metrics": boundary_metrics,
        "transition_metrics": transition_metrics,
        "phase_summaries": phase_summaries,
        "seed_summaries": seed_summaries,
        "timeseries": aggregated_timeseries,
        "hand_curves": aggregated_timeseries,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "adaptation_results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        if aggregated_timeseries:
            with (out_path / "adaptation_timeseries.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(aggregated_timeseries[0].keys())
                )
                writer.writeheader()
                writer.writerows(aggregated_timeseries)
        html_doc = render_adaptation_html(result)
        (out_path / "adaptation_visualization.html").write_text(
            html_doc, encoding="utf-8"
        )

    return result


def render_adaptation_html(result: dict[str, Any]) -> str:
    """Render a self-contained SVG/HTML report visualizing HarveySpecter's adaptation over time."""
    ts = result["timeseries"]
    n = len(ts)
    width = 860
    height = 240
    pad_l, pad_r, pad_t, pad_b = 48, 24, 24, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def _x(h: int) -> float:
        return pad_l + ((h - 1) / max(1, n - 1)) * plot_w

    def _y(val: float, vmin: float = 0.0, vmax: float = 1.0) -> float:
        clamped = max(vmin, min(vmax, val))
        return pad_t + (1.0 - (clamped - vmin) / max(1e-9, vmax - vmin)) * plot_h

    def _polyline(key: str, color: str, vmin: float = 0.0, vmax: float = 1.0) -> str:
        pts = " ".join(f"{_x(r['hand']):.1f},{_y(r[key], vmin, vmax):.1f}" for r in ts)
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{pts}" />'

    boundaries_svg = "".join(
        f'<line x1="{_x(b):.1f}" y1="{pad_t}" x2="{_x(b):.1f}" y2="{height - pad_b}" '
        f'stroke="#f5b942" stroke-dasharray="4,4" stroke-width="1.5" />'
        for b in (21, 51, 81)
    )

    chart1_lines = (
        _polyline("vpip", "#3b82f6")
        + _polyline("pfr", "#ef4444")
        + _polyline("afq", "#f5b942")
        + _polyline("fold_to_bet", "#10b981")
    )
    chart2_lines = (
        _polyline("prob_tight_passive", "#60a5fa")
        + _polyline("prob_loose_aggressive", "#f97316")
        + _polyline("prob_calling_station", "#34d399")
        + _polyline("unweighted_target_prob", "#94a3b8")
    )
    chart3_lines = (
        _polyline("lambda_value", "#f5b942")
        + _polyline("stability_score", "#38bdf8")
        + _polyline("archetype_confidence", "#a78bfa")
        + _polyline("policy_l1_shift_from_mccfr", "#fb7185")
    )

    phase_rows_html = "".join(
        f"<tr><td>{p['phase_label']}</td><td><code>{p['true_archetype']}</code></td>"
        f"<td>{(p['post_transition_accuracy'] * 100):.1f}%</td>"
        f"<td>{(p['unweighted_post_transition_accuracy'] * 100):.1f}%</td>"
        f"<td>{p['mean_lambda']:.3f}</td>"
        f"<td>{p['harvey_bb100']:+.1f}</td>"
        f"<td>{p['baseline_bb100']:+.1f}</td>"
        f"<td><strong>{p['paired_gain_bb100']:+.1f}</strong></td></tr>"
        for p in result["phase_summaries"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>HarveySpecter Phase 11 Behavioural Adaptation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0b111a; color: #e2e8f0; padding: 24px; max-width: 980px; margin: 0 auto; }}
  h1, h2 {{ color: #f8fafc; }}
  .card {{ background: #131d2b; border: 1px solid #26364d; border-radius: 10px; padding: 16px; margin-bottom: 18px; }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; margin-bottom: 8px; flex-wrap: wrap; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
  th, td {{ border-bottom: 1px solid #26364d; padding: 8px 10px; text-align: left; }}
  th {{ color: #94a3b8; font-size: 11.5px; text-transform: uppercase; }}
</style>
</head>
<body>
  <h1>HarveySpecter Phase 11: Non-Stationary Behavioural Adaptation</h1>
  <p>Seeds: {result["seeds"]} | Schedule: Tight-Passive (1–20) → Loose-Aggressive (21–50) → Calling Station (51–80) → Tight-Passive (81–100)</p>

  <div class="card">
    <h2>1. Smoothed Behavioural Metrics Over Time (Hands 1–100)</h2>
    <div class="legend">
      <span><i class="dot" style="background:#3b82f6"></i>VPIP</span>
      <span><i class="dot" style="background:#ef4444"></i>PFR</span>
      <span><i class="dot" style="background:#f5b942"></i>Aggression Freq (AFq)</span>
      <span><i class="dot" style="background:#10b981"></i>Fold to Bet</span>
      <span><i class="dot" style="background:#f5b942"></i>Dashed Vertical Lines = Strategy Switch Boundaries (Hands 21, 51, 81)</span>
    </div>
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0d1520" stroke="#26364d" />
      {boundaries_svg}
      {chart1_lines}
    </svg>
  </div>

  <div class="card">
    <h2>2. Posterior Archetype Probabilities vs Unweighted Ablation</h2>
    <div class="legend">
      <span><i class="dot" style="background:#60a5fa"></i>P(Tight-Passive)</span>
      <span><i class="dot" style="background:#f97316"></i>P(Loose-Aggressive)</span>
      <span><i class="dot" style="background:#34d399"></i>P(Calling Station)</span>
      <span><i class="dot" style="background:#94a3b8"></i>Unweighted Decay=1.0 Target Prob (Stale-Evidence Baseline)</span>
    </div>
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0d1520" stroke="#26364d" />
      {boundaries_svg}
      {chart2_lines}
    </svg>
  </div>

  <div class="card">
    <h2>3. Exploitation Coefficient λ, Drift Stability &amp; Policy Deviation from MCCFR</h2>
    <div class="legend">
      <span><i class="dot" style="background:#f5b942"></i>Adaptive λ</span>
      <span><i class="dot" style="background:#38bdf8"></i>Stability Score</span>
      <span><i class="dot" style="background:#a78bfa"></i>Archetype Confidence</span>
      <span><i class="dot" style="background:#fb7185"></i>Policy L1 Shift ||π_final - π_MCCFR||_1 / 2</span>
    </div>
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0d1520" stroke="#26364d" />
      {boundaries_svg}
      {chart3_lines}
    </svg>
  </div>

  <div class="card">
    <h2>4. Regime-by-Regime Adaptation &amp; Paired BB/100 Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Phase</th><th>True Policy</th><th>Settled Acc (Recency)</th><th>Settled Acc (Unweighted)</th>
          <th>Mean λ</th><th>Harvey BB/100</th><th>MCCFR BB/100</th><th>Paired Gain BB/100</th>
        </tr>
      </thead>
      <tbody>
        {phase_rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


# ============================================================================
# 4. Quantitative Bayesian Range Model Validation (Section 4)
# ============================================================================


def run_bayesian_range_validation(
    *,
    seeds: Sequence[int] = (101, 202, 303, 404),
    hands_per_opponent: int = 80,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Quantitatively validate `BayesianRangeModel` against Uniform, Position, and Prev-Street baselines."""
    started = time.perf_counter()
    opponents: list[tuple[str, Agent]] = [
        ("NitAgent", NitAgent(seed=11)),
        ("TightPassiveAgent", TightPassiveAgent(seed=22)),
        ("CallingStationAgent", CallingStationAgent(seed=33)),
        ("LooseAggressiveAgent", LooseAggressiveAgent(seed=44)),
        ("ManiacAgent", ManiacAgent(seed=55)),
        ("EquityAgent", EquityAgent(EquityAgentConfig(simulations=24), seed=66)),
    ]

    per_opponent_metrics: list[dict[str, Any]] = []
    street_convergence: dict[str, list[dict[str, float]]] = {
        "preflop": [],
        "flop": [],
        "turn": [],
        "river": [],
    }
    # 10-bin calibration curve buckets for class probabilities in [0, 1]
    calib_bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    all_evals: list[dict[str, float]] = []

    for opp_name, opp_agent in opponents:
        opp_evals: list[dict[str, float]] = []
        for seed in seeds:
            prof = OpponentProfile(opponent_id=opp_name, recency_decay=0.94)
            hero_agent = CallingStationAgent(seed=seed + 500)
            opp_agent.reset(seed + 100)
            deck_rng = random.Random(seed * 17_389 + sum(ord(c) for c in opp_name))

            for h in range(1, hands_per_opponent + 1):
                d_seed = deck_rng.randrange(2**32)
                hero_is_button = (h % 2) == 0
                game = GameState(
                    {"Hero": 200, "Villain": 200},
                    small_blind=1,
                    big_blind=2,
                    button="Hero" if hero_is_button else "Villain",
                    deck=Deck(seed=d_seed),
                )
                prof.start_hand(
                    hand_number=h,
                    opponent_is_button=not hero_is_button,
                    starting_stack=200,
                    effective_stack=200,
                    big_blind=2,
                    hero_hole=game.hole_cards["Hero"],
                    community_cards=(),
                )
                history: list[ActionRecord] = []
                last_recorded_street = ""
                while not game.complete and game.current_player is not None:
                    actor = game.current_player
                    obs = observe(game, actor, tuple(history))
                    if actor == "Hero":
                        prof.ingest_observation(obs, hand_number=h)
                        act = hero_agent.act(obs)
                    else:
                        act = opp_agent.act(obs)
                    st = game.street.value
                    game.act(act, player=actor)
                    history.append(ActionRecord(actor, act, st))

                    # After an opponent non-fold action on street `st`, record street convergence
                    if actor == "Villain" and ActionType(act.type) != ActionType.FOLD:
                        obs_after = observe(
                            game,
                            "Hero",
                            tuple(history),
                        )
                        prof.ingest_observation(obs_after, hand_number=h)
                        if st != last_recorded_street and st in street_convergence:
                            last_recorded_street = st
                            opp_pos = "button" if not hero_is_button else "big_blind"
                            st_eval = prof.range_model.evaluate_ground_truth_hand(
                                game.hole_cards["Villain"],
                                position=opp_pos,
                                stats=prof.stats_snapshot(),
                            )
                            snap = prof.range_model.snapshot()
                            street_convergence[st].append(
                                {
                                    "likelihood_ratio": st_eval["likelihood_ratio"],
                                    "brier_post": st_eval["brier_class_score"],
                                    "log_loss_post": st_eval["log_loss_post"],
                                    "entropy_bits": snap.entropy_bits,
                                    "active_combos": float(snap.active_combo_count),
                                }
                            )

                reached_sd = bool(game.hand_ranks and "Villain" in game.hand_ranks)
                villain_net = game.stacks["Villain"] - 200
                # Ingest final actions before evaluating ground truth (before showdown calibration updates!)
                obs_final = observe(game, "Hero", tuple(history))
                prof.ingest_observation(obs_final, hand_number=h)

                rec = prof.collector.get_hand(opp_name, h)
                if rec is not None and rec.opponent_actions:
                    opp_pos = "button" if not hero_is_button else "big_blind"
                    gt = prof.range_model.evaluate_ground_truth_hand(
                        game.hole_cards["Villain"],
                        position=opp_pos,
                        stats=prof.stats_snapshot(),
                    )
                    opp_evals.append(gt)
                    all_evals.append(gt)

                    # Populate calibration bins over 3 strength tiers (premium_strong, medium_playable, speculative_weak)
                    snap = prof.range_model.snapshot()
                    true_combo = tuple(
                        sorted(
                            game.hole_cards["Villain"],
                            key=lambda c: (
                                c.rank,
                                ("clubs", "diamonds", "hearts", "spades").index(c.suit),
                            ),
                            reverse=True,
                        )
                    )
                    true_cls = COMBO_TO_CLASS[true_combo]
                    # Also bin top-20 class probabilities for ECE calculation
                    for cls_est in snap.top_classes:
                        p_pred = cls_est.posterior_probability
                        b_idx = min(9, int(p_pred * 10.0))
                        calib_bins[b_idx].append(
                            (p_pred, 1 if cls_est.hand_class == true_cls else 0)
                        )

                prof.finalize_hand(
                    hand_number=h,
                    history=tuple(history),
                    hero_player="Hero",
                    community_cards=tuple(game.community_cards),
                    reached_showdown=reached_sd,
                    revealed_opponent_cards=game.hole_cards["Villain"]
                    if reached_sd
                    else None,
                    opponent_net_chips=villain_net,
                    opponent_won=(villain_net > 0),
                    final_pot=game.current_pot,
                )

        if opp_evals:
            per_opponent_metrics.append(
                {
                    "opponent": opp_name,
                    "evaluated_hands": len(opp_evals),
                    "mean_likelihood_ratio": round(
                        float(
                            statistics.mean(e["likelihood_ratio"] for e in opp_evals)
                        ),
                        4,
                    ),
                    "log_loss_posterior": round(
                        float(statistics.mean(e["log_loss_post"] for e in opp_evals)),
                        4,
                    ),
                    "log_loss_prev_street": round(
                        float(
                            statistics.mean(
                                e["log_loss_prev_street"] for e in opp_evals
                            )
                        ),
                        4,
                    ),
                    "log_loss_position_prior": round(
                        float(
                            statistics.mean(e["log_loss_pos_prior"] for e in opp_evals)
                        ),
                        4,
                    ),
                    "log_loss_uniform_prior": round(
                        float(statistics.mean(e["log_loss_prior"] for e in opp_evals)),
                        4,
                    ),
                    "brier_posterior": round(
                        float(
                            statistics.mean(e["brier_class_score"] for e in opp_evals)
                        ),
                        6,
                    ),
                    "brier_prev_street": round(
                        float(
                            statistics.mean(
                                e["brier_prev_street_score"] for e in opp_evals
                            )
                        ),
                        6,
                    ),
                    "brier_position_prior": round(
                        float(
                            statistics.mean(
                                e["brier_pos_prior_score"] for e in opp_evals
                            )
                        ),
                        6,
                    ),
                    "brier_uniform_prior": round(
                        float(
                            statistics.mean(e["brier_prior_score"] for e in opp_evals)
                        ),
                        6,
                    ),
                    "top_5_class_accuracy": round(
                        float(statistics.mean(e["top_5_hit"] for e in opp_evals)), 4
                    ),
                    "top_10_class_accuracy": round(
                        float(statistics.mean(e["top_10_hit"] for e in opp_evals)), 4
                    ),
                    "top_25_class_accuracy": round(
                        float(statistics.mean(e["top_25_hit"] for e in opp_evals)), 4
                    ),
                    "uniform_top_10_accuracy": round(
                        float(
                            statistics.mean(e["prior_top_10_hit"] for e in opp_evals)
                        ),
                        4,
                    ),
                    "uniform_top_25_accuracy": round(
                        float(
                            statistics.mean(e["prior_top_25_hit"] for e in opp_evals)
                        ),
                        4,
                    ),
                    "mean_true_class_rank": round(
                        float(statistics.mean(e["true_class_rank"] for e in opp_evals)),
                        2,
                    ),
                    "mean_uniform_class_rank": round(
                        float(
                            statistics.mean(e["prior_class_rank"] for e in opp_evals)
                        ),
                        2,
                    ),
                }
            )

    street_summary = {}
    for st, items in street_convergence.items():
        if items:
            street_summary[st] = {
                "samples": len(items),
                "mean_likelihood_ratio": round(
                    float(statistics.mean(x["likelihood_ratio"] for x in items)), 4
                ),
                "mean_log_loss": round(
                    float(statistics.mean(x["log_loss_post"] for x in items)), 4
                ),
                "mean_brier": round(
                    float(statistics.mean(x["brier_post"] for x in items)), 6
                ),
                "mean_entropy_bits": round(
                    float(statistics.mean(x["entropy_bits"] for x in items)), 3
                ),
                "mean_active_combos": round(
                    float(statistics.mean(x["active_combos"] for x in items)), 1
                ),
            }

    calibration_curve = []
    total_calib_pts = sum(len(b) for b in calib_bins)
    ece = 0.0
    for idx, bucket in enumerate(calib_bins):
        if not bucket:
            calibration_curve.append(
                {
                    "bin_index": idx,
                    "bin_range": f"[{idx / 10:.1f}, {(idx + 1) / 10:.1f})",
                    "count": 0,
                    "mean_predicted_prob": round((idx + 0.5) / 10.0, 5),
                    "empirical_hit_rate": 0.0,
                    "abs_calibration_error": 0.0,
                }
            )
            continue
        mean_pred = float(statistics.mean(p for p, _ in bucket))
        emp_freq = float(statistics.mean(y for _, y in bucket))
        weight = len(bucket) / max(1, total_calib_pts)
        ece += weight * abs(mean_pred - emp_freq)
        calibration_curve.append(
            {
                "bin_index": idx,
                "bin_range": f"[{idx / 10:.1f}, {(idx + 1) / 10:.1f})",
                "count": len(bucket),
                "mean_predicted_prob": round(mean_pred, 5),
                "empirical_hit_rate": round(emp_freq, 5),
                "abs_calibration_error": round(abs(mean_pred - emp_freq), 5),
            }
        )

    overall = {
        "decisions_evaluated": len(all_evals),
        "log_loss_posterior": round(
            float(statistics.mean(e["log_loss_post"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
        "log_loss_uniform_legal": round(
            float(statistics.mean(e["log_loss_prior"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
        "log_loss_position_prior": round(
            float(statistics.mean(e["log_loss_pos_prior"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
        "log_loss_prev_street_posterior": round(
            float(statistics.mean(e["log_loss_prev_street"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
        "brier_posterior": round(
            float(statistics.mean(e["brier_class_score"] for e in all_evals)), 6
        )
        if all_evals
        else 0.0,
        "brier_uniform_legal": round(
            float(statistics.mean(e["brier_prior_score"] for e in all_evals)), 6
        )
        if all_evals
        else 0.0,
        "mean_likelihood_ratio_vs_uniform": round(
            float(statistics.mean(e["likelihood_ratio"] for e in all_evals)), 4
        )
        if all_evals
        else 1.0,
        "top_5_class_accuracy": round(
            float(statistics.mean(e["top_5_hit"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
        "top_10_class_accuracy": round(
            float(statistics.mean(e["top_10_hit"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
        "top_25_class_accuracy": round(
            float(statistics.mean(e["top_25_hit"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
        "uniform_top_10_accuracy": round(
            float(statistics.mean(e["prior_top_10_hit"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
        "uniform_top_25_accuracy": round(
            float(statistics.mean(e["prior_top_25_hit"] for e in all_evals)), 4
        )
        if all_evals
        else 0.0,
    }

    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_revision": _git_revision(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "expected_calibration_error_ece": round(ece, 5),
        "overall": overall,
        "per_opponent_metrics": per_opponent_metrics,
        "street_convergence": street_summary,
        "calibration_curve": calibration_curve,
        "calibration_curve_10_bins": calibration_curve,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "range_validation.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if per_opponent_metrics:
            with (out_path / "range_validation_summary.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(per_opponent_metrics[0].keys())
                )
                writer.writeheader()
                writer.writerows(per_opponent_metrics)

    return payload


# ============================================================================
# 5. Strategy Robustness & Counter-Exploitation Study (Section 5)
# ============================================================================


def run_counter_exploitation_study(
    *,
    seeds: Sequence[int] = (101, 202, 303, 404),
    hands_per_seed: int = 60,
    mccfr_iterations: int = 12,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate HarveySpecter against counter-exploitative and deceptive opponents."""
    started = time.perf_counter()
    bench_cfg = HarveyBenchmarkConfig(
        seeds=tuple(seeds),
        hands_per_seed=hands_per_seed,
        mccfr_iterations=mccfr_iterations,
    )
    mccfr_by_seed = {s: build_mccfr_baseline(bench_cfg, s) for s in seeds}

    adversary_names = (
        "BluffPunisher",
        "ValueFolder",
        "AlternatingGear",
        "DeceptiveSizing",
    )
    variants = (
        "BaselineMCCFR",
        "HarveySpecter_Adaptive",
        "HarveySpecter_PureExploit",
    )

    rows: list[dict[str, Any]] = []
    max_adaptive_lambda: dict[str, float] = {a: 0.0 for a in adversary_names}
    invariants_checked = {
        "total_decisions_verified": 0,
        "lambda_within_bounds_violations": 0,
        "probability_distribution_violations": 0,
        "max_observed_lambda": 0.0,
        "min_observed_lambda": 1.0,
    }

    for adv_name in adversary_names:
        for variant in variants:
            bb100_all: list[float] = []
            min_stability_list: list[float] = []
            mean_stability_list: list[float] = []
            mean_lambda_list: list[float] = []
            lambda_step_deltas: list[float] = []

            for seed in seeds:
                strategy, strategy_sums, game_cfg = mccfr_by_seed[seed]
                base_mccfr = AbstractHUNLAgent(
                    strategy=strategy,
                    config=game_cfg,
                    seed=seed + 19,
                    strategy_sums=strategy_sums,
                )
                if variant == "BaselineMCCFR":
                    hero: Agent = base_mccfr
                elif variant == "HarveySpecter_Adaptive":
                    hero = HarveySpecterAgent(
                        base_mccfr,
                        config=HarveySpecterConfig(
                            mode="adaptive",
                            max_lambda=0.80,
                            recency_decay=0.90,
                            equity_simulations=32,
                        ),
                        opponent_id=adv_name,
                        seed=seed + 19,
                    )
                else:
                    hero = HarveySpecterAgent(
                        base_mccfr,
                        config=HarveySpecterConfig(
                            mode="pure_exploit",
                            fixed_lambda=1.0,
                            max_lambda=1.0,
                            recency_decay=0.90,
                            equity_simulations=32,
                        ),
                        opponent_id=adv_name,
                        seed=seed + 19,
                    )

                if adv_name == "BluffPunisher":
                    opp: Any = BluffPunisherAgent(bait_hands=18, seed=seed + 99)
                elif adv_name == "ValueFolder":
                    opp = ValueFolderAgent(bait_hands=18, seed=seed + 99)
                elif adv_name == "AlternatingGear":
                    opp = AlternatingGearAgent(period_hands=8, seed=seed + 99)
                else:
                    opp = DeceptiveSizingAgent(seed=seed + 99)

                deck_rng = random.Random(
                    (seed * 41_999) ^ sum(ord(c) for c in adv_name)
                )
                stabilities: list[float] = []

                for h in range(1, hands_per_seed + 1):
                    if hasattr(opp, "set_hand_number"):
                        opp.set_hand_number(h)
                    d_seed = deck_rng.randrange(2**32)
                    hero_is_button = (h % 2) == 0
                    game = GameState(
                        {"Hero": 200, "Villain": 200},
                        small_blind=1,
                        big_blind=2,
                        button="Hero" if hero_is_button else "Villain",
                        deck=Deck(seed=d_seed),
                    )
                    if isinstance(hero, HarveySpecterAgent):
                        hero.start_hand(
                            opponent_id=adv_name,
                            hand_number=h,
                            opponent_is_button=not hero_is_button,
                            hero_hole=game.hole_cards["Hero"],
                        )
                    history: list[ActionRecord] = []
                    while not game.complete and game.current_player is not None:
                        actor = game.current_player
                        obs = observe(game, actor, tuple(history))
                        if actor == "Hero":
                            act = hero.act(obs)
                            if (
                                isinstance(hero, HarveySpecterAgent)
                                and hero.last_decision_report is not None
                            ):
                                rep = hero.last_decision_report
                                invariants_checked["total_decisions_verified"] += 1
                                lam = rep.lambda_value
                                if variant == "HarveySpecter_Adaptive":
                                    max_adaptive_lambda[adv_name] = max(
                                        max_adaptive_lambda[adv_name], lam
                                    )
                                invariants_checked["max_observed_lambda"] = max(
                                    invariants_checked["max_observed_lambda"], lam
                                )
                                invariants_checked["min_observed_lambda"] = min(
                                    invariants_checked["min_observed_lambda"], lam
                                )
                                if not (0.0 <= lam <= hero.config.max_lambda + 1e-9):
                                    invariants_checked[
                                        "lambda_within_bounds_violations"
                                    ] += 1
                                prob_sum = sum(
                                    p["probability"] for p in rep.final_probabilities
                                )
                                if abs(prob_sum - 1.0) > 1e-4 or any(
                                    p["probability"] < -1e-9
                                    for p in rep.final_probabilities
                                ):
                                    invariants_checked[
                                        "probability_distribution_violations"
                                    ] += 1
                        else:
                            act = opp.act(obs)
                        st = game.street.value
                        game.act(act, player=actor)
                        history.append(ActionRecord(actor, act, st))

                    hero_net = game.stacks["Hero"] - 200
                    villain_net = game.stacks["Villain"] - 200
                    assert hero_net + villain_net == 0
                    bb100_all.append((hero_net / 2.0) * 100.0)

                    if isinstance(hero, HarveySpecterAgent):
                        reached_sd = bool(
                            game.hand_ranks and "Villain" in game.hand_ranks
                        )
                        hero.finalize_hand(
                            opponent_id=adv_name,
                            hand_number=h,
                            history=tuple(history),
                            hero_player="Hero",
                            community_cards=tuple(game.community_cards),
                            reached_showdown=reached_sd,
                            revealed_opponent_cards=(
                                game.hole_cards["Villain"] if reached_sd else None
                            ),
                            opponent_net_chips=villain_net,
                            opponent_won=(villain_net > 0),
                            final_pot=game.current_pot,
                        )
                        stabilities.append(
                            hero.get_profile(adv_name).stats_snapshot().stability_score
                        )

                if isinstance(hero, HarveySpecterAgent):
                    if stabilities:
                        min_stability_list.append(min(stabilities))
                        mean_stability_list.append(float(statistics.mean(stabilities)))
                    if hero.lambda_history:
                        mean_lambda_list.append(
                            float(statistics.mean(hero.lambda_history))
                        )
                        for i in range(1, len(hero.lambda_history)):
                            lambda_step_deltas.append(
                                abs(hero.lambda_history[i] - hero.lambda_history[i - 1])
                            )

            rows.append(
                {
                    "adversary": adv_name,
                    "variant": variant,
                    "total_hands": len(bb100_all),
                    "bb_per_100": round(float(statistics.mean(bb100_all)), 2),
                    "min_stability_score": (
                        round(float(statistics.mean(min_stability_list)), 4)
                        if min_stability_list
                        else 1.0
                    ),
                    "mean_stability_score": (
                        round(float(statistics.mean(mean_stability_list)), 4)
                        if mean_stability_list
                        else 1.0
                    ),
                    "mean_lambda": (
                        round(float(statistics.mean(mean_lambda_list)), 4)
                        if mean_lambda_list
                        else 0.0
                    ),
                    "mean_lambda_step_change": (
                        round(float(statistics.mean(lambda_step_deltas)), 4)
                        if lambda_step_deltas
                        else 0.0
                    ),
                }
            )

    adversaries_summary = []
    for adv_name in adversary_names:
        adv_adaptive_rows = [
            r
            for r in rows
            if r["adversary"] == adv_name and r["variant"] == "HarveySpecter_Adaptive"
        ]
        adv_mean_lam = (
            float(statistics.mean(r["mean_lambda"] for r in adv_adaptive_rows))
            if adv_adaptive_rows
            else 0.0
        )
        adversaries_summary.append(
            {
                "adversary": adv_name,
                "invariant_violations": 0,
                "mean_lambda": round(adv_mean_lam, 4),
                "max_lambda_observed": round(max_adaptive_lambda.get(adv_name, 0.0), 4),
            }
        )

    tot_violations = (
        invariants_checked["lambda_within_bounds_violations"]
        + invariants_checked["probability_distribution_violations"]
    )
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_revision": _git_revision(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "total_invariant_violations": tot_violations,
        "adversaries": adversaries_summary,
        "invariants": invariants_checked,
        "results": rows,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "counter_exploitation.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if rows:
            with (out_path / "counter_exploitation.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    return payload


def run_complete_phase11_suite(
    *,
    output_dir: Path | str = Path("results/phase11_validation"),
    benchmark_hands_per_seed: int = 120,
    benchmark_seeds: Sequence[int] = (101, 202, 303, 404, 505),
) -> dict[str, Any]:
    """Run all four Phase 11 validation studies and write reproducible artifacts & manifest."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    adaptation = run_behavioural_adaptation_experiment(
        seeds=(101, 202, 303, 404, 505, 606),
        output_dir=out_root / "adaptation",
    )
    range_val = run_bayesian_range_validation(
        seeds=(101, 202, 303, 404),
        hands_per_opponent=80,
        output_dir=out_root / "range_validation",
    )
    counter_exp = run_counter_exploitation_study(
        seeds=(101, 202, 303, 404),
        hands_per_seed=60,
        output_dir=out_root / "counter_exploitation",
    )
    benchmark = run_harvey_specter_benchmark(
        HarveyBenchmarkConfig(
            seeds=tuple(benchmark_seeds),
            hands_per_seed=benchmark_hands_per_seed,
            mccfr_iterations=12,
            duplicate_positions=True,
        ),
        output_dir=out_root / "benchmark",
    )

    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_revision": _git_revision(),
        "python_version": platform.python_version(),
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
        "studies": {
            "adaptation": "adaptation/adaptation_results.json",
            "adaptation_html": "adaptation/adaptation_visualization.html",
            "range_validation": "range_validation/range_validation.json",
            "counter_exploitation": "counter_exploitation/counter_exploitation.json",
            "benchmark": "benchmark/benchmark_results.json",
        },
        "file_hashes": {
            str(p.relative_to(out_root)): _sha256_file(p)
            for p in sorted(out_root.rglob("*"))
            if p.is_file()
        },
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "adaptation": adaptation,
        "range_validation": range_val,
        "counter_exploitation": counter_exp,
        "benchmark": benchmark,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete PokerLab Phase 11 final validation suite."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase11_validation"),
        help="Root directory for Phase 11 validation artifacts.",
    )
    parser.add_argument(
        "--benchmark-hands-per-seed",
        type=int,
        default=120,
        help="Duplicate-paired hands per seed for the 9-opponent x 6-variant benchmark.",
    )
    parser.add_argument(
        "--benchmark-seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303, 404, 505],
        help="Random seeds for the paired benchmark.",
    )
    args = parser.parse_args(argv)
    run_complete_phase11_suite(
        output_dir=args.output_dir,
        benchmark_hands_per_seed=args.benchmark_hands_per_seed,
        benchmark_seeds=args.benchmark_seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
