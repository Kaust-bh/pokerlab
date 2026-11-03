"""Paired multi-seed benchmark suite for HarveySpecter opponent modeling and exploitation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    ActionRecord,
    Agent,
    EquityAgent,
    EquityAgentConfig,
    HeuristicAgent,
    HeuristicConfig,
    RandomAgent,
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
)
from pokerlab.research.nlhe.agent import AbstractHUNLAgent
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.mccfr import (
    ExternalSamplingMCCFR,
    HUNLMCCFRConfig,
)

POLICY_VARIANTS: tuple[str, ...] = (
    "BaselineMCCFR",
    "HarveySpecter_Disabled",
    "HarveySpecter_Fixed_025",
    "HarveySpecter_Fixed",
    "HarveySpecter_Adaptive",
    "HarveySpecter_PureExploit",
)

OPPONENT_SPECS: tuple[tuple[str, str], ...] = (
    ("RandomAgent", "loose_aggressive"),
    ("HeuristicAgent", "tight_passive"),
    ("EquityAgent", "tight_aggressive"),
    ("TightPassive", "tight_passive"),
    ("CallingStation", "calling_station"),
    ("LooseAggressive", "loose_aggressive"),
    ("Maniac", "maniac"),
    ("Nit", "nit"),
    ("MCCFRBaseline", "balanced_unknown"),
)


@dataclass(frozen=True, slots=True)
class HarveyBenchmarkConfig:
    """Configuration for paired HarveySpecter evaluation experiments."""

    seeds: tuple[int, ...] = (101, 202, 303, 404)
    hands_per_seed: int = 80
    mccfr_iterations: int = 12
    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2
    fixed_lambda: float = 0.50
    max_lambda: float = 0.80
    equity_simulations: int = 32
    opponent_equity_simulations: int = 40
    duplicate_positions: bool = True
    postflop_method: str = "sampled_strength"


def _t_critical_95(n: int) -> float:
    if n <= 1:
        return 0.0
    df = n - 1
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        25: 2.060,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    if df in table:
        return table[df]
    for k in sorted(table):
        if df < k:
            return table[k]
    return 1.960


def _mean_and_ci95(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Return `(mean, std_error, ci95_low, ci95_high)`."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_val = float(statistics.mean(values))
    if n == 1:
        return mean_val, 0.0, mean_val, mean_val
    sd = float(statistics.stdev(values))
    se = sd / math.sqrt(n)
    t_crit = _t_critical_95(n)
    return mean_val, se, mean_val - t_crit * se, mean_val + t_crit * se


def _source_revision() -> str:
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


def build_mccfr_baseline(
    cfg: HarveyBenchmarkConfig, seed: int
) -> tuple[dict, dict, HUNLGameConfig]:
    stack_bb = max(2.0, round(cfg.starting_stack / max(1, cfg.big_blind), 1))
    game_cfg = HUNLGameConfig(
        starting_stack_bb=stack_bb,
        small_blind_bb=cfg.small_blind / cfg.big_blind,
        big_blind_bb=1.0,
        chips_per_bb=cfg.big_blind,
        postflop_method=cfg.postflop_method,
        postflop_buckets=8,
        equity_samples=8,
        board_abstraction="isomorphic",
    )
    solver = ExternalSamplingMCCFR(
        HUNLMCCFRConfig(
            iterations=cfg.mccfr_iterations,
            seed=seed,
            game=game_cfg,
        )
    )
    solver.train(cfg.mccfr_iterations)
    strategy = solver.average_strategy_profile()
    strategy_sums = {k: dict(v) for k, v in solver.strategy_sums.items()}
    return strategy, strategy_sums, game_cfg


def instantiate_opponent(
    opponent_name: str,
    *,
    cfg: HarveyBenchmarkConfig,
    seed: int,
    mccfr_bundle: tuple[dict, dict, HUNLGameConfig],
) -> Agent:
    strategy, strategy_sums, game_cfg = mccfr_bundle
    if opponent_name == "RandomAgent":
        return RandomAgent(seed=seed)
    if opponent_name == "HeuristicAgent":
        return HeuristicAgent(HeuristicConfig())
    if opponent_name == "EquityAgent":
        return EquityAgent(
            EquityAgentConfig(simulations=cfg.opponent_equity_simulations),
            seed=seed,
        )
    if opponent_name == "TightPassive":
        return TightPassiveAgent(seed=seed)
    if opponent_name == "CallingStation":
        return CallingStationAgent(seed=seed)
    if opponent_name == "LooseAggressive":
        return LooseAggressiveAgent(seed=seed)
    if opponent_name == "Maniac":
        return ManiacAgent(seed=seed)
    if opponent_name == "Nit":
        return NitAgent(seed=seed)
    if opponent_name == "MCCFRBaseline":
        return AbstractHUNLAgent(
            strategy=strategy,
            config=game_cfg,
            seed=seed,
            strategy_sums=strategy_sums,
        )
    raise ValueError(f"Unknown opponent_name {opponent_name!r}")


def instantiate_policy_variant(
    variant_name: str,
    *,
    cfg: HarveyBenchmarkConfig,
    seed: int,
    opponent_id: str,
    mccfr_bundle: tuple[dict, dict, HUNLGameConfig],
) -> Agent:
    strategy, strategy_sums, game_cfg = mccfr_bundle
    base_mccfr = AbstractHUNLAgent(
        strategy=strategy,
        config=game_cfg,
        seed=seed,
        strategy_sums=strategy_sums,
    )
    if variant_name == "BaselineMCCFR":
        return base_mccfr
    if variant_name == "HarveySpecter_Disabled":
        h_cfg = HarveySpecterConfig(
            mode="baseline_only",
            fixed_lambda=0.0,
            max_lambda=0.0,
            equity_simulations=cfg.equity_simulations,
        )
        return HarveySpecterAgent(
            base_mccfr,
            config=h_cfg,
            opponent_id=opponent_id,
            seed=seed,
        )
    if variant_name in ("HarveySpecter_Fixed_025", "HarveySpecter_Fixed025"):
        h_cfg = HarveySpecterConfig(
            mode="fixed",
            fixed_lambda=0.25,
            max_lambda=cfg.max_lambda,
            equity_simulations=cfg.equity_simulations,
        )
        return HarveySpecterAgent(
            base_mccfr,
            config=h_cfg,
            opponent_id=opponent_id,
            seed=seed,
        )
    if variant_name in ("HarveySpecter_Fixed", "HarveySpecter_Fixed_050"):
        h_cfg = HarveySpecterConfig(
            mode="fixed",
            fixed_lambda=cfg.fixed_lambda,
            max_lambda=cfg.max_lambda,
            equity_simulations=cfg.equity_simulations,
        )
        return HarveySpecterAgent(
            base_mccfr,
            config=h_cfg,
            opponent_id=opponent_id,
            seed=seed,
        )
    if variant_name == "HarveySpecter_Adaptive":
        h_cfg = HarveySpecterConfig(
            mode="adaptive",
            fixed_lambda=cfg.fixed_lambda,
            max_lambda=cfg.max_lambda,
            equity_simulations=cfg.equity_simulations,
        )
        return HarveySpecterAgent(
            base_mccfr,
            config=h_cfg,
            opponent_id=opponent_id,
            seed=seed,
        )
    if variant_name == "HarveySpecter_PureExploit":
        h_cfg = HarveySpecterConfig(
            mode="pure_exploit",
            fixed_lambda=1.0,
            max_lambda=1.0,
            equity_simulations=cfg.equity_simulations,
        )
        return HarveySpecterAgent(
            base_mccfr,
            config=h_cfg,
            opponent_id=opponent_id,
            seed=seed,
        )
    raise ValueError(f"Unknown policy variant {variant_name!r}")


def run_single_series(
    *,
    policy_agent: Agent,
    opponent_agent: Agent,
    opponent_id: str,
    cfg: HarveyBenchmarkConfig,
    series_seed: int,
) -> dict[str, Any]:
    """Run a fixed-hand series with paired deck seeds and collect performance & calibration metrics."""
    deck_rng = random.Random(series_seed)
    if cfg.duplicate_positions:
        # True duplicate pairing: each deal seed is played twice (once Hero on Button, once Villain on Button)
        n_deals = max(1, (cfg.hands_per_seed + 1) // 2)
        deal_seeds = [deck_rng.randrange(2**32) for _ in range(n_deals)]
        hand_deck_seeds: list[int] = []
        for ds in deal_seeds:
            hand_deck_seeds.append(ds)
            if len(hand_deck_seeds) < cfg.hands_per_seed:
                hand_deck_seeds.append(ds)
    else:
        hand_deck_seeds = [deck_rng.randrange(2**32) for _ in range(cfg.hands_per_seed)]

    policy_agent.reset(series_seed + 11)
    opponent_agent.reset(series_seed + 29)

    is_harvey = isinstance(policy_agent, HarveySpecterAgent)
    hand_bb_100: list[float] = []
    pots: list[int] = []
    wins = 0
    ties = 0
    showdowns = 0
    folds = 0
    likelihood_ratios: list[float] = []
    brier_post_scores: list[float] = []
    brier_prior_scores: list[float] = []
    log_loss_post_scores: list[float] = []
    log_loss_prior_scores: list[float] = []
    top10_hits: list[float] = []

    for hand_idx, deck_seed in enumerate(hand_deck_seeds):
        hand_num = hand_idx + 1
        hero_is_button = (hand_idx % 2) == 0
        button_name = "Hero" if hero_is_button else "Villain"
        # `GameState` deals hole cards in dict-insertion order (independent of the
        # button). For a true duplicate mirror the second hand of each pair must
        # give Hero the cards Villain held in the first hand (and vice versa) on
        # the identical board, so the insertion order is swapped on mirror hands.
        # Before Phase 11 the order was fixed, so the "mirror" hand re-dealt Hero
        # the same hole cards and did not cancel card luck.
        if cfg.duplicate_positions and not hero_is_button:
            initial = {"Villain": cfg.starting_stack, "Hero": cfg.starting_stack}
        else:
            initial = {"Hero": cfg.starting_stack, "Villain": cfg.starting_stack}

        game = GameState(
            initial,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            button=button_name,
            deck=Deck(seed=deck_seed),
        )
        if is_harvey:
            assert isinstance(policy_agent, HarveySpecterAgent)
            policy_agent.start_hand(
                opponent_id=opponent_id,
                hand_number=hand_num,
                opponent_is_button=not hero_is_button,
                starting_stack=cfg.starting_stack,
                effective_stack=cfg.starting_stack,
                big_blind=cfg.big_blind,
                hero_hole=game.hole_cards["Hero"],
                community_cards=(),
            )

        history: list[ActionRecord] = []
        steps = 0
        while not game.complete:
            actor = game.current_player
            if actor is None:
                break
            obs = observe(game, actor, tuple(history))
            act = policy_agent.act(obs) if actor == "Hero" else opponent_agent.act(obs)
            street = game.street.value
            game.act(act, player=actor)
            history.append(ActionRecord(actor, act, street))
            steps += 1
            if steps > 200:
                raise RuntimeError("Betting action limit exceeded")

        hero_net = game.stacks["Hero"] - initial["Hero"]
        villain_net = game.stacks["Villain"] - initial["Villain"]
        # Independent chip-conservation verification: zero-sum and bounded by starting_stack
        if hero_net + villain_net != 0:
            raise AssertionError(
                f"Chip conservation violated on hand {hand_num}: hero_net={hero_net}, villain_net={villain_net}"
            )
        if abs(hero_net) > cfg.starting_stack:
            raise AssertionError(
                f"Stack bound violated on hand {hand_num}: |hero_net|={abs(hero_net)} > {cfg.starting_stack}"
            )

        bb100_hand = (hero_net / cfg.big_blind) * 100.0
        hand_bb_100.append(bb100_hand)
        pots.append(game.current_pot)
        if hero_net > 0:
            wins += 1
        elif hero_net == 0:
            ties += 1

        reached_sd = bool(game.hand_ranks and "Villain" in game.hand_ranks)
        if reached_sd:
            showdowns += 1
        else:
            folds += 1
        revealed_villain_cards = game.hole_cards["Villain"] if reached_sd else None

        if is_harvey:
            assert isinstance(policy_agent, HarveySpecterAgent)
            # Evaluate Bayesian range prediction quality BEFORE showdown calibration updates
            prof = policy_agent.get_profile(opponent_id)
            policy_agent.finalize_hand(
                opponent_id=opponent_id,
                hand_number=hand_num,
                history=tuple(history),
                hero_player="Hero",
                community_cards=tuple(game.community_cards),
                reached_showdown=reached_sd,
                revealed_opponent_cards=revealed_villain_cards,
                opponent_net_chips=villain_net,
                opponent_won=(villain_net > 0),
                final_pot=game.current_pot,
            )
            rec = prof.collector.get_hand(opponent_id, hand_num)
            if rec is not None and rec.opponent_actions:
                opp_pos = "button" if not hero_is_button else "big_blind"
                gt_eval = prof.range_model.evaluate_ground_truth_hand(
                    game.hole_cards["Villain"],
                    position=opp_pos,
                    stats=prof.stats_snapshot(),
                )
                likelihood_ratios.append(gt_eval["likelihood_ratio"])
                brier_post_scores.append(gt_eval["brier_class_score"])
                brier_prior_scores.append(gt_eval["brier_prior_score"])
                log_loss_post_scores.append(gt_eval["log_loss_post"])
                log_loss_prior_scores.append(gt_eval["log_loss_prior"])
                top10_hits.append(gt_eval["top_10_hit"])

    if is_harvey:
        assert isinstance(policy_agent, HarveySpecterAgent)
        prof = policy_agent.get_profile(opponent_id)
        arch = prof.archetype_distribution()
        stats = prof.stats_snapshot()
        mean_lam = (
            float(statistics.mean(policy_agent.lambda_history))
            if policy_agent.lambda_history
            else 0.0
        )
        final_lam = (
            policy_agent.lambda_history[-1] if policy_agent.lambda_history else 0.0
        )
        predicted_arch = arch.most_likely
        arch_probs = dict(arch.probabilities)
        arch_conf = arch.confidence
        vpip_sm = stats.vpip.smoothed_value
        pfr_sm = stats.pfr.smoothed_value
        afq_sm = stats.aggression_frequency.smoothed_value
    else:
        mean_lam = 0.0
        final_lam = 0.0
        predicted_arch = "none"
        arch_probs = {}
        arch_conf = 0.0
        vpip_sm = 0.0
        pfr_sm = 0.0
        afq_sm = 0.0

    n_total = max(1, len(hand_bb_100))
    return {
        "hand_bb_100": hand_bb_100,
        "mean_bb_100": float(statistics.mean(hand_bb_100)),
        "win_rate": wins / n_total,
        "tie_rate": ties / n_total,
        "showdown_frequency": showdowns / n_total,
        "fold_frequency": folds / n_total,
        "average_pot": float(statistics.mean(pots)) if pots else 0.0,
        "mean_lambda": mean_lam,
        "final_lambda": final_lam,
        "predicted_archetype": predicted_arch,
        "archetype_probabilities": arch_probs,
        "archetype_confidence": arch_conf,
        "observed_vpip": vpip_sm,
        "observed_pfr": pfr_sm,
        "observed_afq": afq_sm,
        "mean_likelihood_ratio": (
            float(statistics.mean(likelihood_ratios)) if likelihood_ratios else 1.0
        ),
        "mean_brier_post": (
            float(statistics.mean(brier_post_scores)) if brier_post_scores else 1.0
        ),
        "mean_brier_prior": (
            float(statistics.mean(brier_prior_scores)) if brier_prior_scores else 1.0
        ),
        "mean_log_loss_post": (
            float(statistics.mean(log_loss_post_scores))
            if log_loss_post_scores
            else 5.1299
        ),
        "mean_log_loss_prior": (
            float(statistics.mean(log_loss_prior_scores))
            if log_loss_prior_scores
            else 5.1299
        ),
        "top_10_accuracy": (float(statistics.mean(top10_hits)) if top10_hits else 0.0),
    }


def run_harvey_specter_benchmark(
    config: HarveyBenchmarkConfig | None = None,
    *,
    output_dir: Path | str | None = None,
    opponents: Sequence[tuple[str, str]] | None = None,
    variants: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run paired multi-seed benchmark across opponents and policy variants."""
    cfg = config or HarveyBenchmarkConfig()
    opp_list = tuple(opponents) if opponents is not None else OPPONENT_SPECS
    var_list = tuple(variants) if variants is not None else POLICY_VARIANTS

    started = time.perf_counter()

    # Pre-train one MCCFR strategy bundle per seed so all variants share the exact same baseline
    mccfr_by_seed: dict[int, tuple[dict, dict, HUNLGameConfig]] = {
        seed: build_mccfr_baseline(cfg, seed) for seed in cfg.seeds
    }

    summary_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for opp_name, expected_arch in opp_list:
        per_variant_runs: dict[str, list[dict[str, Any]]] = {v: [] for v in var_list}
        for seed in cfg.seeds:
            series_seed = (seed * 31_337) ^ (sum(ord(c) for c in opp_name) * 997)
            bundle = mccfr_by_seed[seed]
            for variant in var_list:
                policy = instantiate_policy_variant(
                    variant,
                    cfg=cfg,
                    seed=series_seed,
                    opponent_id=opp_name,
                    mccfr_bundle=bundle,
                )
                opp = instantiate_opponent(
                    opp_name,
                    cfg=cfg,
                    seed=series_seed + 777,
                    mccfr_bundle=bundle,
                )
                res = run_single_series(
                    policy_agent=policy,
                    opponent_agent=opp,
                    opponent_id=opp_name,
                    cfg=cfg,
                    series_seed=series_seed,
                )
                per_variant_runs[variant].append(res)
                seed_rows.append(
                    {
                        "opponent": opp_name,
                        "expected_archetype": expected_arch,
                        "variant": variant,
                        "seed": seed,
                        "hands": cfg.hands_per_seed,
                        "bb_per_100": round(res["mean_bb_100"], 4),
                        "win_rate": round(res["win_rate"], 4),
                        "average_pot": round(res["average_pot"], 2),
                        "mean_lambda": round(res["mean_lambda"], 4),
                        "final_lambda": round(res["final_lambda"], 4),
                        "predicted_archetype": res["predicted_archetype"],
                        "archetype_confidence": round(res["archetype_confidence"], 4),
                        "range_likelihood_ratio": round(
                            res["mean_likelihood_ratio"], 4
                        ),
                        "brier_post": round(res["mean_brier_post"], 6),
                        "brier_prior": round(res["mean_brier_prior"], 6),
                    }
                )

        baseline_hand_samples: list[float] = []
        if "BaselineMCCFR" in per_variant_runs:
            for r in per_variant_runs["BaselineMCCFR"]:
                baseline_hand_samples.extend(r["hand_bb_100"])

        for variant in var_list:
            runs = per_variant_runs[variant]
            all_hand_bb100: list[float] = []
            for r in runs:
                all_hand_bb100.extend(r["hand_bb_100"])

            mean_bb, se_bb, ci_low, ci_high = _mean_and_ci95(all_hand_bb100)

            if baseline_hand_samples and len(baseline_hand_samples) == len(
                all_hand_bb100
            ):
                paired_diffs = [
                    a - b
                    for a, b in zip(all_hand_bb100, baseline_hand_samples, strict=True)
                ]
                gain_mean, gain_se, gain_ci_low, gain_ci_high = _mean_and_ci95(
                    paired_diffs
                )
                diff_sd = (
                    float(statistics.stdev(paired_diffs))
                    if len(paired_diffs) > 1
                    else 0.0
                )
                cohens_d = (gain_mean / diff_sd) if diff_sd > 1e-9 else 0.0
                paired_t = (gain_mean / gain_se) if gain_se > 1e-9 else 0.0
            else:
                gain_mean = gain_se = gain_ci_low = gain_ci_high = 0.0
                cohens_d = paired_t = 0.0

            win_rate = float(statistics.mean(r["win_rate"] for r in runs))
            sd_freq = float(statistics.mean(r["showdown_frequency"] for r in runs))
            fold_freq = float(statistics.mean(r["fold_frequency"] for r in runs))
            avg_pot = float(statistics.mean(r["average_pot"] for r in runs))
            mean_lam = float(statistics.mean(r["mean_lambda"] for r in runs))
            final_lam = float(statistics.mean(r["final_lambda"] for r in runs))
            lr_mean = float(statistics.mean(r["mean_likelihood_ratio"] for r in runs))
            brier_post = float(statistics.mean(r["mean_brier_post"] for r in runs))
            brier_prior = float(statistics.mean(r["mean_brier_prior"] for r in runs))
            ll_post = float(statistics.mean(r["mean_log_loss_post"] for r in runs))
            ll_prior = float(statistics.mean(r["mean_log_loss_prior"] for r in runs))
            top10_acc = float(statistics.mean(r["top_10_accuracy"] for r in runs))

            arch_hits = sum(
                1
                for r in runs
                if r["predicted_archetype"] == expected_arch
                or (
                    expected_arch in r["archetype_probabilities"]
                    and r["archetype_probabilities"][expected_arch] >= 0.22
                )
            )
            arch_accuracy = arch_hits / max(1, len(runs))
            target_arch_prob = float(
                statistics.mean(
                    r["archetype_probabilities"].get(expected_arch, 0.0) for r in runs
                )
            )
            most_common_pred = runs[-1]["predicted_archetype"]

            summary_rows.append(
                {
                    "opponent": opp_name,
                    "expected_archetype": expected_arch,
                    "variant": variant,
                    "total_hands": len(all_hand_bb100),
                    "seeds_count": len(cfg.seeds),
                    "bb_per_100": round(mean_bb, 4),
                    "se_bb_per_100": round(se_bb, 4),
                    "ci95_low_bb_per_100": round(ci_low, 4),
                    "ci95_high_bb_per_100": round(ci_high, 4),
                    "paired_gain_vs_baseline_bb_per_100": round(gain_mean, 4),
                    "paired_gain_se": round(gain_se, 4),
                    "paired_gain_ci95_low": round(gain_ci_low, 4),
                    "paired_gain_ci95_high": round(gain_ci_high, 4),
                    "paired_t_stat": round(paired_t, 4),
                    "effect_size_cohens_d": round(cohens_d, 4),
                    "win_rate": round(win_rate, 4),
                    "showdown_frequency": round(sd_freq, 4),
                    "fold_frequency": round(fold_freq, 4),
                    "average_pot": round(avg_pot, 2),
                    "mean_lambda": round(mean_lam, 4),
                    "final_lambda": round(final_lam, 4),
                    "predicted_archetype": most_common_pred,
                    "target_archetype_prob": round(target_arch_prob, 4),
                    "archetype_accuracy": round(arch_accuracy, 4),
                    "range_likelihood_ratio": round(lr_mean, 4),
                    "range_brier_post": round(brier_post, 6),
                    "range_brier_prior": round(brier_prior, 6),
                    "range_brier_improvement": round(brier_prior - brier_post, 6),
                    "range_log_loss_post": round(ll_post, 4),
                    "range_log_loss_prior": round(ll_prior, 4),
                    "range_top10_accuracy": round(top10_acc, 4),
                }
            )

    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(),
        "python_version": platform.python_version(),
        "runtime_seconds": round(elapsed, 3),
        "config": asdict(cfg),
        "summary": summary_rows,
        "seed_runs": seed_rows,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        summary_csv_path = out_path / "summary.csv"
        seed_csv_path = out_path / "seed_runs.csv"
        json_path = out_path / "benchmark_results.json"

        if summary_rows:
            with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        if seed_rows:
            with seed_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0].keys()))
                writer.writeheader()
                writer.writerows(seed_rows)

        hashes: dict[str, str] = {}
        if summary_csv_path.exists():
            hashes["summary.csv"] = _sha256_file(summary_csv_path)
        if seed_csv_path.exists():
            hashes["seed_runs.csv"] = _sha256_file(seed_csv_path)
        payload["artifact_sha256"] = hashes
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run paired multi-seed HarveySpecter adaptive poker benchmark."
    )
    parser.add_argument(
        "--hands-per-seed",
        type=int,
        default=80,
        help="Number of paired hands per seed per opponent.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303, 404],
        help="Random seeds for paired evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/harvey_specter/latest"),
        help="Directory to write JSON and CSV benchmark artifacts.",
    )
    args = parser.parse_args(argv)
    cfg = HarveyBenchmarkConfig(
        seeds=tuple(args.seeds),
        hands_per_seed=args.hands_per_seed,
    )
    run_harvey_specter_benchmark(cfg, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
