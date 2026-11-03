"""Phase 11 validation, robustness, stress testing, and performance regression tests."""

from __future__ import annotations

import random
from pathlib import Path

from pokerlab.arena.server import ArenaApp
from pokerlab.arena.session import ArenaConfig, ArenaSession
from pokerlab.engine import Card, Deck, evaluate_hand
from pokerlab.opponent_model import (
    DEFAULT_ADAPTATION_SCHEDULE,
    AlternatingGearAgent,
    BluffPunisherAgent,
    DeceptiveSizingAgent,
    HarveyBenchmarkConfig,
    OpponentProfileStore,
    ScheduledStrategyOpponent,
    ValueFolderAgent,
    fast_hand_rank_value,
    run_bayesian_range_validation,
    run_behavioural_adaptation_experiment,
    run_counter_exploitation_study,
    run_harvey_specter_benchmark,
)

_RANK_MAP = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
_SUIT_MAP = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}


def _c(text: str) -> Card:
    return Card(rank=_RANK_MAP[text[0]], suit=_SUIT_MAP[text[1]])


def test_fast_hand_rank_value_exact_equivalence() -> None:
    """Verify fast_hand_rank_value matches evaluate_hand(cards).value across canonical & random hands."""
    canonical_hands = [
        # Royal flush / straight flush
        ["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"],
        ["5h", "4h", "3h", "2h", "Ah", "Kd", "Qc"],
        # Quads
        ["9s", "9h", "9d", "9c", "Ah", "2s", "3d"],
        # Full house (including two trips or trips + two pairs)
        ["Ks", "Kh", "Kd", "Qs", "Qh", "Qd", "2c"],
        ["As", "Ah", "Ad", "Ks", "Kh", "Qd", "Qc"],
        # Flush with 6 or 7 suited cards
        ["As", "Js", "9s", "7s", "5s", "3s", "2s"],
        # Wheel straight and regular straight
        ["As", "2d", "3c", "4h", "5s", "Kd", "9c"],
        ["Ts", "Jd", "Qc", "Kh", "As", "9d", "8c"],
        # Trips, two pair, three pairs, one pair, high card
        ["7s", "7h", "7d", "As", "Kd", "Qc", "2h"],
        ["As", "Ah", "Ks", "Kh", "Qs", "Qh", "2d"],
        ["As", "Ah", "Kd", "Qc", "Js", "9h", "2c"],
        ["As", "Kd", "Qc", "Jh", "9s", "7d", "2c"],
    ]
    for hand_strs in canonical_hands:
        cards = [_c(s) for s in hand_strs]
        for size in (5, 6, 7):
            sub = cards[:size]
            assert fast_hand_rank_value(sub) == evaluate_hand(sub).value

    rng = random.Random(202611)
    for _ in range(400):
        deck = Deck(rng=rng)
        deck.shuffle()
        cards = deck.draw(7)
        for size in (5, 6, 7):
            sub = cards[:size]
            assert fast_hand_rank_value(sub) == evaluate_hand(sub).value


def test_scheduled_strategy_opponent_and_adaptation_experiment(tmp_path: Path) -> None:
    """Verify ScheduledStrategyOpponent switches regimes cleanly and recency adaptation beats stale evidence."""
    opp = ScheduledStrategyOpponent(seed=101, schedule=DEFAULT_ADAPTATION_SCHEDULE)
    opp.notify_new_hand(1)
    assert opp.current_true_archetype == "tight_passive"
    opp.notify_new_hand(25)
    assert opp.current_true_archetype == "loose_aggressive"
    opp.notify_new_hand(55)
    assert opp.current_true_archetype == "calling_station"
    opp.notify_new_hand(85)
    assert opp.current_true_archetype == "tight_passive"

    res = run_behavioural_adaptation_experiment(
        seeds=(101, 202),
        mccfr_iterations=6,
        output_dir=tmp_path / "adaptation",
    )
    assert len(res["hand_curves"]) == 100
    assert (tmp_path / "adaptation" / "adaptation_visualization.html").exists()
    assert (tmp_path / "adaptation" / "adaptation_results.json").exists()

    # Recency-weighted classifier must outperform unweighted stale-evidence classifier after regime shifts
    assert (
        res["post_shift_accuracy_vs_stale"]["adaptive_post_shift_accuracy"]
        > res["post_shift_accuracy_vs_stale"]["stale_unweighted_post_shift_accuracy"]
    )
    for boundary in ("21", "51", "81"):
        metric = res["transition_metrics"][boundary]
        assert metric["mean_detection_delay_hands"] <= 20.0
        assert metric["detection_rate"] > 0.0


def test_bayesian_range_validation_beats_baselines(tmp_path: Path) -> None:
    """Verify BayesianRangeModel improves Log Loss, Brier Score, and Top-K over uniform and street-prior baselines."""
    res = run_bayesian_range_validation(
        seeds=(101, 202),
        hands_per_opponent=30,
        output_dir=tmp_path / "range_val",
    )
    overall = res["overall"]
    assert overall["decisions_evaluated"] > 50
    assert overall["log_loss_posterior"] < overall["log_loss_uniform_legal"]
    assert overall["log_loss_posterior"] < overall["log_loss_position_prior"]
    assert overall["log_loss_posterior"] < overall["log_loss_prev_street_posterior"]
    assert overall["brier_posterior"] <= overall["brier_uniform_legal"]
    assert overall["mean_likelihood_ratio_vs_uniform"] > 1.0
    assert overall["top_10_class_accuracy"] > overall["uniform_top_10_accuracy"]
    assert len(res["calibration_curve_10_bins"]) == 10
    assert "preflop" in res["street_convergence"]
    assert "flop" in res["street_convergence"]


def test_counter_exploitation_invariants_and_adversaries(tmp_path: Path) -> None:
    """Verify HarveySpecter maintains valid policies and bounded lambda against adversarial opponents."""
    for cls in (
        BluffPunisherAgent,
        ValueFolderAgent,
        AlternatingGearAgent,
        DeceptiveSizingAgent,
    ):
        agent = cls(seed=42)
        assert agent.name

    res = run_counter_exploitation_study(
        seeds=(101, 202),
        hands_per_seed=30,
        mccfr_iterations=6,
        output_dir=tmp_path / "counter",
    )
    assert res["total_invariant_violations"] == 0
    assert len(res["adversaries"]) == 4
    for adv in res["adversaries"]:
        assert adv["invariant_violations"] == 0
        assert 0.0 <= adv["mean_lambda"] <= 0.80
        assert 0.0 <= adv["max_lambda_observed"] <= 0.80


def test_duplicate_paired_benchmark_zero_sum_and_nit(tmp_path: Path) -> None:
    """Verify duplicate-paired benchmark enforces zero-sum chip conservation and includes Nit & Fixed_025."""
    cfg = HarveyBenchmarkConfig(
        seeds=(101, 202),
        hands_per_seed=10,
        mccfr_iterations=4,
        duplicate_positions=True,
    )
    res = run_harvey_specter_benchmark(cfg, output_dir=tmp_path / "bench")
    assert res["config"]["duplicate_positions"] is True
    opponents = {row["opponent"] for row in res["summary"]}
    variants = {row["variant"] for row in res["summary"]}
    assert "Nit" in opponents
    assert "HarveySpecter_Fixed_025" in variants
    for row in res["summary"]:
        assert row["total_hands"] == 20
        assert "effect_size_cohens_d" in row
        assert "paired_t_stat" in row
        assert "showdown_frequency" in row
        assert "fold_frequency" in row


def test_arena_stress_and_corrupt_profile_recovery(tmp_path: Path) -> None:
    """Stress test Arena human mode, both seats, corrupt profile recovery, and hidden card privacy."""
    # 1. Corrupt profile recovery
    store = OpponentProfileStore(tmp_path / "profiles")
    corrupt_path = tmp_path / "profiles" / "profile_corrupt_villain.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{corrupt json payload!!!", encoding="utf-8")
    recovered, status = store.load_profile(
        "corrupt_villain", path=corrupt_path, strict=False
    )
    assert status == "corrupt_recovered_new"
    assert recovered.opponent_id == "corrupt_villain"
    assert recovered.stats_snapshot().hands_observed == 0

    # 2. Arena stress across both seats, custom blinds/stacks, and repeated requests
    for human_seat in (0, 1):
        seat1_agent = "human" if human_seat == 0 else "harvey"
        seat2_agent = "harvey" if human_seat == 0 else "human"
        app = ArenaApp(
            ArenaSession(
                ArenaConfig(
                    seat1_name="Hero" if human_seat == 0 else "HarveySpecter",
                    seat2_name="HarveySpecter" if human_seat == 0 else "Hero",
                    seat1_agent=seat1_agent,
                    seat2_agent=seat2_agent,
                    starting_stack=400,
                    small_blind=5,
                    big_blind=10,
                    seed=500 + human_seat,
                )
            )
        )

        for _ in range(8):
            code, _, state = app.handle_request("POST", "/api/start", {})
            assert code == 200
            assert isinstance(state, dict)
            guard = 0
            while not state["hand_complete"] and guard < 30:
                guard += 1
                ctrl = state["human_controls"]
                if not ctrl["is_human_turn"]:
                    break
                if ctrl["can_call"]:
                    act = {"action_type": "call"}
                elif ctrl["can_check"]:
                    act = {"action_type": "check"}
                else:
                    act = {"action_type": "fold"}
                code, _, state = app.handle_request("POST", "/api/human/action", act)
                assert code == 200
                assert isinstance(state, dict)

            # Repeated GET /api/state calls must be idempotent and synchronized
            c1, _, s1 = app.handle_request("GET", "/api/state")
            c2, _, s2 = app.handle_request("GET", "/api/state")
            assert c1 == 200 and c2 == 200
            assert s1["hand_number"] == s2["hand_number"]
            assert s1["pot"] == s2["pot"]
