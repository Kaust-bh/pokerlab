import json
from dataclasses import replace
from pathlib import Path

import pytest

from pokerlab.agents import RandomAgent
from pokerlab.evaluation import TournamentConfig, TournamentSimulator
from pokerlab.research.nlhe.analysis import (
    mean_interval,
    paired_abstraction_comparisons,
    summarize_matches,
)
from pokerlab.research.nlhe.benchmark import (
    BenchmarkPlan,
    estimate_plan,
    run_benchmark,
)


def test_fixed_hand_batch_reproduces_exact_hand_count_and_conserves_chips():
    simulator = TournamentSimulator()
    simulator.register("left", RandomAgent)
    simulator.register("right", RandomAgent)
    config = TournamentConfig(starting_stack=40, hand_limit=5)
    first = simulator.run_fixed_hands(
        "left", "right", hands=5, config=config, seed=92, button="right"
    )
    second = simulator.run_fixed_hands(
        "left", "right", hands=5, config=config, seed=92, button="right"
    )
    assert first.hand_net_chips == second.hand_net_chips
    assert first.hands_played == 5
    assert first.positions == {"left": 0, "right": 5}
    assert sum(first.net_chips.values()) == 0
    assert sum(first.hand_wins.values()) + first.hand_ties == 5


def test_t_intervals_and_paired_comparisons_use_seed_level_batches():
    average, deviation, standard_error, low, high = mean_interval([10.0, 12.0])
    assert average == 11.0
    assert deviation == pytest.approx(2**0.5)
    assert standard_error == pytest.approx(1.0)
    assert (low, high) == pytest.approx((-1.706, 23.706), abs=0.002)
    assert mean_interval([7.0]) == (7.0, 0.0, 0.0, None, None)

    rows = []
    for seed, baseline_value, variant_value in (
        (1, 2.0, 5.0),
        (2, 4.0, 8.0),
        (3, 7.0, 6.0),
    ):
        for variant, value in (("baseline", baseline_value), ("wide", variant_value)):
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "checkpoint_iteration": 10,
                    "opponent": "Random",
                    "trained_first_button": "trained",
                    "bb_per_100": value,
                    "hands_played": 100,
                    "hand_wins": 75,
                    "hand_ties": 1,
                    "match_win": int(value > 0),
                    "runtime_seconds": 1.0,
                    "trained_seconds_per_decision": 0.01,
                }
            )
    paired = paired_abstraction_comparisons(rows)
    assert len(paired) == 1
    assert paired[0]["mean_delta_bb_per_100"] == pytest.approx(2.0)
    assert paired[0]["paired_seed_matches"] == 3
    summary = summarize_matches(rows)
    assert all(item["matches"] == 3 for item in summary)
    assert all(item["ci95_low_bb_per_100"] is not None for item in summary)
    assert all(
        item["hand_win_rate_excluding_ties"] == pytest.approx(75 / 99)
        for item in summary
    )


def test_benchmark_estimate_and_smoke_run_write_complete_artifacts(tmp_path: Path):
    plan = BenchmarkPlan(
        seeds=(31,),
        iterations=1,
        checkpoints=(1,),
        evaluation_hands=2,
        stack_bb=4,
        equity_samples=2,
        equity_agent_simulations=2,
    )
    estimate = estimate_plan(plan, ("baseline",))
    assert estimate["training_iterations"] == 1
    assert estimate["evaluation_hands_upper_bound"] == 12

    run_dir = run_benchmark(
        plan,
        variants=("baseline",),
        output_root=tmp_path,
    )
    assert (run_dir / "config.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "training.csv").exists()
    assert (run_dir / "evaluation.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert len(list((run_dir / "checkpoints").rglob("strategy.json"))) == 1

    checkpoint = next((run_dir / "checkpoints").rglob("checkpoint.json"))
    resumed = run_benchmark(
        replace(plan, iterations=2, checkpoints=(2,)),
        variants=("baseline",),
        output_root=tmp_path,
        resume_checkpoint=checkpoint,
    )
    assert resumed != run_dir
    assert json.loads((resumed / "summary.json").read_text())["resume_checkpoint"]
    assert len(list((resumed / "checkpoints").rglob("strategy.json"))) == 2
