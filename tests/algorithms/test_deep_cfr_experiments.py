import csv
import json

import pytest

try:
    import torch
except ImportError:
    torch = None

from pokerlab.algorithms.deep_cfr import DeepCFRConfig
from pokerlab.algorithms.deep_cfr.experiments import run_experiment
from pokerlab.algorithms.deep_cfr.study import run_study


@pytest.mark.skipif(torch is None, reason="requires optional PyTorch dependency")
@pytest.mark.parametrize("game", ("kuhn", "leduc"))
def test_experiment_runner_compares_all_solvers_and_writes_hashed_artifacts(
    game, tmp_path
):
    config = DeepCFRConfig(
        iterations=1,
        seed=81,
        traversals_per_player=1,
        hidden_layers=(8,),
        batch_size=32,
        advantage_memory_capacity=500,
        strategy_memory_capacity=500,
        evaluation_interval=1,
        iteration_weighting="uniform",
    )
    run_dir = run_experiment(game, config, output_root=tmp_path)
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "training.csv").is_file()
    assert (run_dir / "evaluation.csv").is_file()
    assert (run_dir / "checkpoints" / "deep-cfr-iter-1.pt").is_file()
    with (run_dir / "convergence.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["algorithm"] for row in rows} == {
        "deep_cfr",
        "vanilla_cfr",
        "cfr_plus",
        "dcfr",
    }
    assert all(float(row["exploitability"]) >= 0 for row in rows)
    deep_row = next(row for row in rows if row["algorithm"] == "deep_cfr")
    assert 0 <= float(deep_row["strategy_tv_from_vanilla_cfr"]) <= 1


@pytest.mark.skipif(torch is None, reason="requires optional PyTorch dependency")
def test_seeded_study_writes_paired_statistics_and_hashes(tmp_path):
    config = DeepCFRConfig(
        iterations=1,
        seed=0,
        traversals_per_player=1,
        hidden_layers=(8,),
        batch_size=32,
        advantage_memory_capacity=100,
        strategy_memory_capacity=100,
        evaluation_interval=1,
        iteration_weighting="uniform",
    )
    study = run_study(
        "kuhn",
        config,
        seeds=(3, 5),
        variants={"baseline": {}},
        output_root=tmp_path,
    )
    assert (study / "evaluation.csv").is_file()
    assert (study / "summary.csv").is_file()
    assert (study / "paired_comparisons.csv").is_file()
    assert (study / "paired_summary.csv").is_file()
    manifest = json.loads((study / "manifest.json").read_text())
    assert manifest["artifact_sha256"]
    summary = json.loads((study / "summary.json").read_text())
    assert summary["seed_count"] == 2
    assert summary["ci_method"].startswith("two-sided Student t")
