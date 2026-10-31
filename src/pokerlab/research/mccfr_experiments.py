"""Run reproducible vanilla CFR versus external-sampling MCCFR comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pokerlab.research.kuhn.cfr import CFRConfig as KuhnCFRConfig
from pokerlab.research.kuhn.cfr import CFRSolver as KuhnCFRSolver
from pokerlab.research.kuhn.evaluation import evaluate_nash as evaluate_kuhn
from pokerlab.research.kuhn.mccfr import MCCFRConfig as KuhnMCCFRConfig
from pokerlab.research.kuhn.mccfr import MCCFRSolver as KuhnMCCFRSolver
from pokerlab.research.kuhn.strategy import save_strategy_snapshot as save_kuhn_strategy
from pokerlab.research.leduc.cfr import CFRConfig as LeducCFRConfig
from pokerlab.research.leduc.cfr import CFRSolver as LeducCFRSolver
from pokerlab.research.leduc.evaluation import evaluate_nash as evaluate_leduc
from pokerlab.research.leduc.mccfr import MCCFRConfig as LeducMCCFRConfig
from pokerlab.research.leduc.mccfr import MCCFRSolver as LeducMCCFRSolver
from pokerlab.research.leduc.strategy import (
    save_strategy_snapshot as save_leduc_strategy,
)

Game = str
Algorithm = str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        return {"head": head, "dirty": bool(status), "status_porcelain": status}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None, "status_porcelain": []}


def _strategy_l1(
    first: Mapping[Any, Mapping[Any, float]],
    second: Mapping[Any, Mapping[Any, float]],
) -> float:
    infos = first.keys() | second.keys()
    total = 0.0
    for info in infos:
        actions = first.get(info, {}).keys() | second.get(info, {}).keys()
        total += sum(
            abs(
                first.get(info, {}).get(action, 0.0)
                - second.get(info, {}).get(action, 0.0)
            )
            for action in actions
        )
    return total


def _table_bytes(solver: Any) -> int:
    """Approximate persistent regret/average tables using Python object sizes."""
    total = sys.getsizeof(solver.regret_sum) + sys.getsizeof(solver.strategy_sum)
    for table in (solver.regret_sum, solver.strategy_sum):
        for info, values in table.items():
            total += sys.getsizeof(info) + sys.getsizeof(values)
            total += sum(
                sys.getsizeof(action) + sys.getsizeof(value)
                for action, value in values.items()
            )
    return total


def _new_solver(
    game: Game, algorithm: Algorithm, iterations: int, seed: int, interval: int
):
    if game == "kuhn" and algorithm == "vanilla_cfr":
        return KuhnCFRSolver(KuhnCFRConfig(iterations, seed, interval, interval))
    if game == "kuhn":
        return KuhnMCCFRSolver(KuhnMCCFRConfig(iterations, seed, interval, interval))
    if game == "leduc" and algorithm == "vanilla_cfr":
        return LeducCFRSolver(LeducCFRConfig(iterations, seed, interval, interval))
    return LeducMCCFRSolver(LeducMCCFRConfig(iterations, seed, interval, interval))


def _evaluate(game: Game, profile):
    return evaluate_kuhn(profile) if game == "kuhn" else evaluate_leduc(profile)


def _save_strategy(game: Game, profile, path: Path) -> None:
    (save_kuhn_strategy if game == "kuhn" else save_leduc_strategy)(profile, path)


def _run_one(
    game: Game,
    algorithm: Algorithm,
    seed: int,
    iterations: int,
    evaluation_interval: int,
    output_dir: Path,
) -> tuple[
    list[dict[str, Any]], dict[int, Mapping[Any, Mapping[Any, float]]], dict[str, Any]
]:
    solver = _new_solver(game, algorithm, iterations, seed, evaluation_interval)
    rows: list[dict[str, Any]] = []
    profiles = {}
    training_elapsed = 0.0
    evaluation_elapsed = 0.0
    peak_table_bytes = _table_bytes(solver)
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        solver.run_iteration()
        training_elapsed += time.perf_counter() - started
        should_evaluate = (
            iteration % evaluation_interval == 0 or iteration == iterations
        )
        if not should_evaluate:
            continue
        started = time.perf_counter()
        profile = solver.average_strategy_profile()
        result = _evaluate(game, profile)
        evaluation_elapsed += time.perf_counter() - started
        profiles[iteration] = profile
        peak_table_bytes = max(peak_table_bytes, _table_bytes(solver))
        row = {
            "game": game,
            "algorithm": algorithm,
            "seed": seed,
            "iteration": iteration,
            "expected_game_value_player0": result.expected_value_player0,
            "expected_game_value_player1": -result.expected_value_player0,
            "nashconv": result.nashconv,
            "exploitability": result.exploitability,
            "training_elapsed_seconds": training_elapsed,
            "evaluation_elapsed_seconds": evaluation_elapsed,
            "iterations_per_second": iteration / training_elapsed
            if training_elapsed
            else 0.0,
            "information_sets": len(profile),
            "nodes_last_iteration": getattr(solver, "sampled_nodes", None),
            "persistent_strategy_table_bytes_approx": peak_table_bytes,
        }
        rows.append(row)
    strategy_path = output_dir / f"strategy_{game}_{algorithm}_seed_{seed}.json"
    _save_strategy(game, profiles[iterations], strategy_path)
    return (
        rows,
        profiles,
        {
            "game": game,
            "algorithm": algorithm,
            "seed": seed,
            "iterations": iterations,
            "training_elapsed_seconds": training_elapsed,
            "evaluation_elapsed_seconds": evaluation_elapsed,
            "persistent_strategy_table_bytes_approx": peak_table_bytes,
            "strategy_file": strategy_path.name,
        },
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty experiment CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import plotly.graph_objects as go

    files = []
    for game in sorted({row["game"] for row in rows}):
        selected = [row for row in rows if row["game"] == game]
        for x_field, suffix, x_title in (
            ("iteration", "iterations", "Iterations"),
            ("training_elapsed_seconds", "training_time", "Training time (seconds)"),
        ):
            figure = go.Figure()
            for algorithm in ("vanilla_cfr", "external_sampling_mccfr"):
                for seed in sorted({row["seed"] for row in selected}):
                    curve = [
                        row
                        for row in selected
                        if row["algorithm"] == algorithm and row["seed"] == seed
                    ]
                    figure.add_trace(
                        go.Scatter(
                            x=[row[x_field] for row in curve],
                            y=[row["exploitability"] for row in curve],
                            mode="lines+markers",
                            name=f"{algorithm}, seed {seed}",
                        )
                    )
            figure.update_layout(
                title=f"{game.title()} exploitability vs {x_title.lower()}",
                xaxis_title=x_title,
                yaxis_title="Exact exploitability (chips per hand)",
            )
            target = output_dir / f"{game}_exploitability_vs_{suffix}.html"
            figure.write_html(target, include_plotlyjs="cdn")
            files.append(target)

        figure = go.Figure()
        for seed in sorted({row["seed"] for row in selected}):
            curve = [
                row
                for row in selected
                if row["algorithm"] == "external_sampling_mccfr" and row["seed"] == seed
            ]
            figure.add_trace(
                go.Scatter(
                    x=[row["iteration"] for row in curve],
                    y=[row["strategy_l1_to_vanilla"] for row in curve],
                    mode="lines+markers",
                    name=f"seed {seed}",
                )
            )
        figure.update_layout(
            title=f"{game.title()} average-strategy distance to vanilla CFR",
            xaxis_title="Iterations",
            yaxis_title="L1 distance summed over information sets",
        )
        target = output_dir / f"{game}_average_strategy_distance.html"
        figure.write_html(target, include_plotlyjs="cdn")
        files.append(target)
    return files


def run_comparison(
    *,
    games: tuple[str, ...] = ("kuhn", "leduc"),
    seeds: tuple[int, ...] = (7, 19),
    iterations: int = 1_000,
    evaluation_interval: int = 250,
    output_root: str | Path = "results/mccfr",
    make_plots: bool = True,
) -> Path:
    if not games or set(games) - {"kuhn", "leduc"}:
        raise ValueError("games must contain only kuhn and/or leduc")
    if not seeds or iterations < 1 or evaluation_interval < 1:
        raise ValueError("seeds, iterations, and evaluation interval must be positive")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_root) / f"run-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "games": list(games),
        "algorithms": ["vanilla_cfr", "external_sampling_mccfr"],
        "seeds": list(seeds),
        "iterations": iterations,
        "evaluation_interval": evaluation_interval,
        "training_rate_excludes_evaluation": True,
        "exact_average_strategy": True,
        "memory_estimate": "Approximate Python size of regret and average-strategy tables",
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    convergence_rows = []
    comparison_rows = []
    run_summaries = []
    for game in games:
        for seed in seeds:
            vanilla_rows, vanilla_profiles, vanilla_summary = _run_one(
                game,
                "vanilla_cfr",
                seed,
                iterations,
                evaluation_interval,
                output_dir,
            )
            sampled_rows, sampled_profiles, sampled_summary = _run_one(
                game,
                "external_sampling_mccfr",
                seed,
                iterations,
                evaluation_interval,
                output_dir,
            )
            run_summaries.extend((vanilla_summary, sampled_summary))
            convergence_rows.extend(vanilla_rows)
            convergence_rows.extend(sampled_rows)
            vanilla_by_iteration = {row["iteration"]: row for row in vanilla_rows}
            sampled_by_iteration = {row["iteration"]: row for row in sampled_rows}
            for iteration in sorted(
                vanilla_by_iteration.keys() & sampled_by_iteration.keys()
            ):
                exact_profile = vanilla_profiles[iteration]
                sampled_profile = sampled_profiles[iteration]
                distance = _strategy_l1(sampled_profile, exact_profile)
                for algorithm, source in (
                    ("vanilla_cfr", vanilla_by_iteration[iteration]),
                    ("external_sampling_mccfr", sampled_by_iteration[iteration]),
                ):
                    comparison_rows.append(
                        {
                            **{
                                key: source[key]
                                for key in (
                                    "game",
                                    "algorithm",
                                    "seed",
                                    "iteration",
                                    "expected_game_value_player0",
                                    "expected_game_value_player1",
                                    "nashconv",
                                    "exploitability",
                                    "training_elapsed_seconds",
                                    "evaluation_elapsed_seconds",
                                    "iterations_per_second",
                                    "persistent_strategy_table_bytes_approx",
                                )
                            },
                            "strategy_l1_to_vanilla": (
                                0.0 if algorithm == "vanilla_cfr" else distance
                            ),
                        }
                    )

    convergence_path = output_dir / "convergence.csv"
    comparison_path = output_dir / "comparison.csv"
    _write_csv(convergence_rows, convergence_path)
    _write_csv(comparison_rows, comparison_path)
    (output_dir / "run_metrics.json").write_text(
        json.dumps(run_summaries, indent=2) + "\n", encoding="utf-8"
    )
    if make_plots:
        _write_plots(comparison_rows, output_dir)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git": _git_metadata(),
        "configuration": config,
        "method_notes": [
            "Vanilla and MCCFR use the same exact best-response evaluator at checkpoints.",
            "Training time includes regret and average-strategy updates, but excludes exact evaluation.",
            "MCCFR regret traversals sample true chance/opponent distributions and enumerate traverser actions.",
            "Average strategies use exact own-reach updates over the precomputed information-set catalog.",
            "Strategy L1 distance is summed across information sets and can exceed 2.",
            "Memory is an approximate persistent Python table-size estimate, not process RSS.",
        ],
        "source_sha256": {
            name: _sha256(Path(name))
            for name in (
                "src/pokerlab/research/mccfr_experiments.py",
                "src/pokerlab/research/kuhn/mccfr.py",
                "src/pokerlab/research/kuhn/cfr.py",
                "src/pokerlab/research/kuhn/game.py",
                "src/pokerlab/research/kuhn/evaluation.py",
                "src/pokerlab/research/leduc/mccfr.py",
                "src/pokerlab/research/leduc/cfr.py",
                "src/pokerlab/research/leduc/game.py",
                "src/pokerlab/research/leduc/evaluation.py",
            )
        },
        "artifact_sha256": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "experiment_manifest.json":
            manifest["artifact_sha256"][path.name] = _sha256(path)
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", nargs="+", choices=("kuhn", "leduc"), default=("kuhn", "leduc")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(7, 19))
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--evaluation-interval", type=int, default=250)
    parser.add_argument("--output-root", type=Path, default=Path("results/mccfr"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = run_comparison(
        games=tuple(args.games),
        seeds=tuple(args.seeds),
        iterations=args.iterations,
        evaluation_interval=args.evaluation_interval,
        output_root=args.output_root,
        make_plots=not args.no_plots,
    )
    print(f"MCCFR comparison artifacts: {output}")


if __name__ == "__main__":
    main()
