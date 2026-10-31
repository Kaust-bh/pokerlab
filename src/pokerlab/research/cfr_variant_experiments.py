"""Reproducible modest comparisons of CFR, CFR+, DCFR, and ES-MCCFR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pokerlab.research.kuhn.cfr import CFRConfig as KuhnCFRConfig
from pokerlab.research.kuhn.cfr import CFRSolver as KuhnCFRSolver
from pokerlab.research.kuhn.cfr_plus import CFRPlusConfig as KuhnCFRPlusConfig
from pokerlab.research.kuhn.cfr_plus import CFRPlusSolver as KuhnCFRPlusSolver
from pokerlab.research.kuhn.dcfr import DCFRConfig as KuhnDCFRConfig
from pokerlab.research.kuhn.dcfr import DCFRSolver as KuhnDCFRSolver
from pokerlab.research.kuhn.evaluation import evaluate_nash as evaluate_kuhn
from pokerlab.research.kuhn.mccfr import MCCFRConfig as KuhnMCCFRConfig
from pokerlab.research.kuhn.mccfr import MCCFRSolver as KuhnMCCFRSolver
from pokerlab.research.kuhn.strategy import save_strategy_snapshot as save_kuhn
from pokerlab.research.leduc.cfr import CFRConfig as LeducCFRConfig
from pokerlab.research.leduc.cfr import CFRSolver as LeducCFRSolver
from pokerlab.research.leduc.cfr_plus import CFRPlusConfig as LeducCFRPlusConfig
from pokerlab.research.leduc.cfr_plus import CFRPlusSolver as LeducCFRPlusSolver
from pokerlab.research.leduc.dcfr import DCFRConfig as LeducDCFRConfig
from pokerlab.research.leduc.dcfr import DCFRSolver as LeducDCFRSolver
from pokerlab.research.leduc.evaluation import evaluate_nash as evaluate_leduc
from pokerlab.research.leduc.mccfr import MCCFRConfig as LeducMCCFRConfig
from pokerlab.research.leduc.mccfr import MCCFRSolver as LeducMCCFRSolver
from pokerlab.research.leduc.strategy import save_strategy_snapshot as save_leduc

ALGORITHMS = ("vanilla_cfr", "cfr_plus", "dcfr", "es_mccfr")


def _git_metadata() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], text=True
    ).splitlines()
    return {"head": head, "dirty": bool(status), "status_porcelain": status}


def _solver_factory(game: str, algorithm: str, iterations: int, seed: int):
    if game == "kuhn":
        config_types = {
            "vanilla_cfr": KuhnCFRConfig,
            "cfr_plus": KuhnCFRPlusConfig,
            "dcfr": KuhnDCFRConfig,
            "es_mccfr": KuhnMCCFRConfig,
        }
        solvers = {
            "vanilla_cfr": KuhnCFRSolver,
            "cfr_plus": KuhnCFRPlusSolver,
            "dcfr": KuhnDCFRSolver,
            "es_mccfr": KuhnMCCFRSolver,
        }
        config = config_types[algorithm](iterations=iterations, seed=seed)
    else:
        config_types = {
            "vanilla_cfr": LeducCFRConfig,
            "cfr_plus": LeducCFRPlusConfig,
            "dcfr": LeducDCFRConfig,
            "es_mccfr": LeducMCCFRConfig,
        }
        solvers = {
            "vanilla_cfr": LeducCFRSolver,
            "cfr_plus": LeducCFRPlusSolver,
            "dcfr": LeducDCFRSolver,
            "es_mccfr": LeducMCCFRSolver,
        }
        config = config_types[algorithm](iterations=iterations, seed=seed)
    return solvers[algorithm](config), config


def _operations_per_iteration(solver, algorithm: str) -> tuple[int, int]:
    traversals = getattr(solver, "traversals", None)
    updates = getattr(solver, "player_updates", None)
    if traversals is None:
        traversals = (
            2 * solver.iterations if algorithm == "es_mccfr" else solver.iterations
        )
    if updates is None:
        updates = 2 * solver.iterations
    return traversals, updates


def _run_game(
    game: str,
    directory: Path,
    iterations: int,
    seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
) -> list[dict]:
    evaluator = evaluate_kuhn if game == "kuhn" else evaluate_leduc
    saver = save_kuhn if game == "kuhn" else save_leduc
    records: list[dict] = []
    configs = []
    for algorithm in ALGORITHMS:
        for seed in seeds:
            solver, config = _solver_factory(game, algorithm, iterations, seed)
            configs.append(
                {"game": game, "algorithm": algorithm, "seed": seed, **config.__dict__}
                if hasattr(config, "__dict__")
                else {
                    "game": game,
                    "algorithm": algorithm,
                    "seed": seed,
                    **{
                        field: getattr(config, field)
                        for field in config.__dataclass_fields__
                    },
                }
            )
            training_elapsed = 0.0
            evaluation_elapsed = 0.0
            for iteration in range(1, iterations + 1):
                import time

                started = time.perf_counter()
                solver.run_iteration()
                training_elapsed += time.perf_counter() - started
                if iteration not in checkpoints:
                    continue
                evaluation_started = time.perf_counter()
                evaluation = evaluator(solver.average_strategy_profile())
                elapsed_eval = time.perf_counter() - evaluation_started
                evaluation_elapsed += elapsed_eval
                traversals, updates = _operations_per_iteration(solver, algorithm)
                records.append(
                    {
                        "game": game,
                        "algorithm": algorithm,
                        "seed": seed,
                        "iteration": iteration,
                        "expected_game_value_player0": evaluation.expected_value_player0,
                        "expected_game_value_player1": -evaluation.expected_value_player0,
                        "nashconv": evaluation.nashconv,
                        "exploitability": evaluation.exploitability,
                        "training_elapsed_seconds": training_elapsed,
                        "evaluation_elapsed_seconds": elapsed_eval,
                        "cumulative_evaluation_seconds": evaluation_elapsed,
                        "iterations_per_second": iteration / training_elapsed,
                        "traversals": traversals,
                        "player_updates": updates,
                        "information_sets": len(solver._legal),
                        "configuration": json.dumps(
                            {
                                field: getattr(config, field)
                                for field in config.__dataclass_fields__
                            },
                            sort_keys=True,
                        ),
                    }
                )
            saver(
                solver.average_strategy_profile(),
                directory / f"strategy_{game}_{algorithm}_seed_{seed}.json",
            )
    (directory / f"{game}_configurations.json").write_text(
        json.dumps(configs, indent=2) + "\n", encoding="utf-8"
    )
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot export empty experiment rows")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = row["game"], row["algorithm"], row["iteration"]
        groups.setdefault(key, []).append(row)
    output = []
    for (game, algorithm, iteration), group in sorted(groups.items()):
        for metric in (
            "exploitability",
            "expected_game_value_player0",
            "training_elapsed_seconds",
        ):
            values = [float(row[metric]) for row in group]
            output.append(
                {
                    "game": game,
                    "algorithm": algorithm,
                    "iteration": iteration,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": statistics.mean(values),
                    "sample_stddev": statistics.stdev(values)
                    if len(values) > 1
                    else 0.0,
                }
            )
    return output


def _plot(rows: list[dict], path: Path, x_field: str, x_title: str) -> None:
    import plotly.graph_objects as go

    figure = go.Figure()
    games = sorted({row["game"] for row in rows})
    for game in games:
        for algorithm in ALGORITHMS:
            selected = [
                row
                for row in rows
                if row["game"] == game and row["algorithm"] == algorithm
            ]
            grouped = {}
            for row in selected:
                grouped.setdefault(row["iteration"], []).append(row)
            x_values = []
            y_values = []
            for _, group in sorted(grouped.items()):
                x_values.append(statistics.mean(float(row[x_field]) for row in group))
                y_values.append(
                    statistics.mean(float(row["exploitability"]) for row in group)
                )
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    name=f"{game} {algorithm}",
                )
            )
    figure.update_layout(
        title=f"CFR variant comparison: exploitability vs {x_title}",
        xaxis_title=x_title,
        yaxis_title="Exact exploitability",
        yaxis_type="log",
    )
    figure.write_html(path, include_plotlyjs="cdn")


def run_experiment(
    *,
    output_root: str | Path = "results/cfr_variants",
    iterations: int = 20,
    seeds: tuple[int, ...] = (11, 23, 47),
) -> Path:
    if iterations < 1 or not seeds:
        raise ValueError("iterations and seeds must be nonempty and positive")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / ("run-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(exist_ok=False)
    checkpoints = tuple(sorted({max(1, iterations // 4), iterations // 2, iterations}))
    checkpoints = tuple(sorted(set(checkpoints)))
    rows = []
    for game in ("kuhn", "leduc"):
        rows.extend(_run_game(game, run_dir, iterations, seeds, checkpoints))
    _write_csv(run_dir / "convergence.csv", rows)
    _write_csv(run_dir / "comparison_aggregated.csv", _aggregate(rows))
    _plot(rows, run_dir / "exploitability_vs_iterations.html", "iteration", "iteration")
    _plot(
        rows,
        run_dir / "exploitability_vs_training_time.html",
        "training_elapsed_seconds",
        "cumulative training seconds",
    )
    config = {
        "iterations": iterations,
        "seeds": list(seeds),
        "checkpoints": list(checkpoints),
        "algorithms": list(ALGORITHMS),
        "games": ["kuhn", "leduc"],
        "command": (
            "PYTHONPATH=src python -m pokerlab.research.cfr_variant_experiments "
            f"--iterations {iterations} --seeds " + " ".join(map(str, seeds))
        ),
        "dcfr_defaults": {"alpha": 1.5, "beta": 0.0, "gamma": 2.0},
    }
    (run_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    metadata = _git_metadata()
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    project_root = Path(__file__).resolve().parents[3]
    source_files = sorted((project_root / "src/pokerlab/research").rglob("*.py"))
    test_files = sorted((project_root / "tests/research").rglob("*.py"))
    source_hashes = {
        path.relative_to(project_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_files + test_files
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                **metadata,
                "source_and_test_files_sha256": source_hashes,
                "files_sha256_before_manifest": hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/cfr_variants")
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    args = parser.parse_args()
    run_dir = run_experiment(
        output_root=args.output_root,
        iterations=args.iterations,
        seeds=tuple(args.seeds),
    )
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
