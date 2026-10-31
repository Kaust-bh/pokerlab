"""Small paired multi-seed Deep CFR configuration studies."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from pokerlab.algorithms.deep_cfr.config import DeepCFRConfig
from pokerlab.algorithms.deep_cfr.experiments import _hash, _write_csv

_T_975 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
    21: 2.086,
    22: 2.080,
    23: 2.074,
    24: 2.069,
    25: 2.064,
    26: 2.060,
    27: 2.056,
    28: 2.052,
    29: 2.048,
    30: 2.045,
}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _confidence_summary(values: list[float]) -> dict[str, float | int | None]:
    count = len(values)
    average = statistics.mean(values)
    sd = statistics.stdev(values) if count > 1 else None
    se = sd / count**0.5 if sd is not None else None
    critical = _T_975.get(count, 1.96 if count > 30 else None)
    half_width = critical * se if critical is not None and se is not None else None
    return {
        "n": count,
        "mean": average,
        "median": statistics.median(values),
        "standard_deviation": sd,
        "standard_error": se,
        "ci95_low": average - half_width if half_width is not None else None,
        "ci95_high": average + half_width if half_width is not None else None,
    }


def _fresh_study_dir(root: Path, game: str, identity: dict) -> Path:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"study-{game}-{timestamp}-{digest}"
    suffix = 1
    while path.exists():
        path = root / f"study-{game}-{timestamp}-{digest}-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def run_study(
    game: str,
    base_config: DeepCFRConfig,
    *,
    seeds: tuple[int, ...],
    variants: dict[str, dict],
    output_root: str | Path = "results/deep_cfr",
) -> Path:
    """Run paired seeds for variants that each change explicit config fields."""
    if game not in {"kuhn", "leduc"}:
        raise ValueError("game must be 'kuhn' or 'leduc'")
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty sequence of unique values")
    if not variants or any(not label for label in variants):
        raise ValueError("at least one named configuration variant is required")
    valid_fields = set(asdict(base_config)) - {"seed"}
    for params in variants.values():
        unknown = set(params) - valid_fields
        if unknown:
            raise ValueError(f"unknown Deep CFR configuration fields: {unknown}")

    identity = {
        "game": game,
        "base_config": asdict(base_config),
        "seeds": list(seeds),
        "variants": variants,
        "variant_configurations": {
            label: asdict(replace(base_config, **params))
            for label, params in variants.items()
        },
        "statistic": "mean per-information-set total variation; unweighted",
    }
    study_dir = _fresh_study_dir(Path(output_root).resolve(), game, identity)
    (study_dir / "runs").mkdir()
    (study_dir / "plots").mkdir()
    (study_dir / "worker-configs").mkdir()
    (study_dir / "study_config.json").write_text(json.dumps(identity, indent=2) + "\n")

    run_index = []
    raw_evaluations = []
    raw_training = []
    for label, overrides in variants.items():
        for seed in seeds:
            config = replace(base_config, seed=seed, **overrides)
            config_file = study_dir / "worker-configs" / f"{label}-seed-{seed}.json"
            config_file.write_text(json.dumps(asdict(config), indent=2) + "\n")
            child_env = os.environ.copy()
            source_root = str(Path("src").resolve())
            child_env["PYTHONPATH"] = os.pathsep.join(
                part for part in (source_root, child_env.get("PYTHONPATH", "")) if part
            )
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pokerlab.algorithms.deep_cfr.experiments",
                    "--game",
                    game,
                    "--config-file",
                    str(config_file),
                    "--output-root",
                    str(study_dir / "runs" / label),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=child_env,
            )
            run_dir = Path(child.stdout.strip().splitlines()[-1])
            if not run_dir.is_absolute():
                run_dir = Path.cwd() / run_dir
            run_summary = json.loads((run_dir / "summary.json").read_text())
            run_index.append(
                {
                    "variant": label,
                    "seed": seed,
                    "run_dir": str(run_dir.relative_to(study_dir)),
                    "process_peak_rss_bytes": run_summary["process_peak_rss_bytes"],
                    "deep_cfr_training_seconds": run_summary[
                        "deep_cfr_training_seconds"
                    ],
                    "vanilla_cfr_training_seconds": run_summary[
                        "vanilla_cfr_training_seconds"
                    ],
                    "cfr_plus_training_seconds": run_summary[
                        "cfr_plus_training_seconds"
                    ],
                    "experiment_wall_seconds": run_summary["experiment_wall_seconds"],
                }
            )
            with (run_dir / "evaluation.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    normalized = dict(row)
                    normalized["variant"] = label
                    normalized["run_dir"] = str(run_dir.relative_to(study_dir))
                    normalized["process_peak_rss_bytes"] = run_summary[
                        "process_peak_rss_bytes"
                    ]
                    raw_evaluations.append(normalized)
            with (run_dir / "training.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    normalized = dict(row)
                    normalized["variant"] = label
                    normalized["seed"] = seed
                    normalized["run_dir"] = str(run_dir.relative_to(study_dir))
                    raw_training.append(normalized)

    _write_csv(study_dir / "run_index.csv", run_index)
    _write_csv(study_dir / "evaluation.csv", raw_evaluations)
    _write_csv(study_dir / "training.csv", raw_training)
    training_groups: dict[tuple[str, int], list[dict]] = {}
    for row in raw_training:
        training_groups.setdefault((row["variant"], int(row["iteration"])), []).append(
            row
        )
    training_summary = []
    training_metrics = (
        "player0_advantage_loss",
        "player1_advantage_loss",
        "strategy_loss",
        "sampled_nodes",
        "runtime_seconds",
        "iterations_per_second",
        "strategy_memory_size",
        "strategy_memory_seen",
        "strategy_memory_replacements",
        "traversal_count",
    )
    for (variant, iteration), rows in sorted(training_groups.items()):
        values_by_metric = {metric: [] for metric in training_metrics}
        for row in rows:
            for metric in training_metrics:
                if row.get(metric) not in (None, ""):
                    values_by_metric[metric].append(float(row[metric]))
            for target, source, index in (
                ("advantage_memory_player0", "advantage_memory_sizes", 0),
                ("advantage_memory_player1", "advantage_memory_sizes", 1),
                ("advantage_memory_seen_player0", "advantage_memory_seen", 0),
                ("advantage_memory_seen_player1", "advantage_memory_seen", 1),
                (
                    "advantage_memory_replacements_player0",
                    "advantage_memory_replacements",
                    0,
                ),
                (
                    "advantage_memory_replacements_player1",
                    "advantage_memory_replacements",
                    1,
                ),
            ):
                values = ast.literal_eval(row[source])
                values_by_metric.setdefault(target, []).append(float(values[index]))
        output = {"variant": variant, "iteration": iteration, "n": len(rows)}
        for metric, values in values_by_metric.items():
            stats = _confidence_summary(values)
            for statistic in (
                "mean",
                "median",
                "standard_deviation",
                "standard_error",
                "ci95_low",
                "ci95_high",
            ):
                output[f"{metric}_{statistic}"] = stats[statistic]
        training_summary.append(output)
    _write_csv(study_dir / "training_summary.csv", training_summary)
    metrics = (
        "exploitability",
        "expected_value_player0",
        "training_seconds",
        "evaluation_seconds",
        "process_peak_rss_bytes",
        "strategy_tv_from_vanilla_cfr",
        "strategy_tv_from_cfr_plus",
        "strategy_tv_from_dcfr",
        "advantage_loss_player0",
        "advantage_loss_player1",
        "strategy_loss",
    )
    grouped: dict[tuple, list[dict]] = {}
    for row in raw_evaluations:
        key = (row["variant"], row["algorithm"], int(row["iteration"]))
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (variant, algorithm, iteration), rows in sorted(grouped.items()):
        output = {"variant": variant, "algorithm": algorithm, "iteration": iteration}
        variant_config = identity["variant_configurations"][variant]
        output["advantage_memory_capacity"] = variant_config[
            "advantage_memory_capacity"
        ]
        output["strategy_memory_capacity"] = variant_config["strategy_memory_capacity"]
        for metric in metrics:
            values = [
                float(row[metric]) for row in rows if row.get(metric) not in (None, "")
            ]
            summary = _confidence_summary(values) if values else None
            for statistic in (
                "n",
                "mean",
                "median",
                "standard_deviation",
                "standard_error",
                "ci95_low",
                "ci95_high",
            ):
                output[f"{metric}_{statistic}"] = (
                    summary[statistic] if summary is not None else None
                )
        summary_rows.append(output)
    _write_csv(study_dir / "summary.csv", summary_rows)

    paired_rows = []
    paired: dict[tuple, dict[str, dict]] = {}
    for row in raw_evaluations:
        if row["algorithm"] == "deep_cfr":
            paired.setdefault(
                (row["variant"], int(row["seed"]), int(row["iteration"])), {}
            )["deep_cfr"] = row
        elif row["algorithm"] in {"vanilla_cfr", "cfr_plus", "dcfr"}:
            paired.setdefault(
                (row["variant"], int(row["seed"]), int(row["iteration"])), {}
            )[row["algorithm"]] = row
    paired_groups: dict[tuple, list[dict]] = {}
    for (variant, seed, iteration), rows in paired.items():
        if not {"deep_cfr", "vanilla_cfr", "cfr_plus", "dcfr"}.issubset(rows):
            continue
        output = {"variant": variant, "seed": seed, "iteration": iteration}
        for baseline in ("vanilla_cfr", "cfr_plus", "dcfr"):
            key = f"deep_minus_{baseline}_exploitability"
            output[key] = float(rows["deep_cfr"]["exploitability"]) - float(
                rows[baseline]["exploitability"]
            )
            paired_groups.setdefault((variant, iteration, key), []).append(output)
        paired_rows.append(output)
    _write_csv(study_dir / "paired_comparisons.csv", paired_rows)

    paired_summary = []
    for (variant, iteration, metric), rows in sorted(paired_groups.items()):
        summary = _confidence_summary([float(row[metric]) for row in rows])
        paired_summary.append(
            {"variant": variant, "iteration": iteration, "metric": metric, **summary}
        )
    _write_csv(study_dir / "paired_summary.csv", paired_summary)

    deep_by_seed_checkpoint = {
        (row["variant"], int(row["seed"]), int(row["iteration"])): row
        for row in raw_evaluations
        if row["algorithm"] == "deep_cfr"
    }
    config_deltas = []
    config_delta_groups: dict[tuple, list[dict]] = {}
    for (variant, seed, iteration), row in deep_by_seed_checkpoint.items():
        baseline = deep_by_seed_checkpoint.get(("baseline", seed, iteration))
        if baseline is None or variant == "baseline":
            continue
        for metric in (
            "exploitability",
            "training_seconds",
            "strategy_tv_from_vanilla_cfr",
            "strategy_tv_from_cfr_plus",
            "strategy_tv_from_dcfr",
        ):
            difference = float(row[metric]) - float(baseline[metric])
            output = {
                "variant": variant,
                "seed": seed,
                "iteration": iteration,
                "metric": metric,
                "variant_minus_baseline": difference,
            }
            config_deltas.append(output)
            config_delta_groups.setdefault((variant, iteration, metric), []).append(
                output
            )
    _write_csv(study_dir / "paired_config_comparisons.csv", config_deltas)
    config_delta_summary = []
    for (variant, iteration, metric), rows in sorted(config_delta_groups.items()):
        stats = _confidence_summary(
            [float(row["variant_minus_baseline"]) for row in rows]
        )
        config_delta_summary.append(
            {"variant": variant, "iteration": iteration, "metric": metric, **stats}
        )
    _write_csv(study_dir / "paired_config_summary.csv", config_delta_summary)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        figure = go.Figure()
        for variant in variants:
            selected = [
                row
                for row in summary_rows
                if row["variant"] == variant and row["algorithm"] == "deep_cfr"
            ]
            figure.add_trace(
                go.Scatter(
                    x=[row["iteration"] for row in selected],
                    y=[row["exploitability_mean"] for row in selected],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [
                            row["exploitability_ci95_high"] - row["exploitability_mean"]
                            if row["exploitability_ci95_high"] is not None
                            else 0
                            for row in selected
                        ],
                        "arrayminus": [
                            row["exploitability_mean"] - row["exploitability_ci95_low"]
                            if row["exploitability_ci95_low"] is not None
                            else 0
                            for row in selected
                        ],
                    },
                    mode="lines+markers",
                    name=variant,
                )
            )
        figure.update_layout(
            title=f"{game.title()} Deep CFR configuration study",
            xaxis_title="Iterations",
            yaxis_title="Exact exploitability (mean, 95% t interval)",
        )
        figure.write_html(study_dir / "plots" / "exploitability_by_iteration.html")

        time_figure = go.Figure()
        for variant in variants:
            selected = [
                row
                for row in summary_rows
                if row["variant"] == variant and row["algorithm"] == "deep_cfr"
            ]
            time_figure.add_trace(
                go.Scatter(
                    x=[row["training_seconds_mean"] for row in selected],
                    y=[row["exploitability_mean"] for row in selected],
                    mode="lines+markers",
                    name=variant,
                )
            )
        time_figure.update_layout(
            title=f"{game.title()} Deep CFR quality against training time",
            xaxis_title="Training seconds (mean across seeds)",
            yaxis_title="Exact exploitability (mean)",
        )
        time_figure.write_html(study_dir / "plots" / "exploitability_by_time.html")

        for horizontal_metric, filename, axis_title in (
            (
                "training_seconds",
                "algorithm_comparison_by_time.html",
                "Training seconds",
            ),
            ("iteration", "algorithm_comparison_by_iteration.html", "Iterations"),
        ):
            comparison = go.Figure()
            for algorithm in ("deep_cfr", "vanilla_cfr", "cfr_plus", "dcfr"):
                selected = sorted(
                    (
                        row
                        for row in summary_rows
                        if row["variant"] == "baseline"
                        and row["algorithm"] == algorithm
                    ),
                    key=lambda row: int(row["iteration"]),
                )
                x_metric = (
                    f"{horizontal_metric}_mean"
                    if horizontal_metric != "iteration"
                    else "iteration"
                )
                comparison.add_trace(
                    go.Scatter(
                        x=[row[x_metric] for row in selected],
                        y=[row["exploitability_mean"] for row in selected],
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": [
                                row["exploitability_ci95_high"]
                                - row["exploitability_mean"]
                                if row["exploitability_ci95_high"] is not None
                                else 0
                                for row in selected
                            ],
                            "arrayminus": [
                                row["exploitability_mean"]
                                - row["exploitability_ci95_low"]
                                if row["exploitability_ci95_low"] is not None
                                else 0
                                for row in selected
                            ],
                        },
                        mode="lines+markers",
                        name=algorithm,
                    )
                )
            comparison.update_layout(
                title=f"{game.title()} baseline algorithm comparison",
                xaxis_title=axis_title,
                yaxis_title="Exact exploitability (mean, 95% t interval)",
            )
            comparison.write_html(study_dir / "plots" / filename)

        loss_figure = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Player 0 advantage loss",
                "Player 1 advantage loss",
                "Strategy loss",
            ),
        )
        for variant in variants:
            selected = sorted(
                (row for row in raw_training if row["variant"] == variant),
                key=lambda row: (int(row["iteration"]), int(row["seed"])),
            )
            by_iteration: dict[int, list[dict]] = {}
            for row in selected:
                by_iteration.setdefault(int(row["iteration"]), []).append(row)
            for col, metric in enumerate(
                ("player0_advantage_loss", "player1_advantage_loss", "strategy_loss"),
                start=1,
            ):
                loss_figure.add_trace(
                    go.Scatter(
                        x=sorted(by_iteration),
                        y=[
                            statistics.mean(
                                float(row[metric]) for row in by_iteration[iteration]
                            )
                            for iteration in sorted(by_iteration)
                        ],
                        mode="lines",
                        name=variant,
                        legendgroup=variant,
                        showlegend=col == 1,
                    ),
                    row=1,
                    col=col,
                )
        loss_figure.update_layout(title=f"{game.title()} training losses")
        loss_figure.write_html(study_dir / "plots" / "training_losses.html")

        capacity_figure = go.Figure()
        for variant in variants:
            selected = [
                row
                for row in raw_evaluations
                if row["variant"] == variant
                and row["algorithm"] == "deep_cfr"
                and int(row["iteration"]) == base_config.iterations
            ]
            capacity_figure.add_trace(
                go.Scatter(
                    x=[
                        identity["variant_configurations"][variant][
                            "advantage_memory_capacity"
                        ]
                    ]
                    * len(selected),
                    y=[float(row["training_seconds"]) for row in selected],
                    mode="markers",
                    name=variant,
                )
            )
        capacity_figure.update_layout(
            title=f"{game.title()} runtime against reservoir capacity",
            xaxis_title="Advantage/strategy reservoir capacity",
            yaxis_title="Deep CFR cumulative training seconds",
        )
        capacity_figure.write_html(study_dir / "plots" / "runtime_by_capacity.html")
    except ImportError:
        pass

    (study_dir / "summary.json").write_text(
        json.dumps(
            {
                "game": game,
                "seed_count": len(seeds),
                "seeds": list(seeds),
                "run_count": len(run_index),
                "runs": run_index,
                "ci_method": "two-sided Student t interval; n<2 has no interval",
                "strategy_distance": "unweighted mean per-information-set TV",
            },
            indent=2,
        )
        + "\n"
    )
    artifacts = sorted(path for path in study_dir.rglob("*") if path.is_file())
    source_files = sorted(Path("src/pokerlab/algorithms/deep_cfr").rglob("*.py"))
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "git_status_porcelain": subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "platform": platform.platform(),
        "python": sys.version,
        "source_sha256": {str(path): _hash(path) for path in source_files},
        "artifact_sha256": {
            str(path.relative_to(study_dir)): _hash(path)
            for path in artifacts
            if path.name != "manifest.json"
        },
    }
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return study_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("kuhn", "leduc"), required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 19, 2026))
    parser.add_argument("--output-root", default="results/deep_cfr")
    parser.add_argument("--traversals", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--memory-capacity", type=int, default=20_000)
    args = parser.parse_args()
    config = DeepCFRConfig(
        iterations=args.iterations,
        evaluation_interval=args.evaluation_interval,
        traversals_per_player=args.traversals,
        training_epochs=args.epochs,
        advantage_memory_capacity=args.memory_capacity,
        strategy_memory_capacity=args.memory_capacity,
    )
    variants = {
        "baseline": {},
        "more_traversals": {"traversals_per_player": args.traversals * 4},
        "more_epochs": {"training_epochs": args.epochs * 4},
        "more_strategy_epochs": {"strategy_training_epochs": args.epochs * 4},
        "smaller_reservoirs": {
            "advantage_memory_capacity": max(32, args.memory_capacity // 16),
            "strategy_memory_capacity": max(32, args.memory_capacity // 16),
        },
    }
    print(
        run_study(
            args.game,
            config,
            seeds=tuple(args.seeds),
            variants=variants,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
