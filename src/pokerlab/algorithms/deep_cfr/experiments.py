"""Small reproducible Deep CFR versus tabular CFR/CFR+ experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pokerlab.algorithms.deep_cfr import (
    DeepCFRConfig,
    DeepCFRTrainer,
    KuhnCodec,
    LeducCodec,
)
from pokerlab.algorithms.deep_cfr.metrics import mean_information_set_total_variation
from pokerlab.research.kuhn.cfr import CFRConfig as KuhnCFRConfig
from pokerlab.research.kuhn.cfr import CFRSolver as KuhnCFR
from pokerlab.research.kuhn.cfr_plus import CFRPlusConfig as KuhnCFRPlusConfig
from pokerlab.research.kuhn.cfr_plus import CFRPlusSolver as KuhnCFRPlus
from pokerlab.research.kuhn.dcfr import DCFRConfig as KuhnDCFRConfig
from pokerlab.research.kuhn.dcfr import DCFRSolver as KuhnDCFR
from pokerlab.research.kuhn.evaluation import evaluate_nash as evaluate_kuhn
from pokerlab.research.kuhn.strategy import save_strategy_snapshot as save_kuhn_strategy
from pokerlab.research.leduc.cfr import CFRConfig as LeducCFRConfig
from pokerlab.research.leduc.cfr import CFRSolver as LeducCFR
from pokerlab.research.leduc.cfr_plus import CFRPlusConfig as LeducCFRPlusConfig
from pokerlab.research.leduc.cfr_plus import CFRPlusSolver as LeducCFRPlus
from pokerlab.research.leduc.dcfr import DCFRConfig as LeducDCFRConfig
from pokerlab.research.leduc.dcfr import DCFRSolver as LeducDCFR
from pokerlab.research.leduc.evaluation import evaluate_nash as evaluate_leduc
from pokerlab.research.leduc.strategy import (
    save_strategy_snapshot as save_leduc_strategy,
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        if not rows:
            stream.write("\n")
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _new_run_dir(root: Path, game: str, digest: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"run-{game}-{timestamp}-{digest}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"run-{game}-{timestamp}-{digest}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def run_experiment(
    game: str,
    config: DeepCFRConfig,
    *,
    output_root: str | Path = "results/deep_cfr",
) -> Path:
    """Train Deep CFR, tabular vanilla CFR and CFR+ with exact evaluation."""
    if game not in {"kuhn", "leduc"}:
        raise ValueError("game must be 'kuhn' or 'leduc'")
    codec = KuhnCodec() if game == "kuhn" else LeducCodec()
    evaluator = evaluate_kuhn if game == "kuhn" else evaluate_leduc
    save_strategy = save_kuhn_strategy if game == "kuhn" else save_leduc_strategy
    deep = DeepCFRTrainer(codec, config)
    if game == "kuhn":
        vanilla = KuhnCFR(KuhnCFRConfig(config.iterations, config.seed))
        cfr_plus = KuhnCFRPlus(KuhnCFRPlusConfig(config.iterations, config.seed))
        dcfr = KuhnDCFR(KuhnDCFRConfig(config.iterations, config.seed))
    else:
        vanilla = LeducCFR(LeducCFRConfig(config.iterations, config.seed))
        cfr_plus = LeducCFRPlus(LeducCFRPlusConfig(config.iterations, config.seed))
        dcfr = LeducDCFR(LeducDCFRConfig(config.iterations, config.seed))

    checkpoint_points = tuple(
        sorted(
            {
                config.iterations,
                *range(
                    config.evaluation_interval,
                    config.iterations + 1,
                    config.evaluation_interval,
                ),
            }
        )
    )
    identity = {
        "game": game,
        "config": asdict(config),
        "comparison": ["Deep CFR", "vanilla CFR", "CFR+", "DCFR"],
        "checkpoints": checkpoint_points,
    }
    run_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    run_dir = _new_run_dir(Path(output_root), game, run_digest)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "strategies").mkdir()
    (run_dir / "logs").mkdir()
    (run_dir / "config.json").write_text(json.dumps(identity, indent=2) + "\n")
    start_time = time.perf_counter()
    rows: list[dict] = []
    vanilla_seconds = 0.0
    cfr_plus_seconds = 0.0
    dcfr_seconds = 0.0

    for iteration in range(1, config.iterations + 1):
        deep_record = deep.run_iteration()
        started = time.perf_counter()
        vanilla.run_iteration()
        vanilla_seconds += time.perf_counter() - started
        started = time.perf_counter()
        cfr_plus.run_iteration()
        cfr_plus_seconds += time.perf_counter() - started
        started = time.perf_counter()
        dcfr.run_iteration()
        dcfr_seconds += time.perf_counter() - started
        if iteration not in checkpoint_points:
            continue

        deep_profile = deep.policy(seed=config.seed).exact_strategy_profile()
        vanilla_profile = vanilla.average_strategy_profile()
        cfr_plus_profile = cfr_plus.average_strategy_profile()
        dcfr_profile = dcfr.average_strategy_profile()
        profiles = {
            "deep_cfr": deep_profile,
            "vanilla_cfr": vanilla_profile,
            "cfr_plus": cfr_plus_profile,
            "dcfr": dcfr_profile,
        }
        distances = {
            name: {
                "vanilla": mean_information_set_total_variation(
                    profile, vanilla_profile
                ),
                "cfr_plus": mean_information_set_total_variation(
                    profile, cfr_plus_profile
                ),
                "dcfr": mean_information_set_total_variation(profile, dcfr_profile),
            }
            for name, profile in profiles.items()
        }
        for name, profile, training_seconds, loss in (
            (
                "deep_cfr",
                deep_profile,
                sum(record.runtime_seconds for record in deep.records),
                (
                    deep_record.player0_advantage_loss,
                    deep_record.player1_advantage_loss,
                    deep_record.strategy_loss,
                ),
            ),
            ("vanilla_cfr", vanilla_profile, vanilla_seconds, (None, None, None)),
            ("cfr_plus", cfr_plus_profile, cfr_plus_seconds, (None, None, None)),
            ("dcfr", dcfr_profile, dcfr_seconds, (None, None, None)),
        ):
            evaluation_started = time.perf_counter()
            evaluation = evaluator(profile)
            evaluation_seconds = time.perf_counter() - evaluation_started
            rows.append(
                {
                    "game": game,
                    "algorithm": name,
                    "seed": config.seed,
                    "iteration": iteration,
                    "expected_value_player0": evaluation.expected_value_player0,
                    "nashconv": evaluation.nashconv,
                    "exploitability": evaluation.exploitability,
                    "strategy_tv_from_vanilla_cfr": distances[name]["vanilla"],
                    "strategy_tv_from_cfr_plus": distances[name]["cfr_plus"],
                    "strategy_tv_from_dcfr": distances[name]["dcfr"],
                    "training_seconds": training_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "iterations_per_training_second": (
                        iteration / training_seconds if training_seconds else 0.0
                    ),
                    "advantage_loss_player0": loss[0],
                    "advantage_loss_player1": loss[1],
                    "strategy_loss": loss[2],
                    "advantage_memory_player0": (
                        len(deep.advantage_memories[0]) if name == "deep_cfr" else None
                    ),
                    "advantage_memory_player1": (
                        len(deep.advantage_memories[1]) if name == "deep_cfr" else None
                    ),
                    "strategy_memory": len(deep.strategy_memory)
                    if name == "deep_cfr"
                    else None,
                    "advantage_memory_seen_player0": (
                        deep.advantage_memories[0].seen if name == "deep_cfr" else None
                    ),
                    "advantage_memory_seen_player1": (
                        deep.advantage_memories[1].seen if name == "deep_cfr" else None
                    ),
                    "strategy_memory_seen": (
                        deep.strategy_memory.seen if name == "deep_cfr" else None
                    ),
                    "advantage_memory_replacements_player0": (
                        deep.advantage_memories[0].replacements
                        if name == "deep_cfr"
                        else None
                    ),
                    "advantage_memory_replacements_player1": (
                        deep.advantage_memories[1].replacements
                        if name == "deep_cfr"
                        else None
                    ),
                    "strategy_memory_replacements": (
                        deep.strategy_memory.replacements
                        if name == "deep_cfr"
                        else None
                    ),
                    "traversals": deep.traversal_count if name == "deep_cfr" else None,
                    "sampled_nodes": deep_record.sampled_nodes
                    if name == "deep_cfr"
                    else None,
                }
            )
            save_strategy(
                profile, run_dir / "strategies" / f"{name}-iter-{iteration}.json"
            )
        deep.save_checkpoint(run_dir / "checkpoints" / f"deep-cfr-iter-{iteration}.pt")
        with (run_dir / "logs" / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "time_utc": datetime.now(UTC).isoformat(),
                        "iteration": iteration,
                        "advantage_memory": [
                            len(memory) for memory in deep.advantage_memories
                        ],
                        "strategy_memory": len(deep.strategy_memory),
                        "advantage_memory_replacements": [
                            memory.replacements for memory in deep.advantage_memories
                        ],
                        "strategy_memory_replacements": deep.strategy_memory.replacements,
                    }
                )
                + "\n"
            )

    training_rows = []
    for record in deep.records:
        item = asdict(record)
        for field in (
            "advantage_memory_sizes",
            "advantage_memory_seen",
            "advantage_memory_replacements",
        ):
            item[field] = json.dumps(item[field])
        training_rows.append(item)
    _write_csv(run_dir / "training.csv", training_rows)
    _write_csv(run_dir / "evaluation.csv", rows)
    _write_csv(run_dir / "convergence.csv", rows)
    try:
        import plotly.graph_objects as go

        figure = go.Figure()
        for algorithm in ("deep_cfr", "vanilla_cfr", "cfr_plus", "dcfr"):
            selected = [row for row in rows if row["algorithm"] == algorithm]
            figure.add_trace(
                go.Scatter(
                    x=[row["iteration"] for row in selected],
                    y=[row["exploitability"] for row in selected],
                    mode="lines+markers",
                    name=algorithm,
                )
            )
        figure.update_layout(
            title=f"{game.title()} Deep CFR and tabular CFR comparison",
            xaxis_title="Iterations",
            yaxis_title="Exact exploitability",
        )
        figure.write_html(run_dir / "exploitability.html", include_plotlyjs="cdn")
    except ImportError:
        pass

    summary = {
        "game": game,
        "seed": config.seed,
        "iterations": config.iterations,
        "traversals_per_player_per_iteration": config.traversals_per_player,
        "comparison_rows": len(rows),
        "experiment_wall_seconds": time.perf_counter() - start_time,
        "deep_cfr_training_seconds": sum(
            record.runtime_seconds for record in deep.records
        ),
        "vanilla_cfr_training_seconds": vanilla_seconds,
        "cfr_plus_training_seconds": cfr_plus_seconds,
        "dcfr_training_seconds": dcfr_seconds,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "final_exploitability": {
            row["algorithm"]: row["exploitability"]
            for row in rows
            if row["iteration"] == config.iterations
        },
        "source_revision": _git("rev-parse", "HEAD"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    source_files = sorted(Path("src/pokerlab").rglob("*.py"))
    artifact_files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "platform": platform.platform(),
        "python": sys.version,
        "source_sha256": {str(path): _hash(path) for path in source_files},
        "artifact_sha256": {
            str(path.relative_to(run_dir)): _hash(path) for path in artifact_files
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("kuhn", "leduc"), default="kuhn")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--traversals-per-player", type=int, default=4)
    parser.add_argument("--memory-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-epochs", type=int, default=1)
    parser.add_argument("--strategy-training-epochs", type=int)
    parser.add_argument("--strategy-batch-size", type=int)
    parser.add_argument("--strategy-learning-rate", type=float)
    parser.add_argument("--hidden-layers", default="64,64")
    parser.add_argument("--strategy-hidden-layers")
    parser.add_argument(
        "--activation", choices=("relu", "tanh", "leaky_relu"), default="relu"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weighting", choices=("linear", "uniform"), default="linear")
    parser.add_argument("--output-root", default="results/deep_cfr")
    parser.add_argument("--config-file")
    args = parser.parse_args()
    if args.config_file:
        config_data = json.loads(Path(args.config_file).read_text())
        for field in ("hidden_layers", "strategy_hidden_layers"):
            if config_data.get(field) is not None:
                config_data[field] = tuple(config_data[field])
        config = DeepCFRConfig(**config_data)
    else:
        config = DeepCFRConfig(
            iterations=args.iterations,
            seed=args.seed,
            traversals_per_player=args.traversals_per_player,
            advantage_memory_capacity=args.memory_capacity,
            strategy_memory_capacity=args.memory_capacity,
            batch_size=args.batch_size,
            training_epochs=args.training_epochs,
            hidden_layers=tuple(int(width) for width in args.hidden_layers.split(",")),
            strategy_hidden_layers=(
                tuple(int(width) for width in args.strategy_hidden_layers.split(","))
                if args.strategy_hidden_layers
                else None
            ),
            activation=args.activation,
            learning_rate=args.learning_rate,
            strategy_learning_rate=args.strategy_learning_rate,
            strategy_batch_size=args.strategy_batch_size,
            strategy_training_epochs=args.strategy_training_epochs,
            optimizer=args.optimizer,
            gradient_clip_norm=args.gradient_clip_norm,
            evaluation_interval=args.evaluation_interval,
            device=args.device,
            iteration_weighting=args.weighting,
        )
    print(run_experiment(args.game, config, output_root=args.output_root))


if __name__ == "__main__":
    main()
