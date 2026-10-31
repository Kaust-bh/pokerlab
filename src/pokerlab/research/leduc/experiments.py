"""Reproducible Leduc training runs with CSV, JSON, and convergence plots."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from pokerlab.research.kuhn.cfr import CFRConfig as KuhnCFRConfig
from pokerlab.research.kuhn.cfr import CFRSolver as KuhnCFRSolver
from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cfr import CFRConfig, CFRSolver, TrainingRecord
from pokerlab.research.leduc.information_sets import InformationSet
from pokerlab.research.leduc.strategy import save_strategy_snapshot


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    records: tuple[TrainingRecord, ...]
    average_strategy: Mapping[InformationSet, Mapping[Action, float]]
    convergence_csv: Path
    strategy_json: Path
    convergence_plot: Path | None
    comparison_csv: Path | None = None


def export_training_records(
    records: tuple[TrainingRecord, ...], path: str | Path
) -> Path:
    if not records:
        raise ValueError("cannot export an empty training history")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return target


def plot_convergence(records: tuple[TrainingRecord, ...], path: str | Path) -> Path:
    if not records:
        raise ValueError("cannot plot an empty training history")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Exploitability", "NashConv", "Expected Player 0 value"),
        vertical_spacing=0.12,
    )
    x = [record.iteration for record in records]
    for row, values, label in (
        (1, [r.exploitability for r in records], "Exploitability"),
        (2, [r.nashconv for r in records], "NashConv"),
        (3, [r.expected_game_value for r in records], "Expected value"),
    ):
        figure.add_trace(
            go.Scatter(x=x, y=values, mode="lines+markers", name=label), row=row, col=1
        )
    for row in (1, 2, 3):
        figure.update_xaxes(title_text="CFR iteration", row=row, col=1)
    figure.update_yaxes(title_text="Chips per hand", row=3, col=1)
    figure.update_layout(title="Leduc Poker vanilla CFR convergence", height=900)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(target, include_plotlyjs="cdn")
    return target


def export_kuhn_comparison(
    leduc_record: TrainingRecord,
    *,
    iterations: int,
    seed: int,
    evaluation_interval: int,
    logging_interval: int,
    path: str | Path,
) -> Path:
    """Train Kuhn at the same iteration/seed settings and save an apples-to-context table."""
    from pokerlab.research.kuhn.game import KuhnGame, KuhnState

    kuhn = KuhnCFRSolver(
        KuhnCFRConfig(iterations, seed, evaluation_interval, logging_interval)
    )
    kuhn_records = kuhn.train()
    if not kuhn_records:
        raise RuntimeError("Kuhn comparison produced no evaluation record")
    record = kuhn_records[-1]

    def node_count(state: KuhnState) -> int:
        if state.is_terminal:
            return 1
        return 1 + sum(node_count(state.act(action)) for action in state.legal_actions)

    kuhn_nodes = sum(
        node_count(KuhnGame.initial_state(deal)) for deal in KuhnGame.deals()
    )
    rows = [
        {
            "game": "Leduc",
            "iterations": leduc_record.iteration,
            "seed": seed,
            "expected_game_value": leduc_record.expected_game_value,
            "nashconv": leduc_record.nashconv,
            "exploitability": leduc_record.exploitability,
            "information_sets": leduc_record.information_sets,
            "nodes_per_iteration": leduc_record.game_tree_nodes,
            "iterations_per_second": leduc_record.iterations_per_second,
            "runtime_seconds": leduc_record.runtime_seconds,
        },
        {
            "game": "Kuhn",
            "iterations": record.iteration,
            "seed": seed,
            "expected_game_value": record.expected_game_value,
            "nashconv": record.nashconv,
            "exploitability": record.exploitability,
            "information_sets": record.information_sets,
            "nodes_per_iteration": kuhn_nodes,
            "iterations_per_second": record.iteration / record.runtime_seconds,
            "runtime_seconds": record.runtime_seconds,
        },
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def run_experiment(
    config: CFRConfig | None = None,
    *,
    output_dir: str | Path = "results/leduc",
    make_plot: bool = True,
    compare_kuhn: bool = False,
) -> ExperimentArtifacts:
    solver = CFRSolver(config or CFRConfig())

    def report(record: TrainingRecord) -> None:
        print(
            f"iteration={record.iteration} value={record.expected_game_value:.6f} "
            f"exploitability={record.exploitability:.6f} nashconv={record.nashconv:.6f} "
            f"infosets={record.information_sets} nodes={record.game_tree_nodes} "
            f"iter/s={record.iterations_per_second:.3f} br_s={record.best_response_seconds:.3f}"
        )

    records = solver.train(callback=report)
    directory = Path(output_dir)
    strategy = solver.average_strategy_profile()
    csv_path = export_training_records(records, directory / "convergence.csv")
    strategy_path = save_strategy_snapshot(
        strategy, directory / "average_strategy.json"
    )
    plot_path = (
        plot_convergence(records, directory / "convergence.html") if make_plot else None
    )
    comparison_path = (
        export_kuhn_comparison(
            records[-1],
            iterations=solver.config.iterations,
            seed=solver.config.seed,
            evaluation_interval=solver.config.evaluation_interval,
            logging_interval=solver.config.logging_interval,
            path=directory / "kuhn_comparison.csv",
        )
        if compare_kuhn
        else None
    )
    return ExperimentArtifacts(
        records, strategy, csv_path, strategy_path, plot_path, comparison_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-interval", type=int, default=1_000)
    parser.add_argument("--logging-interval", type=int, default=1_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/leduc"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--compare-kuhn",
        action="store_true",
        help="run Kuhn CFR at the same iteration count and seed; save a comparison CSV",
    )
    args = parser.parse_args()
    result = run_experiment(
        CFRConfig(
            args.iterations, args.seed, args.evaluation_interval, args.logging_interval
        ),
        output_dir=args.output_dir,
        make_plot=not args.no_plot,
        compare_kuhn=args.compare_kuhn,
    )
    print(f"metrics: {result.convergence_csv}")
    print(f"average strategy: {result.strategy_json}")
    if result.convergence_plot is not None:
        print(f"plot: {result.convergence_plot}")
    if result.comparison_csv is not None:
        print(f"comparison: {result.comparison_csv}")


if __name__ == "__main__":
    main()
