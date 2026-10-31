"""Reproducible Kuhn CFR experiments, CSV/JSON export, and convergence plots."""

import argparse
import csv
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from pokerlab.research.kuhn.cfr import CFRConfig, CFRSolver, TrainingRecord
from pokerlab.research.kuhn.game import KuhnAction
from pokerlab.research.kuhn.information_sets import InformationSet
from pokerlab.research.kuhn.strategy import save_strategy_snapshot


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    records: tuple[TrainingRecord, ...]
    average_strategy: Mapping[InformationSet, Mapping[KuhnAction, float]]
    convergence_csv: Path
    strategy_json: Path
    convergence_plot: Path | None


def export_training_records(
    records: tuple[TrainingRecord, ...] | list[TrainingRecord], path: str | Path
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


def plot_convergence(
    records: tuple[TrainingRecord, ...] | list[TrainingRecord], path: str | Path
) -> Path:
    """Write a Plotly HTML figure based only on supplied training records."""
    if not records:
        raise ValueError("cannot plot an empty training history")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Exploitability", "Expected game value for Player 0"),
        vertical_spacing=0.16,
    )
    iterations = [record.iteration for record in records]
    figure.add_trace(
        go.Scatter(
            x=iterations,
            y=[record.exploitability for record in records],
            mode="lines+markers",
            name="Exploitability",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=iterations,
            y=[record.expected_game_value for record in records],
            mode="lines+markers",
            name="Expected value",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=-1 / 18, line_dash="dash", row=2, col=1)
    figure.update_xaxes(title_text="CFR iteration", row=2, col=1)
    figure.update_yaxes(title_text="Chips per hand", row=2, col=1)
    figure.update_layout(title="Kuhn Poker vanilla CFR convergence", height=700)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(target, include_plotlyjs="cdn")
    return target


def run_experiment(
    config: CFRConfig | None = None,
    *,
    output_dir: str | Path = "results/kuhn",
    make_plot: bool = True,
) -> ExperimentArtifacts:
    solver = CFRSolver(config or CFRConfig())

    def report(record: TrainingRecord) -> None:
        print(
            f"iteration={record.iteration} value={record.expected_game_value:.6f} "
            f"exploitability={record.exploitability:.6f} "
            f"nashconv={record.nashconv:.6f}"
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
    return ExperimentArtifacts(records, strategy, csv_path, strategy_path, plot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--logging-interval", type=int, default=1_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/kuhn"))
    parser.add_argument("--no-plot", action="store_true")
    arguments = parser.parse_args()
    artifacts = run_experiment(
        CFRConfig(
            iterations=arguments.iterations,
            seed=arguments.seed,
            evaluation_interval=arguments.evaluation_interval,
            logging_interval=arguments.logging_interval,
        ),
        output_dir=arguments.output_dir,
        make_plot=not arguments.no_plot,
    )
    print(f"metrics: {artifacts.convergence_csv}")
    print(f"average strategy: {artifacts.strategy_json}")
    if artifacts.convergence_plot is not None:
        print(f"convergence plot: {artifacts.convergence_plot}")
    print("selected average strategy:")
    for info_set, distribution in sorted(
        artifacts.average_strategy.items(), key=lambda item: item[0].identifier
    ):
        probabilities = ", ".join(
            f"{action.value}={probability:.3f}"
            for action, probability in distribution.items()
        )
        print(f"  {info_set.identifier}: {probabilities}")


if __name__ == "__main__":
    main()
