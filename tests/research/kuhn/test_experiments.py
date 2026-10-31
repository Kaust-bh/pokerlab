import csv

from pokerlab.research.kuhn.cfr import CFRConfig
from pokerlab.research.kuhn.experiments import run_experiment
from pokerlab.research.kuhn.strategy import load_strategy_snapshot


def test_experiment_exports_metrics_average_strategy_and_real_plot(tmp_path) -> None:
    artifacts = run_experiment(
        CFRConfig(
            iterations=200,
            seed=42,
            evaluation_interval=50,
            logging_interval=100,
        ),
        output_dir=tmp_path,
        make_plot=True,
    )
    assert artifacts.convergence_csv.exists()
    assert artifacts.strategy_json.exists()
    assert (
        artifacts.convergence_plot is not None and artifacts.convergence_plot.exists()
    )
    with artifacts.convergence_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["iteration"]) for row in rows] == [50, 100, 150, 200]
    assert load_strategy_snapshot(artifacts.strategy_json) == artifacts.average_strategy
    assert len(artifacts.average_strategy) == 12
