import csv

from pokerlab.research.leduc.cfr import TrainingRecord
from pokerlab.research.leduc.experiments import (
    export_kuhn_comparison,
    export_training_records,
)


def test_export_records_and_same_budget_kuhn_comparison(tmp_path):
    leduc_record = TrainingRecord(
        iteration=1,
        expected_game_value=-0.1,
        exploitability=0.4,
        nashconv=0.8,
        information_sets=936,
        runtime_seconds=0.5,
        game_tree_nodes=9450,
        iterations_per_second=2.0,
        best_response_seconds=0.1,
    )
    metrics = export_training_records((leduc_record,), tmp_path / "metrics.csv")
    assert "game_tree_nodes" in metrics.read_text(encoding="utf-8")

    comparison = export_kuhn_comparison(
        leduc_record,
        iterations=1,
        seed=7,
        evaluation_interval=1,
        logging_interval=1,
        path=tmp_path / "comparison.csv",
    )
    with comparison.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["game"] for row in rows] == ["Leduc", "Kuhn"]
    assert [row["nodes_per_iteration"] for row in rows] == ["9450", "54"]
