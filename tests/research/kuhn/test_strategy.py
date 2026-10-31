import json

import pytest

from pokerlab.research.kuhn.game import KuhnAction as A
from pokerlab.research.kuhn.game import KuhnCard as C
from pokerlab.research.kuhn.information_sets import InformationSet
from pokerlab.research.kuhn.strategy import (
    load_strategy_snapshot,
    normalize_average_strategy,
    regret_matching,
    save_strategy_snapshot,
)


def test_regret_matching_positive_regrets_and_uniform_fallback() -> None:
    actions = (A.CHECK, A.BET)
    assert regret_matching({A.CHECK: 2.0, A.BET: 6.0}, actions) == {
        A.CHECK: 0.25,
        A.BET: 0.75,
    }
    assert regret_matching({A.CHECK: -1.0, A.BET: 0.0}, actions) == {
        A.CHECK: 0.5,
        A.BET: 0.5,
    }


def test_strategy_distributions_only_contain_legal_actions() -> None:
    actions = (A.FOLD, A.CALL)
    strategy = normalize_average_strategy(
        {A.FOLD: 1.0, A.CALL: 3.0, A.BET: 9.0}, actions
    )
    assert strategy == {A.FOLD: 0.25, A.CALL: 0.75}
    assert sum(strategy.values()) == pytest.approx(1)
    assert A.BET not in strategy


def test_snapshot_round_trip_and_rejects_illegal_actions(tmp_path) -> None:
    info = InformationSet(0, C.KING, ())
    strategy = {info: {A.CHECK: 0.25, A.BET: 0.75}}
    path = save_strategy_snapshot(strategy, tmp_path / "strategy.json")
    assert load_strategy_snapshot(path) == strategy

    record = json.loads(path.read_text())
    record[0]["strategy"] = {"fold": 1.0}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="illegal actions"):
        load_strategy_snapshot(path)
