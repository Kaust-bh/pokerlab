import pytest

from pokerlab.research.kuhn.cfr import CFRConfig, CFRSolver
from pokerlab.research.kuhn.game import KuhnAction as A
from pokerlab.research.kuhn.game import KuhnCard as C
from pokerlab.research.kuhn.game import legal_actions_for_history
from pokerlab.research.kuhn.information_sets import InformationSet


def test_one_iteration_counterfactual_regret_and_average_reach_weight() -> None:
    solver = CFRSolver(CFRConfig(iterations=1, seed=7))
    solver.run_iteration()

    root_jack = InformationSet(0, C.JACK, ())
    assert solver.regret_sum[root_jack][A.BET] == pytest.approx(1 / 8)
    assert solver.regret_sum[root_jack][A.CHECK] == pytest.approx(-1 / 8)
    assert solver.strategy_sum[root_jack] == {A.CHECK: 0.5, A.BET: 0.5}

    facing_bet_jack = InformationSet(0, C.JACK, (A.CHECK, A.BET))
    assert solver.strategy_sum[facing_bet_jack] == {A.FOLD: 0.25, A.CALL: 0.25}


def test_current_and_average_strategies_are_normalized_and_reproducible() -> None:
    a = CFRSolver(CFRConfig(iterations=100, seed=19))
    b = CFRSolver(CFRConfig(iterations=100, seed=19))
    a.train()
    b.train()
    strategy = a.average_strategy_profile()
    assert strategy == b.average_strategy_profile()
    assert len(strategy) == 12
    for info_set, distribution in strategy.items():
        assert sum(distribution.values()) == pytest.approx(1)
        assert set(distribution) == set(legal_actions_for_history(info_set.history))
        assert sum(a.current_strategy(info_set).values()) == pytest.approx(1)


def test_training_records_final_iteration_even_when_not_on_interval() -> None:
    solver = CFRSolver(CFRConfig(iterations=10, evaluation_interval=6))
    records = solver.train()
    assert [record.iteration for record in records] == [6, 10]
    assert all(record.information_sets == 12 for record in records)


def test_training_rejects_invalid_intervals() -> None:
    solver = CFRSolver()
    with pytest.raises(ValueError, match="positive"):
        solver.train(2, evaluation_interval=0)
