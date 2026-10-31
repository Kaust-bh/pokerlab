import math

from pokerlab.research.leduc.cfr_plus import CFRPlusConfig, CFRPlusSolver
from pokerlab.research.leduc.dcfr import DCFRConfig, DCFRSolver
from pokerlab.research.leduc.evaluation import evaluate_nash

from .test_mccfr import _independent_average_reference


def test_leduc_cfr_plus_clips_alternates_and_linearly_averages(monkeypatch):
    solver = CFRPlusSolver(CFRPlusConfig(iterations=2, seed=13))
    profiles = []
    original = solver._regret_pass

    def capture(*args):
        traverser, profile = args[1], args[2]
        if not profiles or profiles[-1][0] != traverser:
            profiles.append(
                (traverser, {info: dist.copy() for info, dist in profile.items()})
            )
        return original(*args)

    monkeypatch.setattr(solver, "_regret_pass", capture)
    solver.run_iteration()
    solver.run_iteration()
    assert [player for player, _ in profiles] == [0, 1, 0, 1]
    assert profiles[0][1] != profiles[1][1]
    assert solver.iterations == 2
    assert solver.traversals == solver.player_updates == 4
    assert all(
        regret >= 0
        for actions in solver.regret_sum.values()
        for regret in actions.values()
    )

    expected = {
        info: dict.fromkeys(actions, 0.0) for info, actions in solver._legal.items()
    }
    for iteration in (1, 2):
        for player in (0, 1):
            profile = profiles[(iteration - 1) * 2 + player][1]
            reference = _independent_average_reference(solver, profile)
            for info, actions in reference.items():
                if info.player == player:
                    for action, value in actions.items():
                        expected[info][action] += iteration * value
    for info, actions in expected.items():
        for action, value in actions.items():
            assert math.isclose(solver.strategy_sum[info][action], value, abs_tol=1e-12)


def test_leduc_dcfr_uses_defaults_and_exact_evaluator():
    config = DCFRConfig(iterations=1, seed=7)
    assert (config.alpha, config.beta, config.gamma) == (1.5, 0.0, 2.0)
    solver = DCFRSolver(config)
    record = solver.train()[0]
    assert record.iteration == 1
    assert record.traversals == 1
    assert record.player_updates == 2
    direct = evaluate_nash(solver.average_strategy_profile())
    assert math.isclose(record.exploitability, direct.exploitability)
    assert all(
        math.isclose(sum(strategy.values()), 1.0, abs_tol=1e-12)
        for strategy in solver.average_strategy_profile().values()
    )


def test_leduc_variants_are_reproducible():
    first = CFRPlusSolver(CFRPlusConfig(iterations=1, seed=29))
    second = CFRPlusSolver(CFRPlusConfig(iterations=1, seed=29))
    first.run_iteration()
    second.run_iteration()
    assert first.regret_sum == second.regret_sum
    assert first.strategy_sum == second.strategy_sum

    dcfr1 = DCFRSolver(DCFRConfig(iterations=2, seed=31))
    dcfr2 = DCFRSolver(DCFRConfig(iterations=2, seed=31))
    for _ in range(2):
        dcfr1.run_iteration()
        dcfr2.run_iteration()
    assert dcfr1.regret_sum == dcfr2.regret_sum
    assert dcfr1.strategy_sum == dcfr2.strategy_sum
