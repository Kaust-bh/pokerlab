import math

from pokerlab.research.kuhn.cfr_plus import CFRPlusConfig, CFRPlusSolver
from pokerlab.research.kuhn.dcfr import DCFRConfig, DCFRSolver

from .test_mccfr import _independent_vanilla_updates


def test_kuhn_cfr_plus_alternates_and_clips_regrets(monkeypatch):
    solver = CFRPlusSolver(CFRPlusConfig(iterations=1, seed=21))
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
    assert [player for player, _ in profiles] == [0, 1]
    assert profiles[0][1] != profiles[1][1]
    assert solver.iterations == 1
    assert solver.traversals == solver.player_updates == 2
    assert all(
        regret >= 0
        for actions in solver.regret_sum.values()
        for regret in actions.values()
    )


def test_kuhn_cfr_plus_linear_average_uses_own_reach():
    solver = CFRPlusSolver(CFRPlusConfig(iterations=2, seed=4))
    pass_profiles = []
    original = solver._regret_pass

    def capture(*args):
        traverser, profile = args[1], args[2]
        if not pass_profiles or pass_profiles[-1][0] != traverser:
            pass_profiles.append(
                (traverser, {info: dist.copy() for info, dist in profile.items()})
            )
        return original(*args)

    solver._regret_pass = capture
    solver.run_iteration()
    solver.run_iteration()
    assert [player for player, _ in pass_profiles] == [0, 1, 0, 1]
    expected = {
        info: dict.fromkeys(actions, 0.0) for info, actions in solver._legal.items()
    }
    for iteration in (1, 2):
        for player in (0, 1):
            profile = pass_profiles[(iteration - 1) * 2 + player][1]
            _, average = _independent_vanilla_updates(profile)
            for info, actions in average.items():
                if info.player == player:
                    for action, value in actions.items():
                        expected[info][action] += iteration * value
    for info, actions in expected.items():
        for action, value in actions.items():
            assert math.isclose(solver.strategy_sum[info][action], value, abs_tol=1e-12)


def test_kuhn_dcfr_defaults_parameters_and_evaluates_exactly():
    config = DCFRConfig(iterations=2, seed=9)
    assert (config.alpha, config.beta, config.gamma) == (1.5, 0.0, 2.0)
    solver = DCFRSolver(config)
    records = solver.train()
    assert len(records) == 1
    assert records[0].iteration == 2
    assert records[0].player_updates == 4
    assert records[0].traversals == 2
    assert math.isclose(
        records[0].expected_game_value_player0 + records[0].expected_game_value_player1,
        0.0,
        abs_tol=1e-12,
    )
    assert all(
        math.isclose(sum(strategy.values()), 1.0, abs_tol=1e-12)
        for strategy in solver.average_strategy_profile().values()
    )


def test_kuhn_variant_runs_are_reproducible():
    first = CFRPlusSolver(CFRPlusConfig(iterations=3, seed=17))
    second = CFRPlusSolver(CFRPlusConfig(iterations=3, seed=17))
    for _ in range(3):
        first.run_iteration()
        second.run_iteration()
    assert first.regret_sum == second.regret_sum
    assert first.strategy_sum == second.strategy_sum

    dcfr1 = DCFRSolver(DCFRConfig(iterations=3, seed=33))
    dcfr2 = DCFRSolver(DCFRConfig(iterations=3, seed=33))
    for _ in range(3):
        dcfr1.run_iteration()
        dcfr2.run_iteration()
    assert dcfr1.regret_sum == dcfr2.regret_sum
    assert dcfr1.strategy_sum == dcfr2.strategy_sum
