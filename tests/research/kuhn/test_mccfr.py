import math
from itertools import product

from pokerlab.research.kuhn.game import KuhnGame
from pokerlab.research.kuhn.information_sets import InformationSet
from pokerlab.research.kuhn.mccfr import MCCFRConfig, MCCFRSolver


def _fixed_profile(solver):
    profile = {}
    for index, info in enumerate(solver.information_sets):
        actions = solver._legal[info]
        first_probability = 0.2 + 0.1 * (index % 5)
        profile[info] = {
            actions[0]: first_probability,
            actions[1]: 1.0 - first_probability,
        }
    return profile


def _independent_vanilla_updates(profile):
    regrets = {}
    averages = {}
    averaged = set()

    def visit(state, reach, chance_reach):
        if state.is_terminal:
            return float(state.utility(0))
        player = state.current_player
        info = InformationSet.from_state(state)
        actions = state.legal_actions
        action_values = {}
        value = 0.0
        for action in actions:
            next_reach = list(reach)
            next_reach[player] *= profile[info][action]
            action_values[action] = visit(
                state.act(action),
                (next_reach[0], next_reach[1]),
                chance_reach,
            )
            value += profile[info][action] * action_values[action]
        if info not in averaged:
            averages[info] = {
                action: reach[player] * profile[info][action] for action in actions
            }
            averaged.add(info)
        sign = 1.0 if player == 0 else -1.0
        weight = chance_reach * reach[1 - player]
        delta = regrets.setdefault(info, dict.fromkeys(actions, 0.0))
        for action in actions:
            delta[action] += weight * sign * (action_values[action] - value)
        return value

    deals = tuple(KuhnGame.deals())
    for deal in deals:
        visit(KuhnGame.initial_state(deal), (1.0, 1.0), 1.0 / len(deals))
    return regrets, averages


def _exhaustive_external_expectation(solver, profile, traverser):
    external_infos = tuple(
        info for info in solver.information_sets if info.player != traverser
    )
    total = {}
    deals = tuple(KuhnGame.deals())
    for deal in deals:
        for choices in product(*(solver._legal[info] for info in external_infos)):
            sampled = dict(zip(external_infos, choices, strict=True))
            probability = 1.0 / len(deals)
            for info, action in sampled.items():
                probability *= profile[info][action]
            if probability == 0:
                continue
            delta = {}
            solver._traverse_external(
                KuhnGame.initial_state(deal),
                traverser,
                profile,
                delta,
                opponent_choices=sampled,
            )
            for info, actions in delta.items():
                target = total.setdefault(info, dict.fromkeys(actions, 0.0))
                for action, value in actions.items():
                    target[action] += probability * value
    return total


def test_exhaustive_external_sampling_regrets_match_independent_vanilla():
    solver = MCCFRSolver(MCCFRConfig(iterations=1, seed=5))
    profile = _fixed_profile(solver)
    exact_regrets, _ = _independent_vanilla_updates(profile)

    # Enumerating deals and pure external action maps exactly integrates the
    # true chance and opponent distributions for each traverser.
    for traverser in (0, 1):
        expected = _exhaustive_external_expectation(solver, profile, traverser)
        for info, actions in exact_regrets.items():
            if info.player != traverser:
                continue
            for action, value in actions.items():
                assert math.isclose(
                    expected.get(info, {}).get(action, 0.0),
                    value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )


def test_average_strategy_delta_matches_full_tree_own_reach_reference():
    solver = MCCFRSolver(MCCFRConfig(iterations=1, seed=2))
    profile = _fixed_profile(solver)
    _, exact_average = _independent_vanilla_updates(profile)
    actual = solver._average_strategy_delta(profile)
    assert actual.keys() == exact_average.keys()
    for info, actions in exact_average.items():
        for action, value in actions.items():
            assert math.isclose(actual[info][action], value, abs_tol=1e-12)


def test_shared_hidden_deals_use_the_same_information_set_policy():
    solver = MCCFRSolver(MCCFRConfig(iterations=1, seed=3))
    profile = solver.current_strategy_profile()
    deals = tuple(KuhnGame.deals())
    states = [KuhnGame.initial_state(deal) for deal in deals if deal[0].name == "JACK"]
    infos = {InformationSet.from_state(state) for state in states}
    assert len(states) == 2
    assert len(infos) == 1
    info = next(iter(infos))
    assert profile[info] == solver.current_strategy(info)


def test_mccfr_runs_are_seed_reproducible_and_seed_sensitive():
    first = MCCFRSolver(MCCFRConfig(iterations=20, seed=12))
    second = MCCFRSolver(MCCFRConfig(iterations=20, seed=12))
    different = MCCFRSolver(MCCFRConfig(iterations=20, seed=13))
    for _ in range(20):
        first.run_iteration()
        second.run_iteration()
        different.run_iteration()
    assert first.regret_sum == second.regret_sum
    assert first.strategy_sum == second.strategy_sum
    assert first.regret_sum != different.regret_sum


def test_both_player_passes_share_one_frozen_profile(monkeypatch):
    solver = MCCFRSolver(MCCFRConfig(iterations=1, seed=22))
    traversed_profiles = []
    original = solver._traverse_external

    def capture(*args, **kwargs):
        if not args[0].history:
            traversed_profiles.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_traverse_external", capture)
    solver.run_iteration()
    assert len(traversed_profiles) == 2
    assert traversed_profiles[0] is traversed_profiles[1]


def test_training_records_use_exact_evaluation_and_training_only_rate():
    solver = MCCFRSolver(MCCFRConfig(iterations=2, seed=31, evaluation_interval=1))
    records = solver.train()
    assert len(records) == 2
    for record in records:
        assert math.isclose(
            record.expected_game_value_player0 + record.expected_game_value_player1,
            0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(record.nashconv, 2 * record.exploitability, abs_tol=1e-12)
        assert record.training_elapsed_seconds > 0
        assert record.evaluation_elapsed_seconds >= 0
        assert math.isclose(
            record.iterations_per_second,
            record.iteration / record.training_elapsed_seconds,
        )
        assert record.seed == 31
