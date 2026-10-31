import math

import pytest

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cards import DECK
from pokerlab.research.leduc.cfr import CFRConfig, CFRSolver
from pokerlab.research.leduc.game import LeducGame, LeducState
from pokerlab.research.leduc.information_sets import InformationSet
from pokerlab.research.leduc.strategy import normalize_average_strategy, regret_matching


def test_regret_matching_and_average_strategy_are_valid():
    legal = (Action.FOLD, Action.CALL, Action.RAISE)
    assert regret_matching({}, legal) == {a: 1 / 3 for a in legal}
    assert regret_matching(
        {Action.FOLD: -2, Action.CALL: 3, Action.RAISE: 1}, legal
    ) == {
        Action.FOLD: 0,
        Action.CALL: 0.75,
        Action.RAISE: 0.25,
    }
    average = normalize_average_strategy(
        {Action.FOLD: 2, Action.CALL: 0, Action.RAISE: 2}, legal
    )
    assert average == {Action.FOLD: 0.5, Action.CALL: 0, Action.RAISE: 0.5}


def test_full_tree_iteration_builds_valid_strategies_and_counters():
    solver = CFRSolver(CFRConfig(iterations=1, seed=17))
    solver.run_iteration()
    assert solver.iterations == 1
    assert solver.game_tree_nodes == 9_450
    assert len(solver.regret_sum) == 936
    for info, strategy in solver.average_strategy_profile().items():
        legal = solver._legal[info]
        assert set(strategy) == set(legal)
        assert math.isclose(sum(strategy.values()), 1.0)
        assert all(math.isfinite(prob) and prob >= 0 for prob in strategy.values())


def test_cfr_run_is_reproducible_for_fixed_seed():
    first = CFRSolver(CFRConfig(iterations=1, seed=8))
    second = CFRSolver(CFRConfig(iterations=1, seed=8))
    first.run_iteration()
    second.run_iteration()
    assert first.regret_sum == second.regret_sum
    assert first.strategy_sum == second.strategy_sum


def test_info_lookup_derives_only_legal_actions():
    solver = CFRSolver()
    info = InformationSet.from_state(LeducGame.initial_state(DECK[:2]))
    assert set(solver.current_strategy(info)) == {Action.CHECK, Action.BET}


def test_counterfactual_regret_and_average_reach_weights():
    solver = CFRSolver()
    state = LeducGame.initial_state(DECK[:2])
    info = InformationSet.from_state(state)
    solver._ensure(info, state.legal_actions)
    solver.regret_sum[info] = {Action.CHECK: 1.0, Action.BET: 3.0}
    strategy = solver.current_strategy(info)
    regret_delta = {}
    average_delta = {}
    solver._traverse(
        state,
        reach=(0.25, 0.75),
        chance_reach=0.2,
        regrets=regret_delta,
        averages=average_delta,
        averaged=set(),
        node_counter=[0],
    )

    assert math.isclose(strategy[Action.CHECK], 0.25)
    assert math.isclose(strategy[Action.BET], 0.75)
    assert math.isclose(average_delta[info][Action.CHECK], 0.0625)
    assert math.isclose(average_delta[info][Action.BET], 0.1875)
    assert math.isclose(
        sum(strategy[action] * regret_delta[info][action] for action in strategy),
        0.0,
        abs_tol=1e-12,
    )
    assert all(math.isfinite(value) for value in regret_delta[info].values())


def test_first_iteration_regret_matches_independent_postflop_oracle():
    """Pin one first-iteration update using hand-derived terminal utilities.

    This concrete state has private cards J0/K0, public J1, checked pre-flop,
    and a player-1 post-flop bet. Its deal and board event has chance reach
    (1/30)*(1/4)=1/120. Player 0 holds a pair of Jacks and wins every call or
    raise showdown against K0. At this state, fold is -1, call is +5, and
    raise averages (+5 on the opponent's fold, +9 on its call)=+7 because the
    opponent uses its initial uniform fold/call strategy. The uniform
    three-action node value is 11/3, so the unweighted regrets are
    (-14/3, 4/3, 10/3).

    Multiplying by chance reach 1/120 and opponent reach 3/4 gives
    (-7/240, 1/120, 1/48). Player 0's own reach 1/4 is absent; including it
    would make each regret four times smaller. This independently derived
    vector checks the action utilities and all three counterfactual factors.
    """
    from pokerlab.research.leduc.cards import Card, Rank

    solver = CFRSolver()
    state = LeducState(
        private_cards=(Card(Rank.JACK, 0), Card(Rank.KING, 0)),
        public_card=Card(Rank.JACK, 1),
        round_index=1,
        histories=((Action.CHECK, Action.CHECK), (Action.BET,)),
        contributions=(1, 5),
    )
    info = InformationSet.from_state(state)
    regrets: dict[InformationSet, dict[Action, float]] = {}
    averages: dict[InformationSet, dict[Action, float]] = {}

    # These expected values are derived above from the finite Leduc outcomes,
    # independently of the production traversal's regret-update equation.
    solver._traverse(
        state,
        reach=(0.25, 0.75),
        chance_reach=1 / 120,
        regrets=regrets,
        averages=averages,
        averaged=set(),
        node_counter=[0],
    )

    assert regrets[info][Action.FOLD] == pytest.approx(-7 / 240)
    assert regrets[info][Action.CALL] == pytest.approx(1 / 120)
    assert regrets[info][Action.RAISE] == pytest.approx(1 / 48)
    # Average strategy uses own reach, unlike counterfactual regret weight.
    assert averages[info][Action.FOLD] == pytest.approx(1 / 12)
    assert averages[info][Action.CALL] == pytest.approx(1 / 12)
    assert averages[info][Action.RAISE] == pytest.approx(1 / 12)
