import math

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cards import DECK, Card, Rank
from pokerlab.research.leduc.game import LeducGame, LeducState
from pokerlab.research.leduc.information_sets import InformationSet
from pokerlab.research.leduc.mccfr import MCCFRConfig, MCCFRSolver

SAMPLE_COUNT = 12_000


def _set_distribution(profile, info, distribution):
    profile[info] = distribution


def _player0_fixture(solver):
    profile = solver.current_strategy_profile()
    private = Card(Rank.JACK, 0)
    board = Card(Rank.KING, 0)
    first = (Action.CHECK, Action.CHECK)
    for opponent_card in DECK:
        if opponent_card == private:
            continue
        preflop = InformationSet(1, opponent_card, None, 0, ((Action.CHECK,), ()))
        _set_distribution(profile, preflop, {Action.CHECK: 1.0, Action.BET: 0.0})
        if opponent_card in (private, board):
            continue
        postflop = InformationSet(1, opponent_card, board, 1, (first, ()))
        _set_distribution(profile, postflop, {Action.CHECK: 0.0, Action.BET: 1.0})

    root = InformationSet(0, private, None, 0, ((), ()))
    _set_distribution(profile, root, {Action.CHECK: 0.4, Action.BET: 0.6})
    target_state = LeducState(
        (private, Card(Rank.QUEEN, 0)),
        board,
        1,
        (first, (Action.BET,)),
        (1, 5),
    )
    return profile, InformationSet.from_state(target_state)


def _player1_fixture(solver):
    profile = solver.current_strategy_profile()
    private = Card(Rank.KING, 0)
    board = Card(Rank.KING, 1)
    for info in solver.information_sets:
        if info.player == 0 and info.round_index == 0 and not info.histories[0]:
            _set_distribution(profile, info, {Action.CHECK: 0.4, Action.BET: 0.6})

    response_preflop = InformationSet(1, private, None, 0, ((Action.BET,), ()))
    _set_distribution(
        profile,
        response_preflop,
        {Action.FOLD: 0.3, Action.CALL: 0.7, Action.RAISE: 0.0},
    )
    first_history = (Action.BET, Action.CALL)
    opener = InformationSet(1, private, board, 1, (first_history, ()))
    _set_distribution(profile, opener, {Action.CHECK: 1.0, Action.BET: 0.0})
    for opponent_card in DECK:
        if opponent_card in (private, board):
            continue
        bettor = InformationSet(
            0,
            opponent_card,
            board,
            1,
            (first_history, (Action.CHECK,)),
        )
        _set_distribution(profile, bettor, {Action.CHECK: 0.0, Action.BET: 1.0})

    target_state = LeducState(
        (Card(Rank.JACK, 0), private),
        board,
        1,
        (first_history, (Action.CHECK, Action.BET)),
        (7, 3),
    )
    return profile, InformationSet.from_state(target_state)


def _sample_mean_regret(solver, profile, target, traverser):
    solver._rng.seed(solver.config.seed + 100)
    totals = {action: 0.0 for action in solver._legal[target]}
    squares = dict.fromkeys(totals, 0.0)
    deals = LeducGame.deals()
    for _ in range(SAMPLE_COUNT):
        state = LeducGame.initial_state(solver._rng.choice(deals))
        delta = {}
        solver._traverse_external(state, traverser, profile, delta)
        observed = delta.get(target, dict.fromkeys(totals, 0.0))
        for action in totals:
            value = observed[action]
            totals[action] += value
            squares[action] += value * value
    means = {action: value / SAMPLE_COUNT for action, value in totals.items()}
    standard_errors = {}
    for action, total in totals.items():
        variance = max(
            0.0,
            (squares[action] - total**2 / SAMPLE_COUNT) / (SAMPLE_COUNT - 1),
        )
        standard_errors[action] = math.sqrt(variance / SAMPLE_COUNT)
    return means, standard_errors


def test_player0_sampled_regret_matches_independent_leduc_expectation():
    solver = MCCFRSolver(MCCFRConfig(iterations=1, seed=41))
    profile, target = _player0_fixture(solver)

    # For the four hidden opponent cards compatible with J0 and public K0,
    # each deal/board pair has probability 1/30 * 1/4 = 1/120. P1 checks
    # pre-flop and bets post-flop with probability one. At the target, fold
    # values -1; call values are (0,-5,-5,-5); raise values are
    # (2.5,-2,-2,-2), because P1's post-raise fold/call is uniform.
    # Uniform target play gives conditional regret sums (3.5,-7.5,4).
    expected = {
        Action.FOLD: 3.5 / 120,
        Action.CALL: -7.5 / 120,
        Action.RAISE: 4.0 / 120,
    }
    # P0's own pre-flop check probability is 0.4, but the target regrets do
    # not include it. The Monte Carlo draws sample true deal/chance/opponent
    # distributions and enumerate all P0 actions.
    means, standard_errors = _sample_mean_regret(solver, profile, target, 0)
    for action, value in expected.items():
        assert abs(means[action] - value) <= 6 * standard_errors[action] + 0.001


def test_player1_sampled_regret_excludes_own_reach_and_matches_expectation():
    solver = MCCFRSolver(MCCFRConfig(iterations=1, seed=53))
    profile, target = _player1_fixture(solver)

    # P1 holds K0 and K1 is public, so P1 has a pair and beats all four
    # compatible opponent cards. The sampled target is reached with probability
    # (4 * 1/30 * 1/4) * P0's pre-flop bet(0.6) = 1/50. Before it, P1 calls a
    # pre-flop bet with probability 0.7; that own reach is excluded from regret.
    # At the target, P1's utilities are fold=-3, call=+7, and raise=(7+11)/2=9.
    # Uniform play gives regret (-22/3, 8/3, 14/3), then chance/opponent
    # sampling gives expected increments (-11/75, 4/75, 7/75).
    expected = {
        Action.FOLD: -11 / 75,
        Action.CALL: 4 / 75,
        Action.RAISE: 7 / 75,
    }
    means, standard_errors = _sample_mean_regret(solver, profile, target, 1)
    for action, value in expected.items():
        assert abs(means[action] - value) <= 6 * standard_errors[action] + 0.002


def _independent_average_reference(solver, profile):
    average = {}
    seen = set()

    def visit(state, own_reach):
        if state.is_terminal:
            return
        if state.is_chance_node:
            for card, _ in state.chance_outcomes:
                visit(state.deal_public(card), own_reach)
            return
        player = state.current_player
        info = InformationSet.from_state(state)
        if info not in seen:
            average[info] = {
                action: own_reach[player] * profile[info][action]
                for action in state.legal_actions
            }
            seen.add(info)
        for action in state.legal_actions:
            reach = list(own_reach)
            reach[player] *= profile[info][action]
            visit(state.act(action), (reach[0], reach[1]))

    for deal in LeducGame.deals():
        visit(LeducGame.initial_state(deal), (1.0, 1.0))
    return average


def test_average_strategy_matches_independent_full_tree_own_reach():
    solver = MCCFRSolver(MCCFRConfig(iterations=1, seed=7))
    profile = solver.current_strategy_profile()
    expected = _independent_average_reference(solver, profile)
    actual = solver._average_strategy_delta(profile)
    assert actual.keys() == expected.keys()
    for info, actions in expected.items():
        for action, value in actions.items():
            assert math.isclose(actual[info][action], value, abs_tol=1e-12)


def test_shared_hidden_histories_have_one_player_observable_strategy():
    solver = MCCFRSolver(MCCFRConfig(iterations=1, seed=9))
    profile = solver.current_strategy_profile()
    private = Card(Rank.JACK, 0)
    board = Card(Rank.KING, 0)
    infos = set()
    for opponent_card in DECK:
        if opponent_card in (private, board):
            continue
        state = LeducState(
            (private, opponent_card),
            board,
            1,
            (
                (Action.CHECK, Action.CHECK),
                (Action.BET,),
            ),
            (1, 5),
        )
        infos.add(InformationSet.from_state(state))
    assert len(infos) == 1
    info = next(iter(infos))
    assert profile[info] == solver.current_strategy(info)


def test_leduc_mccfr_runs_are_reproducible_and_seed_sensitive():
    first = MCCFRSolver(MCCFRConfig(iterations=6, seed=17))
    second = MCCFRSolver(MCCFRConfig(iterations=6, seed=17))
    different = MCCFRSolver(MCCFRConfig(iterations=6, seed=18))
    for _ in range(6):
        first.run_iteration()
        second.run_iteration()
        different.run_iteration()
    assert first.regret_sum == second.regret_sum
    assert first.strategy_sum == second.strategy_sum
    assert first.regret_sum != different.regret_sum
