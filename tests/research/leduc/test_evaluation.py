import math
from collections import deque

from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.evaluation import (
    best_response_value,
    evaluate_nash,
    expected_game_value,
)
from pokerlab.research.leduc.game import LeducGame
from pokerlab.research.leduc.information_sets import InformationSet


def test_uniform_profile_has_zero_sum_value_and_valid_nashconv():
    profile = {}
    value = expected_game_value(profile)
    assert math.isfinite(value)
    br0 = best_response_value(profile, 0)
    br1 = best_response_value(profile, 1)
    assert br0 >= value
    assert br1 >= -value
    result = evaluate_nash(profile)
    assert math.isclose(result.nashconv, (br0 - value) + (br1 + value))
    assert math.isclose(result.exploitability, result.nashconv / 2)


def test_best_response_is_finite_and_at_least_profile_value():
    result = evaluate_nash({})
    assert math.isfinite(result.best_response_value_player0)
    assert math.isfinite(result.best_response_value_player1)
    assert result.nashconv >= 0


def test_exact_best_response_matches_an_analytic_fold_policy():
    # Player 1 checks whenever free to act and folds to every wager.
    profile = {}
    for deal in LeducGame.deals():
        queue = deque([LeducGame.initial_state(deal)])
        while queue:
            state = queue.popleft()
            if state.is_terminal:
                continue
            if state.is_chance_node:
                queue.extend(
                    state.deal_public(card) for card, _ in state.chance_outcomes
                )
                continue
            if state.current_player == 1:
                info = InformationSet.from_state(state)
                action = (
                    Action.FOLD if Action.FOLD in state.legal_actions else Action.CHECK
                )
                profile[info] = {
                    candidate: float(candidate is action)
                    for candidate in state.legal_actions
                }
            queue.extend(state.act(action) for action in state.legal_actions)

    # Player 0 can guarantee the opponent's uncalled ante (+1) by betting.
    # No response can win more because Player 1 never calls or puts in chips.
    assert math.isclose(best_response_value(profile, 0), 1.0, abs_tol=1e-12)
