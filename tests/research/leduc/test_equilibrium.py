import math

from pokerlab.research.leduc import evaluation
from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cards import DECK, Card, Rank
from pokerlab.research.leduc.cfr import CFRConfig, CFRSolver
from pokerlab.research.leduc.evaluation import evaluate_nash
from pokerlab.research.leduc.game import LeducState
from pokerlab.research.leduc.information_sets import InformationSet


def test_average_profile_remains_valid_after_short_training():
    solver = CFRSolver(CFRConfig(iterations=4, seed=13))
    solver.train(iterations=4, evaluation_interval=4)
    profile = solver.average_strategy_profile()
    assert len(profile) == 936
    assert all(
        math.isclose(sum(distribution.values()), 1.0)
        for distribution in profile.values()
    )
    result = evaluate_nash(profile)
    assert math.isfinite(result.expected_value_player0)
    assert result.exploitability >= 0


def test_best_response_aggregates_hidden_histories_before_choosing(monkeypatch):
    """A single decision must serve four equally weighted opponent cards.

    With J0 for player 0 and K0 public, the opponent may hold J1, Q0, Q1,
    or K1. At the selected postflop decision, folding is worth +1 in every
    hidden state. Calling is worth 0 against J1 (a tied J-high showdown) and
    -5 against each other card, so its aggregate is -15/4. Raising makes the
    opponent fold with J1 (+5 to player 0) and call with the other cards
    (-9 each), for aggregate -11/2. Thus the legal shared action is FOLD.

    A hidden-state-aware response would raise against J1 (+5) and fold against
    the other three (+1 each), averaging +2. That value is unattainable to an
    information-set response. The expected action scores are derived directly
    from Leduc's second-round wager of four chips, equal contributions at
    showdown, and the four equiprobable remaining physical cards.
    """
    aggregate_values = {
        Action.FOLD: 1.0,
        Action.CALL: -15 / 4,
        Action.RAISE: -11 / 2,
    }
    hidden_state_best_response_value = (5 + 1 + 1 + 1) / 4
    assert max(aggregate_values, key=aggregate_values.get) is Action.FOLD
    assert hidden_state_best_response_value == 2.0

    private = Card(Rank.JACK, 0)
    board = Card(Rank.KING, 0)
    hidden_cards = tuple(card for card in DECK if card not in (private, board))
    first_history = (Action.CHECK, Action.CHECK)
    target_history = (Action.BET,)
    target_state = LeducState(
        (private, hidden_cards[0]),
        board,
        1,
        (first_history, target_history),
        (1, 5),
    )
    target = InformationSet.from_state(target_state)
    assert target.player == 0
    assert len(hidden_cards) == 4

    profile = {}
    # Ensure player 1 bets at this postflop node, independent of their private
    # card, so all four hidden states reach the selected information set.
    for opponent_card in hidden_cards:
        bet_info = InformationSet(1, opponent_card, board, 1, (first_history, ()))
        profile[bet_info] = {Action.CHECK: 0.0, Action.BET: 1.0}

        response_info = InformationSet(
            1, opponent_card, board, 1, (first_history, (Action.BET, Action.RAISE))
        )
        response = Action.FOLD if opponent_card == Card(Rank.JACK, 1) else Action.CALL
        profile[response_info] = {
            Action.FOLD: float(response is Action.FOLD),
            Action.CALL: float(response is Action.CALL),
        }

    captured = {}

    def capture_response_profile(response_profile):
        captured.update(response_profile)
        return 0.0

    monkeypatch.setattr(evaluation, "expected_game_value", capture_response_profile)
    evaluation.best_response_value(profile, player=0)

    assert captured[target] == {
        Action.FOLD: 1.0,
        Action.CALL: 0.0,
        Action.RAISE: 0.0,
    }
