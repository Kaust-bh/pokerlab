"""Tests for Local Best Response (LBR) evaluation framework."""

from dataclasses import replace

import pytest

from pokerlab.agents import RandomAgent
from pokerlab.agents.base import Action, ActionType, Observation
from pokerlab.engine import Card
from pokerlab.evaluation.lbr import (
    BayesianRangeTracker,
    LocalBestResponseAgent,
    evaluate_lbr_hunl,
    evaluate_lbr_kuhn,
    evaluate_lbr_leduc,
)
from pokerlab.evaluation.tournament import TournamentConfig
from pokerlab.research.kuhn.cfr import CFRConfig as KuhnCFRConfig
from pokerlab.research.kuhn.cfr import CFRSolver as KuhnCFRSolver
from pokerlab.research.kuhn.evaluation import (
    best_response_value as kuhn_br_value,
)
from pokerlab.research.kuhn.evaluation import (
    expected_game_value as kuhn_ev0,
)
from pokerlab.research.leduc.cfr import (
    CFRConfig as LeducCFRConfig,
)
from pokerlab.research.leduc.cfr import (
    CFRSolver as LeducCFRSolver,
)
from pokerlab.research.leduc.evaluation import (
    best_response_value as leduc_br_value,
)
from pokerlab.research.leduc.evaluation import (
    expected_game_value as leduc_ev0,
)
from pokerlab.research.nlhe.agent import AbstractHUNLAgent
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.mccfr import ExternalSamplingMCCFR, HUNLMCCFRConfig


def test_bayesian_range_tracker_initialization_and_dead_cards():
    hero = (Card(14, "spades"), Card(14, "hearts"))
    tracker = BayesianRangeTracker(hero)
    assert len(tracker.weights) == 1225
    assert all(w == 1.0 for w in tracker.weights.values())

    flop = (Card(14, "clubs"), Card(10, "spades"), Card(2, "diamonds"))
    tracker.reveal_board_cards(flop)
    clashing = [h for h, w in tracker.weights.items() if w == 0.0]
    assert len(clashing) > 0
    assert tracker.weights.get((Card(14, "clubs"), Card(14, "diamonds")), 0.0) == 0.0


def test_bayesian_range_tracker_uses_exact_probabilities_without_mutating_rng():
    hero = (Card(14, "spades"), Card(13, "spades"))
    tracker = BayesianRangeTracker(hero)
    random_agent = RandomAgent(seed=12345)
    rng_before = random_agent._rng.getstate()

    obs = Observation(
        player="Target",
        hole_cards=(Card(2, "clubs"), Card(7, "diamonds")),
        community_cards=(),
        street="preflop",
        in_position=True,
        stack=100,
        opponent_stack=100,
        pot=3,
        to_call=1,
        big_blind=2,
        min_raise=2,
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.RAISE, 4),
        ),
    )
    tracker.update_beliefs(
        obs, Action(ActionType.CALL), random_agent.action_probabilities
    )
    assert random_agent._rng.getstate() == rng_before
    # Under uniform random play, all 1,225 non-dead hands have identical 1/3 likelihood
    active = [w for w in tracker.weights.values() if w > 0]
    assert len(active) == 1225
    assert max(active) == pytest.approx(min(active), abs=1e-12)


def test_lbr_matches_exact_best_response_on_kuhn_independently():
    solver = KuhnCFRSolver(KuhnCFRConfig(iterations=50))
    solver.train(50)
    cfr_profile = solver.average_strategy_profile()

    for profile in ({}, cfr_profile):
        ev0 = kuhn_ev0(profile)
        for player in (0, 1):
            exact_br = kuhn_br_value(profile, player)
            lbr_val = evaluate_lbr_kuhn(profile, player)
            assert lbr_val == pytest.approx(exact_br, abs=1e-9)
            player_ev = ev0 if player == 0 else -ev0
            assert lbr_val >= player_ev - 1e-12


def test_lbr_matches_and_bounds_exact_best_response_on_leduc_independently():
    solver = LeducCFRSolver(LeducCFRConfig(iterations=20))
    solver.train(20)
    cfr_profile = solver.average_strategy_profile()

    for profile in ({}, cfr_profile):
        ev0 = leduc_ev0(profile)
        for player in (0, 1):
            exact_br = leduc_br_value(profile, player)
            lbr_exact = evaluate_lbr_leduc(profile, player, mode="local_br")
            lbr_rollout = evaluate_lbr_leduc(profile, player, mode="check_call_rollout")
            player_ev = ev0 if player == 0 else -ev0
            assert lbr_exact == pytest.approx(exact_br, abs=1e-9)
            assert player_ev - 1e-9 <= lbr_rollout <= exact_br + 1e-9


def test_lbr_candidate_action_evs_account_for_all_in_risk_and_pot_odds():
    # River board: As Ks Qs Js 2d
    board = (
        Card(14, "spades"),
        Card(13, "spades"),
        Card(12, "spades"),
        Card(11, "spades"),
        Card(2, "diamonds"),
    )
    random_target = RandomAgent(seed=7)
    lbr = LocalBestResponseAgent(
        random_target, seed=7, opponent_samples=64, board_rollouts=1
    )

    # Case 1: Nut hand (10s 9s -> Royal Flush) facing a pot-sized bet (pot=40, to_call=20)
    nut_hole = (Card(10, "spades"), Card(9, "spades"))
    lbr.initialize_hand(nut_hole)
    nut_obs = Observation(
        player="LBR",
        hole_cards=nut_hole,
        community_cards=board,
        street="river",
        in_position=True,
        stack=80,
        opponent_stack=80,
        pot=40,
        to_call=20,
        big_blind=2,
        min_raise=20,
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.ALL_IN),
        ),
        street_contribution=0,
        opponent_street_contribution=20,
        is_button=True,
    )
    nut_evs = lbr.candidate_action_evs(nut_obs)
    assert nut_evs[Action(ActionType.FOLD)] == 0.0
    assert nut_evs[Action(ActionType.CALL)] == pytest.approx(40.0, abs=1e-6)
    assert nut_evs[Action(ActionType.ALL_IN)] > nut_evs[Action(ActionType.CALL)]
    assert lbr.act(nut_obs) == Action(ActionType.ALL_IN)

    # Case 2: Dead-low hand (3c 4d -> J-high on board, 0% win equity, ties only board-play hands)
    # Facing an all-in call where opponent has 10s (Royal Flush) locked in posterior
    trash_hole = (Card(3, "clubs"), Card(4, "diamonds"))
    lbr.initialize_hand(trash_hole)
    assert lbr.tracker is not None
    for h in lbr.tracker.weights:
        lbr.tracker.weights[h] = (
            1.0 if h == (Card(9, "spades"), Card(10, "spades")) else 0.0
        )
    trash_obs = replace(nut_obs, hole_cards=trash_hole)
    trash_evs = lbr.candidate_action_evs(trash_obs)
    assert trash_evs[Action(ActionType.FOLD)] == 0.0
    assert trash_evs[Action(ActionType.CALL)] == pytest.approx(-20.0, abs=1e-6)
    # ALL_IN risks 80 chips; opponent with Royal Flush folds 1/3 under RandomAgent and calls 2/3
    # EV(ALL_IN) = (1/3)*40 + (2/3)*(-80) = -40 < 0
    assert trash_evs[Action(ActionType.ALL_IN)] == pytest.approx(-40.0, abs=1e-6)
    assert lbr.act(trash_obs) == Action(ActionType.FOLD)


def test_lbr_hunl_evaluates_and_bounds_exploitability():
    game = HUNLGameConfig(
        starting_stack_bb=5,
        postflop_buckets=4,
        opponent_hand_samples=8,
        board_abstraction="isomorphic",
        average_strategy_mode="reach_weighted",
    )
    mccfr = ExternalSamplingMCCFR(HUNLMCCFRConfig(iterations=4, seed=123, game=game))
    mccfr.train(4)
    strategy = mccfr.average_strategy_profile()
    agent = AbstractHUNLAgent(strategy, game, seed=123)

    tourney_cfg = TournamentConfig(starting_stack=10, small_blind=1, big_blind=2)
    result = evaluate_lbr_hunl(agent, hands=20, config=tourney_cfg, seed=42)

    assert result.game == "HUNL"
    assert result.hands_evaluated == 20
    assert result.runtime_seconds > 0
    assert isinstance(result.lbr_mbb_per_hand, float)
    assert result.ci95_low_mbb <= result.ci95_high_mbb


def test_lbr_exploits_random_agent():
    random_agent = RandomAgent(seed=99)
    tourney_cfg = TournamentConfig(starting_stack=10, small_blind=1, big_blind=2)
    result = evaluate_lbr_hunl(random_agent, hands=40, config=tourney_cfg, seed=42)

    assert result.hands_evaluated == 40
    assert result.runtime_seconds > 0
    assert result.lbr_bb_per_100 > 0.0
