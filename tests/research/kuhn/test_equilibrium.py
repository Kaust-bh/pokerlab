import pytest

from pokerlab.research.kuhn.cfr import CFRConfig, CFRSolver
from pokerlab.research.kuhn.evaluation import (
    best_response_value,
    evaluate_nash,
    expected_game_value,
)
from pokerlab.research.kuhn.game import KuhnAction as A
from pokerlab.research.kuhn.game import KuhnCard as C
from pokerlab.research.kuhn.information_sets import InformationSet


def known_equilibrium() -> dict[InformationSet, dict[A, float]]:
    profile = {}
    for card in C:
        p0_root_bet = {C.JACK: 1 / 3, C.QUEEN: 0.0, C.KING: 1.0}[card]
        profile[InformationSet(0, card, ())] = {
            A.CHECK: 1 - p0_root_bet,
            A.BET: p0_root_bet,
        }
        p0_call = {C.JACK: 0.0, C.QUEEN: 2 / 3, C.KING: 1.0}[card]
        profile[InformationSet(0, card, (A.CHECK, A.BET))] = {
            A.FOLD: 1 - p0_call,
            A.CALL: p0_call,
        }

        p1_after_check_bet = {C.JACK: 1 / 3, C.QUEEN: 0.0, C.KING: 1.0}[card]
        profile[InformationSet(1, card, (A.CHECK,))] = {
            A.CHECK: 1 - p1_after_check_bet,
            A.BET: p1_after_check_bet,
        }
        p1_call = {C.JACK: 0.0, C.QUEEN: 1 / 3, C.KING: 1.0}[card]
        profile[InformationSet(1, card, (A.BET,))] = {
            A.FOLD: 1 - p1_call,
            A.CALL: p1_call,
        }
    return profile


def test_exact_known_equilibrium_has_value_minus_one_eighteenth() -> None:
    profile = known_equilibrium()
    assert expected_game_value(profile) == pytest.approx(-1 / 18, abs=1e-12)
    result = evaluate_nash(profile)
    assert result.best_response_value_player0 == pytest.approx(-1 / 18, abs=1e-12)
    assert result.best_response_value_player1 == pytest.approx(1 / 18, abs=1e-12)
    assert result.nashconv == pytest.approx(0, abs=1e-12)
    assert result.exploitability == pytest.approx(0, abs=1e-12)
    assert best_response_value(profile, 0) == pytest.approx(-1 / 18)


def test_cfr_average_strategy_converges_in_value_and_exploitability() -> None:
    solver = CFRSolver(CFRConfig(iterations=5_000, seed=3, evaluation_interval=1_000))
    records = solver.train()
    final = records[-1]
    assert final.expected_game_value == pytest.approx(-1 / 18, abs=0.001)
    assert final.exploitability < 0.01
    assert final.exploitability < records[0].exploitability

    profile = solver.average_strategy_profile()
    assert len(profile) == 12
    for distribution in profile.values():
        assert sum(distribution.values()) == pytest.approx(1)
