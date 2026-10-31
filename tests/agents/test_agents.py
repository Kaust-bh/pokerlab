import pytest

from pokerlab.agents import (
    EquityAgent,
    EquityAgentConfig,
    HeuristicAgent,
    Observation,
    RandomAgent,
)
from pokerlab.agents.base import observe
from pokerlab.engine import Action, ActionType, Card, Deck, GameState


def obs(
    *,
    hole=None,
    board=(),
    legal=None,
    to_call=0,
    pot=20,
    position=True,
    stack=100,
):
    if hole is None:
        hole = (Card(14, "spades"), Card(14, "hearts"))
    if legal is None:
        legal = (Action(ActionType.CHECK), Action(ActionType.BET, 10))
    return Observation(
        "hero",
        hole,
        board,
        "preflop" if not board else "flop",
        position,
        stack,
        100,
        pot,
        to_call,
        2,
        2,
        legal,
    )


def test_random_agent_is_reproducible_and_selects_only_legal_actions() -> None:
    observation = obs(legal=(Action(ActionType.CHECK), Action(ActionType.BET, 10)))
    a, b = RandomAgent(seed=7), RandomAgent(seed=7)
    assert [a.act(observation) for _ in range(20)] == [
        b.act(observation) for _ in range(20)
    ]
    assert all(a.act(observation) in observation.legal_actions for _ in range(50))
    with pytest.raises(ValueError, match="no legal"):
        a.act(obs(legal=()))


def test_heuristic_checks_free_and_uses_strong_hands() -> None:
    agent = HeuristicAgent()
    check = Action(ActionType.CHECK)
    free_check = obs(
        hole=(Card(2, "spades"), Card(7, "hearts")),
        legal=(check, Action(ActionType.BET, 10)),
    )
    assert agent.act(free_check) == check
    value = obs(
        hole=(Card(14, "spades"), Card(14, "hearts")),
        board=(Card(14, "clubs"), Card(2, "clubs"), Card(7, "diamonds")),
        legal=(check, Action(ActionType.BET, 10)),
    )
    assert agent.act(value) == Action(ActionType.BET, 10)


def test_observation_hides_opponent_hole_cards_and_has_legal_actions() -> None:
    game = GameState({"hero": 100, "villain": 100}, deck=Deck(seed=3))
    view = observe(game, "hero")
    assert view.hole_cards == game.hole_cards["hero"]
    assert not hasattr(view, "opponent_hole_cards")
    assert game.hole_cards["villain"] not in view.hole_cards
    assert view.legal_actions


def test_big_blind_can_check_or_raise_after_preflop_call() -> None:
    game = GameState({"hero": 100, "villain": 100}, deck=Deck(seed=8))
    game.act(ActionType.CALL, player="hero")
    view = observe(game, "villain")
    assert Action(ActionType.CHECK) in view.legal_actions
    assert any(action.type == ActionType.RAISE for action in view.legal_actions)
    assert not any(action.type == ActionType.BET for action in view.legal_actions)


def test_heuristic_folds_weak_holding_to_large_price() -> None:
    agent = HeuristicAgent()
    weak = obs(
        hole=(Card(2, "spades"), Card(7, "hearts")),
        legal=(Action(ActionType.FOLD), Action(ActionType.CALL)),
        to_call=80,
        pot=20,
        position=False,
    )
    assert agent.act(weak) == Action(ActionType.FOLD)


def test_pot_odds_account_for_a_short_stack_and_equity_agent_seed() -> None:
    short = obs(stack=5, to_call=10, pot=20, legal=(Action(ActionType.CALL),))
    assert short.pot_odds == 0.2
    agent = EquityAgent(EquityAgentConfig(simulations=20), seed=11)
    agent.reset(11)
    first = agent.estimate(obs())
    agent.reset(11)
    second = agent.estimate(obs())
    assert first.equity == second.equity
    assert agent.act(obs()) in obs().legal_actions
