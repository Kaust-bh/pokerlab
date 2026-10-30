import csv

from pokerlab.agents import RandomAgent
from pokerlab.evaluation import (
    BenchmarkRunner,
    TournamentConfig,
    TournamentSimulator,
    export_metrics_csv,
    summarize_matches,
)
from pokerlab.evaluation.tournament import MatchResult


def test_tournament_conserves_chips_alternates_button_and_repeats() -> None:
    simulator = TournamentSimulator()
    simulator.register("left", lambda: RandomAgent())
    simulator.register("right", lambda: RandomAgent())
    config = TournamentConfig(
        starting_stack=40, small_blind=1, big_blind=2, hand_limit=8
    )
    first = simulator.run_match("left", "right", config=config, seed=17)
    second = simulator.run_match("left", "right", config=config, seed=17)
    assert first.net_chips == second.net_chips
    assert first.ending_stacks == second.ending_stacks
    assert first.hands_played == second.hands_played
    assert first.positions == second.positions
    assert sum(first.ending_stacks.values()) == 80
    assert sum(first.net_chips.values()) == 0
    assert sum(first.positions.values()) == first.hands_played
    assert abs(first.positions["left"] - first.positions["right"]) <= 1
    assert first.hands_played <= 8
    assert first.match_winner in {"left", "right", None}


def test_all_in_agent_match_settles_and_exports_match_metrics(tmp_path) -> None:
    from pokerlab.agents.base import Agent
    from pokerlab.engine import ActionType

    class ShoveAgent(Agent):
        def act(self, observation):
            return next(
                action
                for action in observation.legal_actions
                if action.type
                in (ActionType.ALL_IN, ActionType.CALL, ActionType.CHECK)
            )

    simulator = TournamentSimulator()
    simulator.register("shove1", ShoveAgent)
    simulator.register("shove2", ShoveAgent)
    result = simulator.run_match(
        "shove1",
        "shove2",
        config=TournamentConfig(starting_stack=10, hand_limit=10),
        seed=2,
    )
    assert sum(result.ending_stacks.values()) == 20
    assert result.hands_played <= 10
    summary = summarize_matches([result], agent="shove1", opponent="shove2")
    path = export_metrics_csv([summary], tmp_path / "metrics.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["agent"] == "shove1"
    assert row["hands_played"] == str(result.hands_played)


def test_metric_math_and_benchmark_smoke() -> None:
    results = [
        MatchResult(
            "a",
            "b",
            {"a": 100, "b": 100},
            {"a": 120, "b": 80},
            {"a": 20, "b": -20},
            10,
            {"a": 6, "b": 4},
            0,
            "a",
            {"a": 5, "b": 5},
            1.0,
            1,
        ),
        MatchResult(
            "a",
            "b",
            {"a": 100, "b": 100},
            {"a": 90, "b": 110},
            {"a": -10, "b": 10},
            10,
            {"a": 4, "b": 6},
            0,
            "b",
            {"a": 5, "b": 5},
            1.0,
            2,
        ),
    ]
    metrics = summarize_matches(results, agent="a", opponent="b", big_blind=2)
    assert metrics.net_chips == 10
    assert metrics.average_profit_per_hand == 0.5
    assert metrics.bb_per_100 == 25
    assert metrics.win_rate == 0.5

    runner = BenchmarkRunner(equity_simulations=5, pairings=(("random_a", "random_b"),))
    report = runner.run(
        matches=1, config=TournamentConfig(starting_stack=20, hand_limit=2), seed=4
    )
    assert len(report) == 1
    assert report[0].player1.hands_played <= 2


def test_repeated_matches_alternate_the_initial_button() -> None:
    simulator = TournamentSimulator()
    simulator.register("a", lambda: RandomAgent())
    simulator.register("b", lambda: RandomAgent())
    results = simulator.run_matches(
        "a",
        "b",
        matches=2,
        config=TournamentConfig(starting_stack=100, hand_limit=1),
        seed=31,
    )
    assert results[0].positions["a"] == 1
    assert results[1].positions["b"] == 1
