"""Repeatable baseline-agent comparisons."""

from collections.abc import Sequence
from dataclasses import dataclass

from pokerlab.agents.equity_agent import EquityAgent, EquityAgentConfig
from pokerlab.agents.heuristic_agent import HeuristicAgent
from pokerlab.agents.random_agent import RandomAgent
from pokerlab.evaluation.metrics import AgentMetrics, summarize_matches
from pokerlab.evaluation.tournament import (
    MatchResult,
    TournamentConfig,
    TournamentSimulator,
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    player1: AgentMetrics
    player2: AgentMetrics
    matches: tuple[MatchResult, ...]


DEFAULT_PAIRINGS = (
    ("random_a", "random_b"),
    ("heuristic", "random"),
    ("equity", "random"),
    ("equity", "heuristic"),
    ("heuristic_a", "heuristic_b"),
)


class BenchmarkRunner:
    """Runs a configurable set of matchups; results remain preliminary samples."""

    def __init__(
        self,
        *,
        equity_simulations: int = 150,
        pairings: Sequence[tuple[str, str]] = DEFAULT_PAIRINGS,
    ) -> None:
        if equity_simulations < 1:
            raise ValueError("equity_simulations must be positive")
        self.simulator = TournamentSimulator()
        self.simulator.register("random_a", lambda: RandomAgent())
        self.simulator.register("random_b", lambda: RandomAgent())
        self.simulator.register("random", lambda: RandomAgent())
        self.simulator.register("heuristic", HeuristicAgent)
        self.simulator.register("heuristic_a", HeuristicAgent)
        self.simulator.register("heuristic_b", HeuristicAgent)
        self.simulator.register(
            "equity",
            lambda: EquityAgent(EquityAgentConfig(simulations=equity_simulations)),
        )
        self.pairings = tuple(pairings)

    def run(
        self,
        *,
        matches: int = 20,
        config: TournamentConfig | None = None,
        seed: int = 0,
    ) -> list[BenchmarkResult]:
        settings = config or TournamentConfig()
        results: list[BenchmarkResult] = []
        for index, (player1, player2) in enumerate(self.pairings):
            match_results = self.simulator.run_matches(
                player1,
                player2,
                matches=matches,
                config=settings,
                seed=seed + index,
            )
            results.append(
                BenchmarkResult(
                    summarize_matches(
                        match_results,
                        agent=player1,
                        opponent=player2,
                        big_blind=settings.big_blind,
                    ),
                    summarize_matches(
                        match_results,
                        agent=player2,
                        opponent=player1,
                        big_blind=settings.big_blind,
                    ),
                    tuple(match_results),
                )
            )
        return results


def flatten_metrics(results: Sequence[BenchmarkResult]) -> list[AgentMetrics]:
    return [metric for result in results for metric in (result.player1, result.player2)]
