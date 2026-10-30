"""Match-level performance summaries and CSV export."""

import csv
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pokerlab.evaluation.tournament import MatchResult


@dataclass(frozen=True, slots=True)
class AgentMetrics:
    agent: str
    opponent: str
    matches: int
    hands_played: int
    match_wins: int
    match_losses: int
    match_ties: int
    win_rate: float
    win_rate_ci: tuple[float, float]
    net_chips: float
    average_profit_per_hand: float
    profit_per_hand_stddev: float
    bb_per_100: float
    bb_per_100_ci: tuple[float, float]
    runtime_per_hand_seconds: float


def summarize_matches(
    results: Sequence[MatchResult],
    *,
    agent: str,
    opponent: str,
    big_blind: int = 2,
    confidence_z: float = 1.96,
) -> AgentMetrics:
    if not results or big_blind <= 0:
        raise ValueError("at least one result and a positive big blind are required")
    if confidence_z <= 0:
        raise ValueError("confidence z-score must be positive")
    if any(
        agent not in result.net_chips or opponent not in result.net_chips
        for result in results
    ):
        raise ValueError("match results do not contain the requested players")
    hands = sum(result.hands_played for result in results)
    net = sum(result.net_chips[agent] for result in results)
    per_match = [
        result.net_chips[agent] / max(1, result.hands_played) for result in results
    ]
    average = statistics.fmean(per_match)
    stddev = statistics.stdev(per_match) if len(per_match) > 1 else 0.0
    margin = confidence_z * stddev / math.sqrt(len(per_match))
    wins = sum(result.match_winner == agent for result in results)
    losses = sum(result.match_winner == opponent for result in results)
    ties = len(results) - wins - losses
    outcomes = [
        1.0
        if result.match_winner == agent
        else 0.5
        if result.match_winner is None
        else 0.0
        for result in results
    ]
    win_rate = statistics.fmean(outcomes)
    win_stddev = statistics.stdev(outcomes) if len(outcomes) > 1 else 0.0
    win_margin = confidence_z * win_stddev / math.sqrt(len(outcomes))
    bb100 = average / big_blind * 100
    bb100_margin = margin / big_blind * 100
    runtime = sum(result.runtime_seconds for result in results)
    return AgentMetrics(
        agent,
        opponent,
        len(results),
        hands,
        wins,
        losses,
        ties,
        win_rate,
        (max(0.0, win_rate - win_margin), min(1.0, win_rate + win_margin)),
        net,
        average,
        stddev,
        bb100,
        (bb100 - bb100_margin, bb100 + bb100_margin),
        runtime / hands if hands else 0.0,
    )


def export_metrics_csv(metrics: Sequence[AgentMetrics], path: str | Path) -> Path:
    """Write one agent-versus-opponent summary per CSV row."""
    if not metrics:
        raise ValueError("at least one metrics row is required")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in metrics:
        row = asdict(metric)
        row["win_rate_ci"] = f"{metric.win_rate_ci[0]:.6f}:{metric.win_rate_ci[1]:.6f}"
        row["bb_per_100_ci"] = (
            f"{metric.bb_per_100_ci[0]:.6f}:{metric.bb_per_100_ci[1]:.6f}"
        )
        rows.append(row)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target
