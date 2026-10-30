"""Command-line entry point for reproducible baseline benchmarks."""

import argparse
from pathlib import Path

from pokerlab.evaluation.benchmark import BenchmarkRunner, flatten_metrics
from pokerlab.evaluation.metrics import export_metrics_csv
from pokerlab.evaluation.tournament import TournamentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=20)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--starting-stack", type=int, default=500)
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--equity-simulations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path, default=Path("results/phase2_benchmark.csv")
    )
    args = parser.parse_args()
    config = TournamentConfig(
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        hand_limit=args.hands,
    )
    report = BenchmarkRunner(equity_simulations=args.equity_simulations).run(
        matches=args.matches,
        config=config,
        seed=args.seed,
    )
    output = export_metrics_csv(flatten_metrics(report), args.output)
    print("Preliminary sample; use more matches/hands for stable comparisons.")
    for matchup in report:
        for metric in (matchup.player1, matchup.player2):
            print(
                f"{metric.agent} vs {metric.opponent}: "
                f"{metric.bb_per_100:.2f} BB/100, "
                f"win rate {metric.win_rate:.3f}, "
                f"{metric.hands_played} hands"
            )
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
