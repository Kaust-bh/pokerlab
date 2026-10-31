"""Small reproducible MCCFR training and agent-evaluation experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pokerlab.agents import EquityAgent, EquityAgentConfig, HeuristicAgent, RandomAgent
from pokerlab.evaluation import TournamentConfig, TournamentSimulator
from pokerlab.research.nlhe.agent import AbstractHUNLAgent, serialize_strategy
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.mccfr import ExternalSamplingMCCFR, HUNLMCCFRConfig


def _git(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _wilson(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def run_experiment(
    *,
    iterations: int = 1,
    seed: int = 7,
    hands: int = 4,
    stack_bb: float = 5.0,
    postflop_method: str = "sampled_strength",
    postflop_buckets: int = 20,
    opponent_hand_samples: int = 128,
    equity_samples: int = 64,
    opponent_range: str = "uniform",
    action_profile: str = "legacy",
    output_root: Path | str = "results/nlhe_abstraction",
) -> Path:
    """Train and evaluate a small run, writing only to a fresh output folder."""
    config = HUNLGameConfig(
        starting_stack_bb=stack_bb,
        postflop_method=postflop_method,
        postflop_buckets=postflop_buckets,
        opponent_hand_samples=opponent_hand_samples,
        equity_samples=equity_samples,
        opponent_range=opponent_range,
        action_profile=action_profile,
    )
    solver = ExternalSamplingMCCFR(HUNLMCCFRConfig(iterations, seed, config))
    train_metrics = solver.train()
    strategy = (
        solver.average_strategy_profile()
        if hasattr(solver, "average_strategy_profile")
        else {info: solver.average_strategy(info) for info in solver.strategy_sums}
    )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = root / f"run-{stamp}-{seed}"
    suffix = 1
    while destination.exists():
        destination = root / f"run-{stamp}-{seed}-{suffix}"
        suffix += 1
    destination.mkdir()

    config_payload = {
        "solver": {
            "type": "external_sampling_mccfr",
            "iterations": iterations,
            "seed": seed,
        },
        "game_and_abstraction": asdict(config),
        "evaluation": {
            "hands_per_match": hands,
            "seed": seed,
            "note": "exploitability is not calculated",
        },
    }
    config_path = destination / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n")
    strategy_path = destination / "average_strategy.json"
    strategy_path.write_text(json.dumps(serialize_strategy(strategy), indent=2) + "\n")
    (destination / "training.json").write_text(
        json.dumps(train_metrics, indent=2) + "\n"
    )
    with (destination / "training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "training_seconds",
                "sampled_nodes",
                "information_sets",
                "cumulative_regret_l1",
                "average_strategy_l1_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(solver.training_records)

    tournament = TournamentSimulator()
    generated_agents = []
    tournament.register(
        "MCCFR",
        lambda: (
            generated_agents.append(AbstractHUNLAgent(strategy, config, seed))
            or generated_agents[-1]
        ),
    )
    tournament.register("Random", lambda: RandomAgent())
    tournament.register("Heuristic", lambda: HeuristicAgent())
    tournament.register(
        "Equity", lambda: EquityAgent(EquityAgentConfig(simulations=32))
    )
    rows = []
    match_seed = seed
    for opponent in ("Random", "Heuristic", "Equity"):
        for match_index in range(2):
            match_seed += 1
            result = tournament.run_match(
                "MCCFR",
                opponent,
                config=TournamentConfig(
                    starting_stack=config.starting_stack_chips,
                    small_blind=config.small_blind_chips,
                    big_blind=config.big_blind_chips,
                    hand_limit=hands,
                    action_profile=config.action_profile,
                ),
                seed=match_seed,
                first_button="MCCFR" if match_index == 0 else opponent,
            )
            agent = generated_agents[-1]
            played = max(result.hands_played, 1)
            bb100 = result.net_chips["MCCFR"] / config.chips_per_bb / played * 100
            rows.append(
                {
                    "opponent": opponent,
                    "match": match_index,
                    "seed": match_seed,
                    "hands": result.hands_played,
                    "net_chips": result.net_chips["MCCFR"],
                    "bb_per_100": bb100,
                    "match_win": int(result.match_winner == "MCCFR"),
                    "runtime_seconds": result.runtime_seconds,
                    "mccfr_decisions": agent.decisions,
                    "mccfr_seconds_per_decision": agent.decision_seconds
                    / max(agent.decisions, 1),
                    "mccfr_decision_seconds": agent.decision_seconds,
                }
            )
    csv_path = destination / "evaluation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison = []
    for opponent in ("Random", "Heuristic", "Equity"):
        selected = [row for row in rows if row["opponent"] == opponent]
        wins = sum(row["match_win"] for row in selected)
        low, high = _wilson(wins, len(selected))
        comparison.append(
            {
                "opponent": opponent,
                "matches": len(selected),
                "match_win_rate": wins / len(selected),
                "match_win_ci95_low": low,
                "match_win_ci95_high": high,
                "mean_bb_per_100": sum(row["bb_per_100"] for row in selected)
                / len(selected),
                "mean_mccfr_seconds_per_decision": sum(
                    row["mccfr_seconds_per_decision"] for row in selected
                )
                / len(selected),
            }
        )
    with (destination / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    files = sorted(path for path in destination.iterdir() if path.is_file())
    source_files = sorted(Path("src/pokerlab/research/nlhe").glob("*.py")) + [
        Path("src/pokerlab/agents/action_profiles.py"),
        Path("src/pokerlab/agents/base.py"),
        Path("src/pokerlab/evaluation/tournament.py"),
    ]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_head": _git(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": _git(["git", "status", "--porcelain"]),
        "artifacts_sha256": {path.name: _hash(path) for path in files},
        "source_sha256": {str(path): _hash(path) for path in source_files},
        "methodology": "two matches per simple-agent baseline; exploratory, not an equilibrium or strength claim",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hands", type=int, default=4)
    parser.add_argument("--stack-bb", type=float, default=5.0)
    parser.add_argument(
        "--postflop-method",
        choices=("sampled_strength", "sampled_equity"),
        default="sampled_strength",
    )
    parser.add_argument("--postflop-buckets", type=int, default=20)
    parser.add_argument("--opponent-hand-samples", type=int, default=128)
    parser.add_argument("--equity-samples", type=int, default=64)
    parser.add_argument(
        "--opponent-range", choices=("uniform", "premium"), default="uniform"
    )
    parser.add_argument(
        "--action-profile", choices=("legacy", "standard", "wide"), default="legacy"
    )
    args = parser.parse_args()
    print(
        run_experiment(
            iterations=args.iterations,
            seed=args.seed,
            hands=args.hands,
            stack_bb=args.stack_bb,
            postflop_method=args.postflop_method,
            postflop_buckets=args.postflop_buckets,
            opponent_hand_samples=args.opponent_hand_samples,
            equity_samples=args.equity_samples,
            opponent_range=args.opponent_range,
            action_profile=args.action_profile,
        )
    )


if __name__ == "__main__":
    main()
