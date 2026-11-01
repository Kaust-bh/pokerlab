"""Paired, checkpointed abstraction benchmarks for abstracted HUNL MCCFR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from pokerlab.agents import EquityAgent, EquityAgentConfig, HeuristicAgent, RandomAgent
from pokerlab.agents.base import Agent
from pokerlab.evaluation import TournamentConfig, TournamentSimulator
from pokerlab.research.nlhe.actions import abstract_action_specs_from_observation
from pokerlab.research.nlhe.agent import (
    AbstractHUNLAgent,
    deserialize_strategy,
    serialize_strategy,
)
from pokerlab.research.nlhe.analysis import write_analysis
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.mccfr import ExternalSamplingMCCFR, HUNLMCCFRConfig

VARIANT_NAMES = (
    "baseline",
    "equity_8",
    "equity_20",
    "action_wide",
    "combined_wide",
    "isomorphic",
)
BASELINES = ("Random", "Heuristic", "Equity")


@dataclass(frozen=True, slots=True)
class BenchmarkPlan:
    seeds: tuple[int, ...] = (11, 23, 47, 89)
    iterations: int = 20
    checkpoints: tuple[int, ...] = (5, 10, 20)
    evaluation_hands: int = 100
    stack_bb: float = 10.0
    equity_samples: int = 16
    equity_agent_simulations: int = 32

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.iterations < 1 or self.evaluation_hands < 1:
            raise ValueError("iterations and evaluation_hands must be positive")
        if self.equity_samples < 1 or self.equity_agent_simulations < 1:
            raise ValueError("equity sample counts must be positive")
        points = tuple(sorted({*self.checkpoints, self.iterations}))
        if any(point < 1 or point > self.iterations for point in points):
            raise ValueError("checkpoints must be within the training budget")
        object.__setattr__(self, "checkpoints", points)


def variant_config(name: str, plan: BenchmarkPlan) -> HUNLGameConfig:
    common = {
        "starting_stack_bb": plan.stack_bb,
        "equity_samples": plan.equity_samples,
    }
    if name == "baseline":
        return HUNLGameConfig(**common)
    if name == "equity_8":
        return HUNLGameConfig(
            **common,
            postflop_method="sampled_equity",
            postflop_buckets=8,
        )
    if name == "equity_20":
        return HUNLGameConfig(
            **common,
            postflop_method="sampled_equity",
        )
    if name == "action_wide":
        return HUNLGameConfig(**common, action_profile="wide")
    if name == "combined_wide":
        return HUNLGameConfig(
            **common,
            postflop_method="sampled_equity",
            action_profile="wide",
        )
    if name == "isomorphic":
        return HUNLGameConfig(
            **common,
            postflop_method="sampled_equity",
            board_abstraction="isomorphic",
        )
    raise ValueError(f"unknown variant {name!r}")


def estimate_plan(plan: BenchmarkPlan, variants: tuple[str, ...]) -> dict:
    checkpoints = len(plan.checkpoints)
    trained_matches = len(variants) * len(plan.seeds) * checkpoints * len(BASELINES) * 2
    previous_matches = len(variants) * len(plan.seeds) * max(checkpoints - 1, 0) * 2
    total_matches = trained_matches + previous_matches
    max_hands = total_matches * plan.evaluation_hands
    training_iterations = len(variants) * len(plan.seeds) * plan.iterations
    # Conservative planning rates from local smoke profiling: 0.1s/training
    # iteration and 0.02s/equity-agent hand, 0.002s/other-agent hand.
    estimated_training_seconds = training_iterations * 0.1
    estimated_evaluation_seconds = max_hands * (0.02 / 3 + 0.002 * 2 / 3)
    return {
        "variants": variants,
        "seeds": len(plan.seeds),
        "training_iterations": training_iterations,
        "checkpoints_per_configuration": checkpoints,
        "evaluation_matches_upper_bound": total_matches,
        "evaluation_hands_upper_bound": max_hands,
        "estimated_training_seconds": estimated_training_seconds,
        "estimated_evaluation_seconds": estimated_evaluation_seconds,
        "estimated_total_seconds": estimated_training_seconds
        + estimated_evaluation_seconds,
        "memory_note": "Strategies/checkpoints are processed one variant-seed pair at a time; table memory scales with that pair, not the full suite.",
    }


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _new_run_dir(root: Path, run_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = root / f"run-{stamp}-{run_id}"
    path = base
    suffix = 1
    while path.exists():
        path = root / f"{base.name}-{suffix}"
        suffix += 1
    path.mkdir()
    return path


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stable_match_seed(seed: int, checkpoint: int, opponent: str, position: str) -> int:
    payload = f"{seed}|{checkpoint}|{opponent}|{position}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _peak_rss_bytes() -> int | None:
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError):
        return None


def _checkpoint_directory(
    run_dir: Path, variant: str, seed: int, iteration: int
) -> Path:
    path = (
        run_dir
        / "checkpoints"
        / variant
        / f"seed-{seed}"
        / f"iteration-{iteration:06d}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save_checkpoint(run_dir, solver, config, variant, seed, iteration, resumed_from):
    directory = _checkpoint_directory(run_dir, variant, seed, iteration)
    strategy_path = directory / "strategy.json"
    strategy_path.write_text(
        json.dumps(serialize_strategy(solver.average_strategy_profile()), indent=2)
        + "\n"
    )
    state_path = directory / "checkpoint.json"
    state_path.write_text(json.dumps(solver.checkpoint_state(), indent=2) + "\n")
    metadata = {
        "variant": variant,
        "seed": seed,
        "iteration": iteration,
        "created_utc": datetime.now(UTC).isoformat(),
        "game_and_abstraction": asdict(config),
        "diagnostics": solver.diagnostics(),
        "training_seconds_cumulative": sum(
            float(row["training_seconds"]) for row in solver.training_records
        ),
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "resumed_from": resumed_from,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return directory


def _make_opponent_factory(name: str, strategy, config, equity_simulations):
    if name == "Random":
        return lambda: RandomAgent()
    if name == "Heuristic":
        return lambda: HeuristicAgent()
    if name == "Equity":
        return lambda: EquityAgent(EquityAgentConfig(simulations=equity_simulations))
    if name == "PreviousMCCFR":
        return lambda: AbstractHUNLAgent(strategy, config)
    raise ValueError(f"unknown opponent {name!r}")


class _LegacyMenuAgent(Agent):
    """Give baseline policies a fixed legacy action menu in every variant."""

    def __init__(self, wrapped: Agent) -> None:
        self.wrapped = wrapped

    def reset(self, seed: int | None = None) -> None:
        self.wrapped.reset(seed)

    def _legacy_observation(self, observation):
        legacy_observation = replace(observation, action_profile="legacy")
        legacy_actions = tuple(
            spec.engine
            for spec in abstract_action_specs_from_observation(legacy_observation)
        )
        if not legacy_actions:
            raise RuntimeError("legacy action menu unexpectedly empty")
        return replace(legacy_observation, legal_actions=legacy_actions)

    def act(self, observation):
        return self.wrapped.act(self._legacy_observation(observation))

    def action_probabilities(self, observation):
        return self.wrapped.action_probabilities(self._legacy_observation(observation))


def _evaluate_one(
    *,
    strategy,
    previous_strategy,
    game_config: HUNLGameConfig,
    variant: str,
    seed: int,
    checkpoint: int,
    opponent: str,
    opponent_checkpoint: int | None,
    first_button_role: str,
    plan: BenchmarkPlan,
    strategy_sums=None,
    previous_strategy_sums=None,
) -> dict:
    tournament = TournamentSimulator()
    generated_agents = []
    tournament.register(
        "Trained",
        lambda: (
            generated_agents.append(
                AbstractHUNLAgent(strategy, game_config, strategy_sums=strategy_sums)
            )
            or generated_agents[-1]
        ),
    )
    competitor_strategy = previous_strategy if opponent == "PreviousMCCFR" else None
    opponent_factory = _make_opponent_factory(
        opponent,
        competitor_strategy,
        game_config,
        plan.equity_agent_simulations,
    )
    if opponent != "PreviousMCCFR":
        tournament.register(
            opponent,
            lambda: _LegacyMenuAgent(opponent_factory()),
        )
    else:
        tournament.register(
            opponent,
            lambda: AbstractHUNLAgent(
                competitor_strategy,
                game_config,
                strategy_sums=previous_strategy_sums,
            ),
        )
    first_button = "Trained" if first_button_role == "trained" else opponent
    match_seed = _stable_match_seed(seed, checkpoint, opponent, first_button_role)
    game_settings = TournamentConfig(
        starting_stack=game_config.starting_stack_chips,
        small_blind=game_config.small_blind_chips,
        big_blind=game_config.big_blind_chips,
        hand_limit=plan.evaluation_hands,
        action_profile=game_config.action_profile,
    )
    started = time.perf_counter()
    result = tournament.run_fixed_hands(
        "Trained",
        opponent,
        hands=plan.evaluation_hands,
        config=game_settings,
        seed=match_seed,
        button=first_button,
    )
    wall = time.perf_counter() - started
    trained = generated_agents[-1]
    rates = trained.lookup_rates()
    net = result.net_chips["Trained"]
    hands = result.hands_played
    if sum(result.net_chips.values()) != 0:
        raise RuntimeError("evaluation match violated chip conservation")
    return {
        "variant": variant,
        "seed": seed,
        "checkpoint_iteration": checkpoint,
        "opponent": opponent,
        "opponent_checkpoint": opponent_checkpoint,
        "trained_first_button": first_button_role,
        "trained_button_hands": result.positions["Trained"],
        "opponent_button_hands": result.positions[opponent],
        "match_seed": match_seed,
        "hands_played": hands,
        "net_chips": net,
        "bb_per_100": net / game_config.chips_per_bb / max(hands, 1) * 100,
        "match_win": int(net > 0),
        "match_tie": int(net == 0),
        "hand_wins": result.hand_wins["Trained"],
        "hand_ties": result.hand_ties,
        "runtime_seconds": wall,
        "trained_decisions": trained.decisions,
        "exact_hits": rates["exact_hits"],
        "generalized_hits": rates["generalized_hits"],
        "fallback_hits": rates["fallback_hits"],
        "exact_lookup_rate": rates["exact_rate"],
        "generalized_lookup_rate": rates["generalized_rate"],
        "fallback_lookup_rate": rates["fallback_rate"],
        "trained_seconds_per_decision": trained.decision_seconds
        / max(trained.decisions, 1),
    }


def run_benchmark(
    plan: BenchmarkPlan | None = None,
    *,
    variants: tuple[str, ...] = VARIANT_NAMES,
    output_root: Path | str = "results/nlhe_abstraction",
    resume_checkpoint: Path | str | None = None,
) -> Path:
    benchmark_started = time.perf_counter()
    plan = plan or BenchmarkPlan()
    if not variants or any(name not in VARIANT_NAMES for name in variants):
        raise ValueError("variants must be selected from the built-in comparison suite")
    if resume_checkpoint is not None and (len(variants) != 1 or len(plan.seeds) != 1):
        raise ValueError("resume requires exactly one selected variant and one seed")
    resume_payload = None
    resume_seed = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        resume_payload = json.loads(resume_path.read_text())
        resume_seed = int(resume_payload["seed"])
        if plan.seeds != (resume_seed,):
            raise ValueError("plan seed must match resume checkpoint seed")

    config_data = {
        "algorithm": "external_sampling_mccfr",
        "plan": asdict(plan),
        "variants": {name: asdict(variant_config(name, plan)) for name in variants},
        "evaluation_opponents": [*BASELINES, "PreviousMCCFR"],
        "evaluation_position_policy": "paired fixed-hand batches with the trained agent and opponent each holding the button for one batch; fresh equal stacks and a fixed hand count per batch",
        "baseline_action_menu_policy": "Random, Heuristic and Equity opponents receive the legacy action menu across all variants",
        "evaluation_note": "No exact full-game exploitability is calculated.",
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }
    run_digest = hashlib.sha256(
        json.dumps(config_data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    run_dir = _new_run_dir(Path(output_root), run_digest)
    for child in ("checkpoints", "logs", "plots"):
        (run_dir / child).mkdir()
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config_data, indent=2) + "\n")
    log_path = run_dir / "logs" / "run.jsonl"

    def log_event(event: str, **fields):
        with log_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "time_utc": datetime.now(UTC).isoformat(),
                        "event": event,
                        **fields,
                    }
                )
                + "\n"
            )

    training_rows = []
    checkpoint_dirs: dict[tuple[str, int, int], Path] = {}
    resumed_from = str(resume_checkpoint) if resume_checkpoint else None
    for variant in variants:
        game_config = variant_config(variant, plan)
        for seed in plan.seeds:
            solver = ExternalSamplingMCCFR(
                HUNLMCCFRConfig(plan.iterations, seed, game_config)
            )
            start_at = 0
            if (
                resume_payload is not None
                and variant == variants[0]
                and seed == plan.seeds[0]
            ):
                solver.load_checkpoint(resume_payload)
                start_at = solver.iterations
                if start_at > plan.iterations:
                    raise ValueError(
                        "resume checkpoint is past requested iteration target"
                    )
            targets = sorted({*plan.checkpoints, *([start_at] if start_at else [])})
            if start_at and start_at not in checkpoint_dirs:
                checkpoint_dirs[(variant, seed, start_at)] = _save_checkpoint(
                    run_dir,
                    solver,
                    game_config,
                    variant,
                    seed,
                    start_at,
                    resumed_from,
                )
            for target in targets:
                if target <= solver.iterations:
                    continue
                solver.train(target - solver.iterations)
                for record in solver.training_records:
                    if record["iteration"] > start_at and not any(
                        row["variant"] == variant
                        and row["seed"] == seed
                        and row["iteration"] == record["iteration"]
                        for row in training_rows
                    ):
                        training_rows.append(
                            {"variant": variant, "seed": seed, **record}
                        )
                if target in plan.checkpoints:
                    checkpoint_dirs[(variant, seed, target)] = _save_checkpoint(
                        run_dir,
                        solver,
                        game_config,
                        variant,
                        seed,
                        target,
                        resumed_from,
                    )
                    log_event(
                        "checkpoint_saved",
                        variant=variant,
                        seed=seed,
                        iteration=target,
                        information_sets=len(solver.regrets),
                    )
            checkpoint_key = (variant, seed, solver.iterations)
            if (
                checkpoint_key not in checkpoint_dirs
                and solver.iterations in plan.checkpoints
            ):
                checkpoint_dirs[(variant, seed, solver.iterations)] = _save_checkpoint(
                    run_dir,
                    solver,
                    game_config,
                    variant,
                    seed,
                    solver.iterations,
                    resumed_from,
                )
    _write_csv(run_dir / "training.csv", training_rows)

    evaluation_rows = []
    checkpoints_by_variant_seed = {}
    for variant in variants:
        for seed in plan.seeds:
            points = sorted(
                checkpoint
                for candidate, candidate_seed, checkpoint in checkpoint_dirs
                if candidate == variant and candidate_seed == seed
            )
            checkpoints_by_variant_seed[(variant, seed)] = points
            previous_point = None
            previous_strategy = None
            previous_sums = None
            for checkpoint in points:
                directory = checkpoint_dirs[(variant, seed, checkpoint)]
                strategy = deserialize_strategy(
                    json.loads((directory / "strategy.json").read_text())
                )
                ckpt_payload = json.loads((directory / "checkpoint.json").read_text())
                strategy_sums = deserialize_strategy(
                    {
                        "information_sets": [
                            {"info": row["info"], "strategy": row["values"]}
                            for row in ckpt_payload.get("strategy_sums", [])
                        ]
                    }
                )
                game_config = variant_config(variant, plan)
                for opponent in BASELINES:
                    for position in ("trained", "opponent"):
                        evaluation_rows.append(
                            _evaluate_one(
                                strategy=strategy,
                                previous_strategy=None,
                                game_config=game_config,
                                variant=variant,
                                seed=seed,
                                checkpoint=checkpoint,
                                opponent=opponent,
                                opponent_checkpoint=None,
                                first_button_role=position,
                                plan=plan,
                                strategy_sums=strategy_sums,
                            )
                        )
                if previous_point is not None and previous_strategy is not None:
                    for position in ("trained", "opponent"):
                        evaluation_rows.append(
                            _evaluate_one(
                                strategy=strategy,
                                previous_strategy=previous_strategy,
                                game_config=game_config,
                                variant=variant,
                                seed=seed,
                                checkpoint=checkpoint,
                                opponent="PreviousMCCFR",
                                opponent_checkpoint=previous_point,
                                first_button_role=position,
                                plan=plan,
                                strategy_sums=strategy_sums,
                                previous_strategy_sums=previous_sums,
                            )
                        )
                previous_point = checkpoint
                previous_strategy = strategy
                previous_sums = strategy_sums
            log_event(
                "evaluated_variant_seed",
                variant=variant,
                seed=seed,
                checkpoints=points,
            )
    _write_csv(run_dir / "evaluation.csv", evaluation_rows)
    write_analysis(run_dir, evaluation_rows)
    repo_root = Path(__file__).resolve().parents[4]
    git_head = _git(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    summary = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "variants": list(variants),
        "seeds": list(plan.seeds),
        "training_iterations_completed": len(training_rows),
        "evaluation_matches_completed": len(evaluation_rows),
        "evaluation_seconds_sum_of_batch_wall_times": sum(
            float(row["runtime_seconds"]) for row in evaluation_rows
        ),
        "evaluation_hands_completed": sum(
            int(row["hands_played"]) for row in evaluation_rows
        ),
        "training_seconds_sum_of_iteration_wall_times": sum(
            float(row["training_seconds"]) for row in training_rows
        ),
        "experiment_wall_seconds": time.perf_counter() - benchmark_started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "exact_exploitability_available": False,
        "resume_checkpoint": resumed_from,
        "source_revision": git_head,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    src_root = (
        Path("src/pokerlab")
        if Path("src/pokerlab").exists()
        else repo_root / "src/pokerlab"
    )
    source_files = sorted(src_root.rglob("*.py"))
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_head": git_head,
        "git_status_porcelain": _git(
            ["git", "-C", str(repo_root), "status", "--porcelain"]
        ),
        "platform": platform.platform(),
        "python": sys.version,
        "artifact_sha256": {
            str(path.relative_to(run_dir)): _hash(path) for path in files
        },
        "source_sha256": {
            str(path.relative_to(repo_root) if path.is_absolute() else path): _hash(
                path
            )
            for path in source_files
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47, 89])
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANT_NAMES, default=list(VARIANT_NAMES)
    )
    parser.add_argument("--evaluation-hands", type=int, default=100)
    parser.add_argument("--stack-bb", type=float, default=10.0)
    parser.add_argument("--equity-samples", type=int, default=16)
    parser.add_argument("--equity-agent-simulations", type=int, default=32)
    parser.add_argument("--output-root", default="results/nlhe_abstraction")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    plan = BenchmarkPlan(
        seeds=tuple(args.seeds),
        iterations=args.iterations,
        checkpoints=tuple(args.checkpoints),
        evaluation_hands=args.evaluation_hands,
        stack_bb=args.stack_bb,
        equity_samples=args.equity_samples,
        equity_agent_simulations=args.equity_agent_simulations,
    )
    if args.estimate_only:
        print(json.dumps(estimate_plan(plan, tuple(args.variants)), indent=2))
        return
    print(
        run_benchmark(
            plan,
            variants=tuple(args.variants),
            output_root=args.output_root,
            resume_checkpoint=args.resume_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
