"""Variance and multi-seed investigation of the PokerLab MCCFR solvers.

Long Leduc vanilla CFR runs are intentionally not launched by default. Existing
vanilla 1k and 5k artifacts are reused where available; new Leduc vanilla runs
stop at 1k because the recorded 1k cost is about 68 seconds and the measured
5k run took several minutes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pokerlab.research.kuhn.cfr import CFRConfig as KuhnCFRConfig
from pokerlab.research.kuhn.cfr import CFRSolver as KuhnCFRSolver
from pokerlab.research.kuhn.evaluation import evaluate_nash as evaluate_kuhn
from pokerlab.research.kuhn.game import KuhnAction, KuhnGame
from pokerlab.research.kuhn.information_sets import InformationSet as KuhnInfo
from pokerlab.research.kuhn.mccfr import (
    MCCFRConfig as KuhnMCCFRConfig,
)
from pokerlab.research.kuhn.mccfr import (
    MCCFRSolver as KuhnMCCFRSolver,
)
from pokerlab.research.kuhn.strategy import (
    load_strategy_snapshot as load_kuhn_snapshot,
)
from pokerlab.research.kuhn.strategy import (
    save_strategy_snapshot as save_kuhn_snapshot,
)
from pokerlab.research.leduc.betting import Action
from pokerlab.research.leduc.cfr import CFRConfig as LeducCFRConfig
from pokerlab.research.leduc.cfr import CFRSolver as LeducCFRSolver
from pokerlab.research.leduc.evaluation import evaluate_nash as evaluate_leduc
from pokerlab.research.leduc.game import LeducGame
from pokerlab.research.leduc.information_sets import InformationSet as LeducInfo
from pokerlab.research.leduc.mccfr import (
    MCCFRConfig as LeducMCCFRConfig,
)
from pokerlab.research.leduc.mccfr import (
    MCCFRSolver as LeducMCCFRSolver,
)
from pokerlab.research.leduc.strategy import (
    load_strategy_snapshot as load_leduc_snapshot,
)
from pokerlab.research.leduc.strategy import (
    save_strategy_snapshot as save_leduc_snapshot,
)

KUHN_PROFILE = Mapping[KuhnInfo, Mapping[KuhnAction, float]]
LEDUC_PROFILE = Mapping[LeducInfo, Mapping[Action, float]]
ROW_FIELDS = (
    "game",
    "algorithm",
    "seed",
    "iteration",
    "expected_game_value_player0",
    "expected_game_value_player1",
    "nashconv",
    "exploitability",
    "training_elapsed_seconds",
    "evaluation_elapsed_seconds",
    "iterations_per_second",
    "information_sets",
    "sampled_nodes_last_iteration",
    "source",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        return {"head": head, "dirty": bool(status), "status_porcelain": status}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None, "status_porcelain": []}


def _independent_exact_updates(game: str, profile, targets: set[Any]):
    """Compute exact CFR deltas and external reach without calling CFR code."""
    deltas: dict[Any, dict[Any, float]] = {}
    external_reach: dict[Any, float] = defaultdict(float)

    def visit(state, reach: tuple[float, float], chance_reach: float) -> float:
        if state.is_terminal:
            return float(state.utility(0))
        if game == "leduc" and state.is_chance_node:
            value = 0.0
            for card, probability in state.chance_outcomes:
                value += probability * visit(
                    state.deal_public(card),
                    reach,
                    chance_reach * probability,
                )
            return value

        info = (
            KuhnInfo.from_state(state)
            if game == "kuhn"
            else LeducInfo.from_state(state)
        )
        player = state.current_player
        actions = state.legal_actions
        action_values = {}
        node_value = 0.0
        for action in actions:
            next_reach = list(reach)
            next_reach[player] *= profile[info][action]
            action_values[action] = visit(
                state.act(action),
                (next_reach[0], next_reach[1]),
                chance_reach,
            )
            node_value += profile[info][action] * action_values[action]

        if info in targets:
            external_reach[info] += chance_reach * reach[1 - player]
            sign = 1.0 if player == 0 else -1.0
            updates = deltas.setdefault(info, dict.fromkeys(actions, 0.0))
            for action in actions:
                updates[action] += (
                    chance_reach
                    * reach[1 - player]
                    * sign
                    * (action_values[action] - node_value)
                )
        return node_value

    if game == "kuhn":
        deals = tuple(KuhnGame.deals())
    else:
        deals = LeducGame.deals()
    for deal in deals:
        chance = 1.0 / len(deals)
        state = (
            KuhnGame.initial_state(deal)
            if game == "kuhn"
            else LeducGame.initial_state(deal)
        )
        visit(state, (1.0, 1.0), chance)
    return deltas, dict(external_reach)


def _target_infos(game: str, profile) -> list[tuple[str, Any]]:
    identifiers = (
        {
            "P0 J root": "p0|J|root",
            "P1 J facing preflop bet": "p1|J|bet",
            "P0 Q facing bet after check": "p0|Q|check/bet",
        }
        if game == "kuhn"
        else {
            "P0 J0 preflop root": "p0|J0|-|r0|root|-",
            "P1 Q0 facing preflop bet": "p1|Q0|-|r0|bet|-",
            "P1 J0/J1 flop entry": "p1|J0|J1|r1|check/check|-",
            "P0 J0/K0 facing flop bet": "p0|J0|K0|r1|check/check|bet",
            "P1 J0/J1 facing flop raise": "p1|J0|J1|r1|check/check|bet/raise",
        }
    )
    by_id = {info.identifier: info for info in profile}
    missing = [
        identifier for identifier in identifiers.values() if identifier not in by_id
    ]
    if missing:
        raise ValueError(f"missing diagnostic information sets: {missing}")
    return [(name, by_id[identifier]) for name, identifier in identifiers.items()]


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return center - half / denominator, center + half / denominator


def _variance_for_game(
    game: str,
    profile,
    snapshot_source: Path,
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    solver = (
        KuhnMCCFRSolver(KuhnMCCFRConfig(iterations=1, seed=seed))
        if game == "kuhn"
        else LeducMCCFRSolver(LeducMCCFRConfig(iterations=1, seed=seed))
    )
    targets = _target_infos(game, profile)
    target_map = dict(targets)
    reference, reach = _independent_exact_updates(
        game, profile, set(target_map.values())
    )
    output = []
    for target_index, (label, info) in enumerate(targets):
        traverser = info.player
        solver._rng.seed(seed + target_index * 100_003)
        actions = solver._legal[info]
        sums = dict.fromkeys(actions, 0.0)
        squares = dict.fromkeys(actions, 0.0)
        observed = 0
        for sample in range(trials):
            deals = tuple(KuhnGame.deals()) if game == "kuhn" else LeducGame.deals()
            deal = solver._rng.choice(deals)
            state = (
                KuhnGame.initial_state(deal)
                if game == "kuhn"
                else LeducGame.initial_state(deal)
            )
            delta = {}
            solver._traverse_external(state, traverser, profile, delta)
            updates = delta.get(info)
            if updates is not None:
                observed += 1
            for action in actions:
                value = 0.0 if updates is None else updates[action]
                sums[action] += value
                squares[action] += value * value
        exact = reference.get(info, dict.fromkeys(actions, 0.0))
        sampling_probability = reach.get(info, 0.0)
        coverage_low, coverage_high = _wilson_interval(observed, trials)
        for action in actions:
            mean = sums[action] / trials
            variance = max(
                0.0,
                (squares[action] - sums[action] * sums[action] / trials)
                / max(1, trials - 1),
            )
            standard_error = math.sqrt(variance / trials)
            output.append(
                {
                    "game": game,
                    "traverser": traverser,
                    "information_set": info.identifier,
                    "label": label,
                    "action": action.value,
                    "fixed_profile_source": str(snapshot_source),
                    "diagnostic_seed": seed + target_index * 100_003,
                    "trials": trials,
                    "visits": observed,
                    "visit_rate": observed / trials,
                    "visit_rate_ci95_low": coverage_low,
                    "visit_rate_ci95_high": coverage_high,
                    "exact_counterfactual_reach": sampling_probability,
                    "sampled_mean_regret": mean,
                    "sampled_variance": variance,
                    "sampled_standard_deviation": math.sqrt(variance),
                    "standard_error_of_mean": standard_error,
                    "mean_ci95_low": mean - 1.96 * standard_error,
                    "mean_ci95_high": mean + 1.96 * standard_error,
                    "independent_exact_regret": exact[action],
                    "standardized_error": (
                        (mean - exact[action]) / standard_error
                        if standard_error
                        else None
                    ),
                    "zero_increment_fraction": 1 - observed / trials,
                }
            )
    return output


def _run_solver(
    game: str,
    algorithm: str,
    seed: int,
    iterations: int,
    checkpoints: tuple[int, ...],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], Any]:
    interval = min(b - a for a, b in zip((0, *checkpoints[:-1]), checkpoints))
    if game == "kuhn" and algorithm == "vanilla_cfr":
        solver = KuhnCFRSolver(KuhnCFRConfig(iterations, seed, interval, interval))
    elif game == "kuhn":
        solver = KuhnMCCFRSolver(KuhnMCCFRConfig(iterations, seed, interval, interval))
    elif algorithm == "vanilla_cfr":
        solver = LeducCFRSolver(LeducCFRConfig(iterations, seed, interval, interval))
    else:
        solver = LeducMCCFRSolver(
            LeducMCCFRConfig(iterations, seed, interval, interval)
        )

    rows = []
    training_elapsed = 0.0
    evaluation_elapsed = 0.0
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        solver.run_iteration()
        training_elapsed += time.perf_counter() - started
        if iteration not in checkpoints:
            continue
        profile = solver.average_strategy_profile()
        started = time.perf_counter()
        result = evaluate_kuhn(profile) if game == "kuhn" else evaluate_leduc(profile)
        evaluation_elapsed += time.perf_counter() - started
        rows.append(
            {
                "game": game,
                "algorithm": algorithm,
                "seed": seed,
                "iteration": iteration,
                "expected_game_value_player0": result.expected_value_player0,
                "expected_game_value_player1": -result.expected_value_player0,
                "nashconv": result.nashconv,
                "exploitability": result.exploitability,
                "training_elapsed_seconds": training_elapsed,
                "evaluation_elapsed_seconds": evaluation_elapsed,
                "iterations_per_second": iteration / training_elapsed,
                "information_sets": len(profile),
                "sampled_nodes_last_iteration": getattr(
                    solver, "sampled_nodes", getattr(solver, "game_tree_nodes", None)
                ),
                "source": "new_run",
            }
        )

    profile = solver.average_strategy_profile()
    path = (
        output_dir / f"strategy_{game}_{algorithm}_seed_{seed}_iter_{iterations}.json"
    )
    (save_kuhn_snapshot if game == "kuhn" else save_leduc_snapshot)(profile, path)
    return rows, solver


def _read_pilot_rows(
    run_dir: Path, game: str, algorithm: str, seed: int
) -> list[dict[str, Any]]:
    rows = []
    with (run_dir / "convergence.csv").open(newline="", encoding="utf-8") as stream:
        for source_row in csv.DictReader(stream):
            if (
                source_row["game"] != game
                or source_row["algorithm"] != algorithm
                or int(source_row["seed"]) != seed
            ):
                continue
            rows.append(
                {
                    "game": game,
                    "algorithm": algorithm,
                    "seed": seed,
                    "iteration": int(source_row["iteration"]),
                    "expected_game_value_player0": float(
                        source_row["expected_game_value_player0"]
                    ),
                    "expected_game_value_player1": float(
                        source_row["expected_game_value_player1"]
                    ),
                    "nashconv": float(source_row["nashconv"]),
                    "exploitability": float(source_row["exploitability"]),
                    "training_elapsed_seconds": float(
                        source_row["training_elapsed_seconds"]
                    ),
                    "evaluation_elapsed_seconds": float(
                        source_row["evaluation_elapsed_seconds"]
                    ),
                    "iterations_per_second": float(source_row["iterations_per_second"]),
                    "information_sets": int(source_row["information_sets"]),
                    "sampled_nodes_last_iteration": source_row["nodes_last_iteration"],
                    "source": str(run_dir / "convergence.csv"),
                }
            )
    return rows


def _reuse_leduc_five_k(output_dir: Path) -> list[dict[str, Any]]:
    source = Path("results/leduc/convergence.csv")
    if not source.exists():
        return []
    parsed = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    evaluation_elapsed = 0.0
    rows = []
    for source_row in parsed:
        evaluation_elapsed += float(source_row["best_response_seconds"])
        runtime = float(source_row["runtime_seconds"])
        iteration = int(source_row["iteration"])
        rows.append(
            {
                "game": "leduc",
                "algorithm": "vanilla_cfr",
                "seed": 7,
                "iteration": iteration,
                "expected_game_value_player0": float(source_row["expected_game_value"]),
                "expected_game_value_player1": -float(
                    source_row["expected_game_value"]
                ),
                "nashconv": float(source_row["nashconv"]),
                "exploitability": float(source_row["exploitability"]),
                "training_elapsed_seconds": max(0.0, runtime - evaluation_elapsed),
                "evaluation_elapsed_seconds": evaluation_elapsed,
                "iterations_per_second": iteration
                / max(1e-12, runtime - evaluation_elapsed),
                "information_sets": int(source_row["information_sets"]),
                "sampled_nodes_last_iteration": source_row["game_tree_nodes"],
                "source": str(source),
            }
        )
    return rows


def _ci95(values: list[float]) -> tuple[float | None, float | None]:
    n = len(values)
    if n < 2:
        return (None, None)
    t_critical_by_n = (
        0.0,
        0.0,
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
    )
    t_critical = t_critical_by_n[n] if n < 31 else 1.96
    half = t_critical * statistics.stdev(values) / math.sqrt(n)
    mean = statistics.mean(values)
    return mean - half, mean + half


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["iteration"]) in (1_000, 5_000, 10_000):
            groups[(row["game"], row["algorithm"], int(row["iteration"]))].append(row)
    output = []
    for (game, algorithm, iteration), group in sorted(groups.items()):
        values = [float(row["exploitability"]) for row in group]
        low, high = _ci95(values)
        output.append(
            {
                "game": game,
                "algorithm": algorithm,
                "iteration": iteration,
                "seed_count": len(values),
                "exploitability_mean": statistics.mean(values),
                "exploitability_median": statistics.median(values),
                "exploitability_std": statistics.stdev(values)
                if len(values) > 1
                else None,
                "exploitability_min": min(values),
                "exploitability_max": max(values),
                "exploitability_ci95_low": low,
                "exploitability_ci95_high": high,
                "player0_value_mean": statistics.mean(
                    float(row["expected_game_value_player0"]) for row in group
                ),
                "training_time_mean_seconds": statistics.mean(
                    float(row["training_elapsed_seconds"]) for row in group
                ),
                "evaluation_time_mean_seconds": statistics.mean(
                    float(row["evaluation_elapsed_seconds"]) for row in group
                ),
            }
        )
    return output


def _time_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["game"], row["algorithm"], int(row["seed"]))].append(row)
    curves = {}
    max_time_by_game = defaultdict(float)
    for key, values in grouped.items():
        values.sort(key=lambda row: float(row["training_elapsed_seconds"]))
        curves[key] = values
        max_time_by_game[key[0]] = max(
            max_time_by_game[key[0]], float(values[-1]["training_elapsed_seconds"])
        )
    output = []
    for game, max_time in max_time_by_game.items():
        step = 5.0 if game == "leduc" else 0.1
        grid = [step * i for i in range(1, int(max_time / step) + 1)]
        for algorithm in ("vanilla_cfr", "external_sampling_mccfr"):
            for seconds in grid:
                values = []
                for (curve_game, curve_algorithm, seed), curve in curves.items():
                    if curve_game != game or curve_algorithm != algorithm:
                        continue
                    xs = [float(row["training_elapsed_seconds"]) for row in curve]
                    ys = [float(row["exploitability"]) for row in curve]
                    if seconds < xs[0] or seconds > xs[-1]:
                        continue
                    right = next(
                        (index for index, value in enumerate(xs) if value >= seconds),
                        len(xs) - 1,
                    )
                    if xs[right] == seconds or right == 0:
                        estimate = ys[right]
                    else:
                        left = right - 1
                        fraction = (seconds - xs[left]) / (xs[right] - xs[left])
                        estimate = ys[left] + fraction * (ys[right] - ys[left])
                    values.append((seed, estimate))
                if not values:
                    continue
                sample = [value for _, value in values]
                low, high = _ci95(sample)
                output.append(
                    {
                        "game": game,
                        "algorithm": algorithm,
                        "training_seconds": seconds,
                        "seed_count": len(sample),
                        "exploitability_mean": statistics.mean(sample),
                        "exploitability_median": statistics.median(sample),
                        "exploitability_std": (
                            statistics.stdev(sample) if len(sample) > 1 else None
                        ),
                        "exploitability_min": min(sample),
                        "exploitability_max": max(sample),
                        "exploitability_ci95_low": low,
                        "exploitability_ci95_high": high,
                        "seed_values": json.dumps(
                            {str(seed): value for seed, value in values},
                            sort_keys=True,
                        ),
                    }
                )
    return output


def _coverage_probe(
    game: str,
    solver,
    profile,
    seed: int,
    trials: int = 500,
    *,
    algorithm: str = "unspecified",
    run_seed: int | None = None,
    profile_iteration: int | None = None,
    profile_source: str | None = None,
):
    visited_counts: dict[Any, int] = defaultdict(int)
    if not hasattr(solver, "_traverse_external"):
        solver = (
            KuhnMCCFRSolver(KuhnMCCFRConfig(iterations=1, seed=seed))
            if game == "kuhn"
            else LeducMCCFRSolver(LeducMCCFRConfig(iterations=1, seed=seed))
        )
    solver._rng.seed(seed)
    original = solver._traverse_external
    info_type = KuhnInfo if game == "kuhn" else LeducInfo

    def tracked(state, traverser, frozen_profile, regret_delta, **kwargs):
        if (
            not state.is_terminal
            and not getattr(state, "is_chance_node", False)
            and state.current_player == traverser
        ):
            visited_this_trajectory.add(info_type.from_state(state))
        return original(
            state,
            traverser,
            frozen_profile,
            regret_delta,
            **kwargs,
        )

    solver._traverse_external = tracked
    deals = tuple(KuhnGame.deals()) if game == "kuhn" else LeducGame.deals()
    try:
        for traverser in (0, 1):
            for _ in range(trials):
                visited_this_trajectory: set[Any] = set()
                deal = solver._rng.choice(deals)
                state = (
                    KuhnGame.initial_state(deal)
                    if game == "kuhn"
                    else LeducGame.initial_state(deal)
                )
                solver._traverse_external(state, traverser, profile, {})
                for info in visited_this_trajectory:
                    visited_counts[info] += 1
    finally:
        solver._traverse_external = original
    return [
        {
            "game": game,
            "algorithm": algorithm,
            "seed": seed if run_seed is None else run_seed,
            "diagnostic_seed": seed,
            "profile_iteration": profile_iteration,
            "profile_source": profile_source,
            "traverser": info.player,
            "information_set": info.identifier,
            "coverage_trials_per_traverser": trials,
            "visits": visited_counts.get(info, 0),
            "visit_rate": visited_counts.get(info, 0) / trials,
        }
        for info in solver.information_sets
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    if not rows:
        return
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(iteration_rows, time_rows, output_dir: Path) -> list[Path]:
    import plotly.graph_objects as go

    paths = []
    for game in ("kuhn", "leduc"):
        for source_rows, x_key, suffix, x_title in (
            (
                iteration_rows,
                "iteration",
                "exploitability_vs_iterations.html",
                "Iterations",
            ),
            (
                time_rows,
                "training_seconds",
                "exploitability_vs_training_time.html",
                "Training time (seconds)",
            ),
        ):
            figure = go.Figure()
            for algorithm, color in (
                ("vanilla_cfr", "#1f77b4"),
                ("external_sampling_mccfr", "#d62728"),
            ):
                group = [
                    row
                    for row in source_rows
                    if row["game"] == game and row["algorithm"] == algorithm
                ]
                if not group:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=[row[x_key] for row in group],
                        y=[row["exploitability_mean"] for row in group],
                        mode="lines+markers",
                        name=f"{algorithm} mean",
                        line={"color": color, "width": 3},
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": [
                                0.0
                                if row["exploitability_ci95_high"] is None
                                else row["exploitability_ci95_high"]
                                - row["exploitability_mean"]
                                for row in group
                            ],
                            "arrayminus": [
                                0.0
                                if row["exploitability_ci95_low"] is None
                                else row["exploitability_mean"]
                                - row["exploitability_ci95_low"]
                                for row in group
                            ],
                        },
                    )
                )
            figure.update_layout(
                title=f"{game.title()} exact exploitability {x_title.lower()}",
                xaxis_title=x_title,
                yaxis_title="Exploitability (chips per hand)",
            )
            target = output_dir / f"{game}_{suffix}"
            figure.write_html(target, include_plotlyjs="cdn")
            paths.append(target)
    return paths


def run_investigation(
    *,
    seeds: tuple[int, ...] = (7, 19, 31, 43, 59),
    kuhn_iterations: int = 10_000,
    leduc_mccfr_iterations: int = 10_000,
    leduc_vanilla_iterations: int = 5_000,
    variance_trials: int = 25_000,
    coverage_trials: int = 500,
    output_root: str | Path = "results/mccfr/diagnostics",
) -> Path:
    if len(seeds) < 5:
        raise ValueError("at least five seeds are required for this investigation")
    if min(kuhn_iterations, leduc_mccfr_iterations, leduc_vanilla_iterations) < 1:
        raise ValueError("iteration budgets must be positive")
    output_dir = Path(output_root) / datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    config = {
        "seeds": list(seeds),
        "kuhn_iterations": kuhn_iterations,
        "leduc_mccfr_iterations": leduc_mccfr_iterations,
        "leduc_vanilla_iterations": leduc_vanilla_iterations,
        "variance_trials_per_information_set": variance_trials,
        "coverage_trials_per_traverser_and_checkpoint": coverage_trials,
        "checkpoints_kuhn": [1_000, 5_000, 10_000],
        "checkpoints_leduc_mccfr": [1_000, 5_000, 10_000],
        "checkpoints_leduc_vanilla": [1_000, 5_000],
        "exact_evaluation_interval_leduc": 1_000,
        "baseline_reuse": {
            "leduc_vanilla_1k_seed_7": "results/mccfr/run-20261030T213604Z/convergence.csv",
            "leduc_vanilla_5k_seed_7": "results/leduc/convergence.csv",
        },
        "leduc_vanilla_10k": "not run; measured 1k runtime projects to roughly 11 minutes per seed",
        "leduc_vanilla_5k_other_seeds": "new runs for seeds 19, 31, 43 and 59; seed 7 reuses existing exact convergence data",
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    all_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    run_notes = []
    mccfr_snapshot_for_variance: dict[str, Path] = {}

    # Reuse the 1k seed-7 anchor; seed 19 gets a fresh, longer trajectory.
    pilot_dir = Path("results/mccfr/run-20261030T213604Z")
    for seed in seeds:
        if seed == 7:
            rows = _read_pilot_rows(pilot_dir, "leduc", "vanilla_cfr", seed)
            all_rows.extend(rows)
            _write_csv(raw_dir / f"leduc_vanilla_cfr_seed_{seed}.csv", rows)
            run_notes.append(
                {
                    "game": "leduc",
                    "algorithm": "vanilla_cfr",
                    "seed": seed,
                    "iterations": 1_000,
                    "source": str(pilot_dir / "convergence.csv"),
                    "reused": True,
                }
            )

    # New vanilla controls provide matched 1k and 5k results for four seeds.
    for seed in seeds:
        if seed == 7:
            continue
        vanilla_checkpoints = tuple(
            sorted(
                checkpoint
                for checkpoint in (1_000, leduc_vanilla_iterations)
                if checkpoint <= leduc_vanilla_iterations
            )
        )
        rows, solver = _run_solver(
            "leduc",
            "vanilla_cfr",
            seed,
            leduc_vanilla_iterations,
            vanilla_checkpoints,
            output_dir,
        )
        all_rows.extend(rows)
        _write_csv(raw_dir / f"leduc_vanilla_cfr_seed_{seed}.csv", rows)
        run_notes.append(
            {
                "game": "leduc",
                "algorithm": "vanilla_cfr",
                "seed": seed,
                "iterations": leduc_vanilla_iterations,
                "reused": False,
            }
        )

    # Reuse the recorded exact 5k vanilla seed-7 convergence checkpoints.
    leduc_five_k = _reuse_leduc_five_k(output_dir)
    # The newer 1k pilot is the preferred seed-7 anchor for this comparison.
    leduc_five_k = [row for row in leduc_five_k if int(row["iteration"]) > 1_000]
    all_rows.extend(leduc_five_k)
    if leduc_five_k:
        seed7_path = raw_dir / "leduc_vanilla_cfr_seed_7.csv"
        prior = list(csv.DictReader(seed7_path.open(newline="", encoding="utf-8")))
        _write_csv(
            seed7_path,
            prior + leduc_five_k,
            fields=ROW_FIELDS,
        )
        run_notes.append(
            {
                "game": "leduc",
                "algorithm": "vanilla_cfr",
                "seed": 7,
                "iterations": 5_000,
                "source": "results/leduc/convergence.csv; training time reconstructed by subtracting recorded best-response durations",
                "reused": True,
            }
        )

    # Full multi-seed Kuhn check at a common 10k budget.
    kuhn_checkpoints = tuple(sorted({1_000, 5_000, kuhn_iterations}))
    for algorithm in ("vanilla_cfr", "external_sampling_mccfr"):
        for seed in seeds:
            rows, solver = _run_solver(
                "kuhn",
                algorithm,
                seed,
                kuhn_iterations,
                kuhn_checkpoints,
                output_dir,
            )
            all_rows.extend(rows)
            _write_csv(
                raw_dir / f"kuhn_{algorithm}_seed_{seed}.csv",
                rows,
                fields=ROW_FIELDS,
            )
            if algorithm == "external_sampling_mccfr" and seed == 7:
                mccfr_snapshot_for_variance["kuhn"] = (
                    output_dir
                    / f"strategy_kuhn_external_sampling_mccfr_seed_7_iter_{kuhn_iterations}.json"
                )
            final_profile = solver.average_strategy_profile()
            coverage_rows.extend(
                _coverage_probe(
                    "kuhn",
                    solver,
                    final_profile,
                    seed + 700_000,
                    trials=coverage_trials,
                    algorithm=algorithm,
                    run_seed=seed,
                    profile_iteration=kuhn_iterations,
                    profile_source=str(
                        output_dir
                        / f"strategy_kuhn_{algorithm}_seed_{seed}_iter_{kuhn_iterations}.json"
                    ),
                )
            )

    # Leduc MCCFR gets the long 1k/5k/10k multi-seed trajectory.
    leduc_checkpoints = tuple(sorted({1_000, 5_000, leduc_mccfr_iterations}))
    for seed in seeds:
        rows, solver = _run_solver(
            "leduc",
            "external_sampling_mccfr",
            seed,
            leduc_mccfr_iterations,
            leduc_checkpoints,
            output_dir,
        )
        all_rows.extend(rows)
        _write_csv(
            raw_dir / f"leduc_external_sampling_mccfr_seed_{seed}.csv",
            rows,
            fields=ROW_FIELDS,
        )
        if seed == 7:
            mccfr_snapshot_for_variance["leduc"] = (
                output_dir
                / f"strategy_leduc_external_sampling_mccfr_seed_7_iter_{leduc_mccfr_iterations}.json"
            )
        final_profile = solver.average_strategy_profile()
        coverage_rows.extend(
            _coverage_probe(
                "leduc",
                solver,
                final_profile,
                seed + 800_000,
                trials=coverage_trials,
                algorithm="external_sampling_mccfr",
                run_seed=seed,
                profile_iteration=leduc_mccfr_iterations,
                profile_source=str(
                    output_dir
                    / f"strategy_leduc_external_sampling_mccfr_seed_{seed}_iter_{leduc_mccfr_iterations}.json"
                ),
            )
        )

    # Sample coverage for each vanilla profile at the latest available budget.
    for seed in seeds:
        profile_path = (
            output_dir
            / f"strategy_leduc_vanilla_cfr_seed_{seed}_iter_{leduc_vanilla_iterations}.json"
        )
        profile_iteration = leduc_vanilla_iterations
        if not profile_path.exists() and seed == 7:
            profile_path = pilot_dir / f"strategy_leduc_vanilla_cfr_seed_{seed}.json"
            profile_iteration = 1_000
        solver = LeducMCCFRSolver(LeducMCCFRConfig(iterations=1, seed=seed))
        profile = load_leduc_snapshot(profile_path)
        coverage_rows.extend(
            _coverage_probe(
                "leduc",
                solver,
                profile,
                seed + 900_000,
                trials=coverage_trials,
                algorithm="vanilla_cfr",
                run_seed=seed,
                profile_iteration=profile_iteration,
                profile_source=str(profile_path),
            )
        )

    # Seeded regret variance diagnostics use the fixed 1k MCCFR seed-7 profile
    # from the original investigation runs, not a profile learned during tests.
    old_snapshot_paths = {
        "kuhn": Path(
            "results/mccfr/run-20261030T213604Z/strategy_kuhn_external_sampling_mccfr_seed_7.json"
        ),
        "leduc": Path(
            "results/mccfr/run-20261030T213604Z/strategy_leduc_external_sampling_mccfr_seed_7.json"
        ),
    }
    variance_rows = []
    for game, snapshot_path in old_snapshot_paths.items():
        profile = (
            load_kuhn_snapshot(snapshot_path)
            if game == "kuhn"
            else load_leduc_snapshot(snapshot_path)
        )
        variance_rows.extend(
            _variance_for_game(
                game,
                profile,
                snapshot_path,
                variance_trials,
                seed=1_930_000 if game == "kuhn" else 1_940_000,
            )
        )
    _write_csv(output_dir / "regret_variance.csv", variance_rows)
    _write_csv(output_dir / "information_set_coverage.csv", coverage_rows)
    _write_csv(output_dir / "per_seed_all_runs.csv", all_rows, fields=ROW_FIELDS)
    aggregate_rows = _aggregate(all_rows)
    time_rows = _time_aggregate(all_rows)
    _write_csv(output_dir / "aggregated_comparison.csv", aggregate_rows)
    _write_csv(output_dir / "time_aggregated.csv", time_rows)
    (output_dir / "run_notes.json").write_text(
        json.dumps(run_notes, indent=2) + "\n", encoding="utf-8"
    )
    _make_plots(aggregate_rows, time_rows, output_dir)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "configuration": config,
        "methodology": {
            "variance_reference": "Independent full-tree recursion in mccfr_investigation.py; does not call either production CFR update.",
            "variance_profile": "Fixed 1,000-iteration seed-7 MCCFR average strategy from the existing 50-checkpoint run.",
            "variance_samples_include_zero_when_unvisited": True,
            "coverage_reference": "Empirical sampled traversals under each run's final average strategy.",
            "exploitability": "Exact expected value and best-response evaluators; not sampled.",
            "time_comparison": "Training elapsed time excludes exact evaluation; interpolated only within each seed's observed training-time range.",
            "confidence_intervals": "Student-t intervals for mean when n<30 (t critical values for n=2..5); normal 1.96 above that; coverage uses Wilson intervals.",
        },
        "run_notes": run_notes,
        "source_sha256": {},
        "artifact_sha256": {},
    }
    source_paths = (
        Path(__file__),
        Path("src/pokerlab/research/kuhn/mccfr.py"),
        Path("src/pokerlab/research/leduc/mccfr.py"),
        Path("src/pokerlab/research/kuhn/cfr.py"),
        Path("src/pokerlab/research/leduc/cfr.py"),
        Path("src/pokerlab/research/kuhn/evaluation.py"),
        Path("src/pokerlab/research/leduc/evaluation.py"),
        Path("src/pokerlab/research/kuhn/game.py"),
        Path("src/pokerlab/research/leduc/game.py"),
    )
    manifest["source_sha256"] = {str(path): _hash(path) for path in source_paths}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "experiment_manifest.json":
            manifest["artifact_sha256"][str(path.relative_to(output_dir))] = _hash(path)
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def resume_partial_investigation(output_dir: str | Path) -> Path:
    """Complete a run whose raw Leduc controls were saved before a harness error."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    config = json.loads(
        (output_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    seeds = tuple(int(seed) for seed in config["seeds"])
    kuhn_iterations = int(config["kuhn_iterations"])
    leduc_mccfr_iterations = int(config["leduc_mccfr_iterations"])
    coverage_trials = int(config["coverage_trials_per_traverser_and_checkpoint"])
    all_rows = []
    run_notes = [{"resumed_from_saved_raw_files": True}]
    for raw_path in sorted(raw_dir.glob("*.csv")):
        with raw_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        all_rows.extend(rows)
        run_notes.append(
            {"source": str(raw_path), "reused_rows": len(rows), "reused": True}
        )

    coverage_rows = []
    kuhn_checkpoints = tuple(sorted({1_000, 5_000, kuhn_iterations}))
    for algorithm in ("vanilla_cfr", "external_sampling_mccfr"):
        for seed in seeds:
            raw_path = raw_dir / f"kuhn_{algorithm}_seed_{seed}.csv"
            snapshot_path = (
                output_dir
                / f"strategy_kuhn_{algorithm}_seed_{seed}_iter_{kuhn_iterations}.json"
            )
            was_saved = raw_path.exists() and snapshot_path.exists()
            if was_saved:
                with raw_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                profile = load_kuhn_snapshot(snapshot_path)
            else:
                rows, solver = _run_solver(
                    "kuhn",
                    algorithm,
                    seed,
                    kuhn_iterations,
                    kuhn_checkpoints,
                    output_dir,
                )
                _write_csv(raw_path, rows, fields=ROW_FIELDS)
                profile = solver.average_strategy_profile()
                all_rows.extend(rows)
            sampler = KuhnMCCFRSolver(KuhnMCCFRConfig(iterations=1, seed=seed))
            coverage_rows.extend(
                _coverage_probe(
                    "kuhn",
                    sampler,
                    profile,
                    seed + 700_000,
                    trials=coverage_trials,
                    algorithm=algorithm,
                    run_seed=seed,
                    profile_iteration=kuhn_iterations,
                    profile_source=str(snapshot_path),
                )
            )
            run_notes.append(
                {
                    "game": "kuhn",
                    "algorithm": algorithm,
                    "seed": seed,
                    "iterations": kuhn_iterations,
                    "source": str(raw_path),
                    "reused": was_saved,
                }
            )

    leduc_checkpoints = tuple(sorted({1_000, 5_000, leduc_mccfr_iterations}))
    for seed in seeds:
        raw_path = raw_dir / f"leduc_external_sampling_mccfr_seed_{seed}.csv"
        rows, solver = _run_solver(
            "leduc",
            "external_sampling_mccfr",
            seed,
            leduc_mccfr_iterations,
            leduc_checkpoints,
            output_dir,
        )
        _write_csv(raw_path, rows, fields=ROW_FIELDS)
        all_rows.extend(rows)
        profile = solver.average_strategy_profile()
        snapshot_path = (
            output_dir
            / f"strategy_leduc_external_sampling_mccfr_seed_{seed}_iter_{leduc_mccfr_iterations}.json"
        )
        coverage_rows.extend(
            _coverage_probe(
                "leduc",
                solver,
                profile,
                seed + 800_000,
                trials=coverage_trials,
                algorithm="external_sampling_mccfr",
                run_seed=seed,
                profile_iteration=leduc_mccfr_iterations,
                profile_source=str(snapshot_path),
            )
        )

    pilot_dir = Path("results/mccfr/run-20261030T213604Z")
    for seed in seeds:
        snapshot_path = (
            raw_dir.parent / f"strategy_leduc_vanilla_cfr_seed_{seed}_iter_5000.json"
        )
        profile_iteration = 5_000
        if not snapshot_path.exists():
            if seed != 7:
                raise FileNotFoundError(snapshot_path)
            snapshot_path = pilot_dir / f"strategy_leduc_vanilla_cfr_seed_{seed}.json"
            profile_iteration = 1_000
        profile = load_leduc_snapshot(snapshot_path)
        sampler = LeducMCCFRSolver(LeducMCCFRConfig(iterations=1, seed=seed))
        coverage_rows.extend(
            _coverage_probe(
                "leduc",
                sampler,
                profile,
                seed + 900_000,
                trials=coverage_trials,
                algorithm="vanilla_cfr",
                run_seed=seed,
                profile_iteration=profile_iteration,
                profile_source=str(snapshot_path),
            )
        )

    old_snapshot_paths = {
        "kuhn": Path(
            "results/mccfr/run-20261030T213604Z/strategy_kuhn_external_sampling_mccfr_seed_7.json"
        ),
        "leduc": Path(
            "results/mccfr/run-20261030T213604Z/strategy_leduc_external_sampling_mccfr_seed_7.json"
        ),
    }
    variance_rows = []
    for game, snapshot_path in old_snapshot_paths.items():
        profile = (
            load_kuhn_snapshot(snapshot_path)
            if game == "kuhn"
            else load_leduc_snapshot(snapshot_path)
        )
        variance_rows.extend(
            _variance_for_game(
                game,
                profile,
                snapshot_path,
                int(config["variance_trials_per_information_set"]),
                seed=1_930_000 if game == "kuhn" else 1_940_000,
            )
        )

    _write_csv(output_dir / "regret_variance.csv", variance_rows)
    _write_csv(output_dir / "information_set_coverage.csv", coverage_rows)
    _write_csv(output_dir / "per_seed_all_runs.csv", all_rows, fields=ROW_FIELDS)
    aggregate_rows = _aggregate(all_rows)
    time_rows = _time_aggregate(all_rows)
    _write_csv(output_dir / "aggregated_comparison.csv", aggregate_rows)
    _write_csv(output_dir / "time_aggregated.csv", time_rows)
    (output_dir / "run_notes.json").write_text(
        json.dumps(run_notes, indent=2) + "\n", encoding="utf-8"
    )
    _make_plots(aggregate_rows, time_rows, output_dir)

    source_paths = (
        Path(__file__),
        Path("src/pokerlab/research/kuhn/mccfr.py"),
        Path("src/pokerlab/research/leduc/mccfr.py"),
        Path("src/pokerlab/research/kuhn/cfr.py"),
        Path("src/pokerlab/research/leduc/cfr.py"),
        Path("src/pokerlab/research/kuhn/evaluation.py"),
        Path("src/pokerlab/research/leduc/evaluation.py"),
        Path("src/pokerlab/research/kuhn/game.py"),
        Path("src/pokerlab/research/leduc/game.py"),
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "configuration": config,
        "methodology": {
            "variance_reference": "Independent full-tree recursion; no production CFR update method is called.",
            "variance_samples_include_zero_when_unvisited": True,
            "coverage_reference": "Labeled sampled traversals under per-seed average strategies; profile iteration and source are recorded.",
            "exploitability": "Exact expected value and best-response evaluators.",
            "time_comparison": "Training elapsed excludes exact evaluation; interpolation stays inside each seed's measured range.",
            "confidence_intervals": "Student-t intervals for small seed counts; Wilson intervals for coverage.",
        },
        "resume_note": "The first execution completed all Leduc vanilla controls and saved their raw files, then the coverage helper failed on Kuhn vanilla's absent _traverse_external method. This resume used a separate MCCFR traversal sampler for coverage; no production solver code changed.",
        "run_notes": run_notes,
        "source_sha256": {str(path): _hash(path) for path in source_paths},
        "artifact_sha256": {},
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "experiment_manifest.json":
            manifest["artifact_sha256"][str(path.relative_to(output_dir))] = _hash(path)
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(7, 19, 31, 43, 59))
    parser.add_argument("--kuhn-iterations", type=int, default=10_000)
    parser.add_argument("--leduc-mccfr-iterations", type=int, default=10_000)
    parser.add_argument("--leduc-vanilla-iterations", type=int, default=5_000)
    parser.add_argument("--variance-trials", type=int, default=25_000)
    parser.add_argument("--coverage-trials", type=int, default=500)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/mccfr/diagnostics")
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="resume the unfinished timestamped run from its saved raw seed files",
    )
    args = parser.parse_args()
    if args.resume_dir is not None:
        print(
            f"MCCFR investigation artifacts: {resume_partial_investigation(args.resume_dir)}"
        )
        return
    output = run_investigation(
        seeds=tuple(args.seeds),
        kuhn_iterations=args.kuhn_iterations,
        leduc_mccfr_iterations=args.leduc_mccfr_iterations,
        leduc_vanilla_iterations=args.leduc_vanilla_iterations,
        variance_trials=args.variance_trials,
        coverage_trials=args.coverage_trials,
        output_root=args.output_root,
    )
    print(f"MCCFR investigation artifacts: {output}")


if __name__ == "__main__":
    main()
