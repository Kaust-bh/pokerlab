"""Statistical summaries for paired HUNL abstraction benchmark matches."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

T_975 = (
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
    2.042,
)


def _t_critical(df: int) -> float:
    return T_975[df] if df < len(T_975) else 1.96


def mean_interval(
    values: list[float],
) -> tuple[float, float, float, float | None, float | None]:
    """Return mean, sample SD, SE and 95% t CI (undefined for one sample)."""
    if not values:
        return 0.0, 0.0, 0.0, None, None
    average = mean(values)
    if len(values) == 1:
        return average, 0.0, 0.0, None, None
    deviation = stdev(values)
    standard_error = deviation / math.sqrt(len(values))
    margin = _t_critical(len(values) - 1) * standard_error
    return average, deviation, standard_error, average - margin, average + margin


def wilson_interval(wins: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return 0.0, 0.0, 1.0
    proportion = wins / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return proportion, max(0.0, center - margin), min(1.0, center + margin)


def summarize_matches(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["variant"],
                int(row["checkpoint_iteration"]),
                row["opponent"],
                row["trained_first_button"],
            )
        ].append(row)
    summaries = []
    for (variant, checkpoint, opponent, first_button), group in sorted(groups.items()):
        values = [float(row["bb_per_100"]) for row in group]
        average, deviation, standard_error, low, high = mean_interval(values)
        win_count = sum(int(row["match_win"]) for row in group)
        win_rate, win_low, win_high = wilson_interval(win_count, len(group))
        hand_wins = sum(int(row["hand_wins"]) for row in group)
        hand_ties = sum(int(row["hand_ties"]) for row in group)
        hands = sum(int(row["hands_played"]) for row in group)
        hands_decided = hands - hand_ties
        summaries.append(
            {
                "variant": variant,
                "checkpoint_iteration": checkpoint,
                "opponent": opponent,
                "trained_first_button": first_button,
                "matches": len(group),
                "hands": hands,
                "hand_wins": hand_wins,
                "hand_ties": hand_ties,
                "hand_win_rate_excluding_ties": (
                    hand_wins / hands_decided if hands_decided else 0.0
                ),
                "mean_bb_per_100": average,
                "median_bb_per_100": median(values),
                "sd_bb_per_100": deviation,
                "se_bb_per_100": standard_error,
                "ci95_low_bb_per_100": low,
                "ci95_high_bb_per_100": high,
                "match_win_rate": win_rate,
                "win_ci95_low": win_low,
                "win_ci95_high": win_high,
                "mean_match_runtime_seconds": mean(
                    float(row["runtime_seconds"]) for row in group
                ),
                "mean_trained_seconds_per_decision": mean(
                    float(row["trained_seconds_per_decision"]) for row in group
                ),
            }
        )
    return summaries


def paired_abstraction_comparisons(
    rows: list[dict], baseline: str = "baseline"
) -> list[dict]:
    indexed = {}
    for row in rows:
        if row["opponent"] == "PreviousMCCFR":
            continue
        key = (
            row["variant"],
            int(row["seed"]),
            int(row["checkpoint_iteration"]),
            row["opponent"],
            row["trained_first_button"],
        )
        indexed[key] = float(row["bb_per_100"])
    baseline_rows = {
        key[1:]: value for key, value in indexed.items() if key[0] == baseline
    }
    variants = sorted({key[0] for key in indexed} - {baseline})
    result = []
    for variant in variants:
        paired = defaultdict(list)
        for key, value in indexed.items():
            if key[0] != variant:
                continue
            comparison_key = key[1:]
            if comparison_key in baseline_rows:
                paired[(key[2], key[3], key[4])].append(
                    value - baseline_rows[comparison_key]
                )
        for (checkpoint, opponent, first_button), deltas in sorted(paired.items()):
            avg, sd, se, low, high = mean_interval(deltas)
            result.append(
                {
                    "variant": variant,
                    "baseline": baseline,
                    "checkpoint_iteration": checkpoint,
                    "opponent": opponent,
                    "trained_first_button": first_button,
                    "paired_seed_matches": len(deltas),
                    "mean_delta_bb_per_100": avg,
                    "sd_delta_bb_per_100": sd,
                    "se_delta_bb_per_100": se,
                    "ci95_low_delta_bb_per_100": low,
                    "ci95_high_delta_bb_per_100": high,
                    "ci_excludes_zero": int(low is not None and (low > 0 or high < 0)),
                }
            )
    return result


def write_analysis(run_dir: Path, rows: list[dict]) -> None:
    summary = summarize_matches(rows)
    paired = paired_abstraction_comparisons(rows)
    for filename, data in (
        ("summary.csv", summary),
        ("paired_comparisons.csv", paired),
    ):
        path = run_dir / filename
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        with path.open("w", newline="") as handle:
            if not data:
                handle.write("\n")
            else:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    report = {
        "summary_rows": len(summary),
        "paired_comparison_rows": len(paired),
        "exact_exploitability_available": False,
        "confidence_intervals": "two-sided 95% Student t intervals across independent training seeds for batch BB/100 means; Wilson score intervals for strict batch-win proportions; hand win rate is descriptive and excludes tied hands",
        "paired_method": "same seed, checkpoint, opponent, hand seeds and initial button; intervals are over matched seed-level differences",
        "interpretation": "Confidence intervals quantify observed match variation; they do not establish equilibrium convergence or practical superiority.",
    }
    report_path = run_dir / "statistical_summary.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _write_plots(run_dir, summary)


def _write_plots(run_dir: Path, summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    figure = go.Figure()
    series = sorted(
        {
            (row["variant"], row["opponent"], row["trained_first_button"])
            for row in summary
        }
    )
    for variant, opponent, first_button in series:
        selected = sorted(
            (
                row
                for row in summary
                if row["variant"] == variant
                and row["opponent"] == opponent
                and row["trained_first_button"] == first_button
            ),
            key=lambda row: row["checkpoint_iteration"],
        )
        if not selected:
            continue
        trace = go.Scatter(
            x=[row["checkpoint_iteration"] for row in selected],
            y=[row["mean_bb_per_100"] for row in selected],
            mode="lines+markers",
            name=f"{variant} vs {opponent} ({first_button} button)",
        )
        if all(row["ci95_low_bb_per_100"] is not None for row in selected):
            trace.error_y = {
                "type": "data",
                "symmetric": False,
                "array": [
                    row["ci95_high_bb_per_100"] - row["mean_bb_per_100"]
                    for row in selected
                ],
                "arrayminus": [
                    row["mean_bb_per_100"] - row["ci95_low_bb_per_100"]
                    for row in selected
                ],
            }
        figure.add_trace(trace)
    figure.update_layout(
        title="Exploratory BB/100 by MCCFR checkpoint",
        xaxis_title="Training iterations",
        yaxis_title="Fixed-hand batch BB/100 (95% t interval across seeds)",
    )
    path = run_dir / "performance_by_iteration.html"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    figure.write_html(path, include_plotlyjs="cdn")
