"""Strategy comparison metrics for exactly matching small-game information sets."""

from __future__ import annotations

import math
from collections.abc import Mapping


def mean_information_set_total_variation[InformationSet, Action](
    candidate: Mapping[InformationSet, Mapping[Action, float]],
    reference: Mapping[InformationSet, Mapping[Action, float]],
) -> float:
    """Return unweighted mean TV distance across the same infosets.

    Each row is normalized and legal actions are compared by key. Equal
    information-set weighting makes this a strategy approximation diagnostic,
    not an estimate of on-policy exploitability or reach-weighted error.
    """
    if not candidate or candidate.keys() != reference.keys():
        raise ValueError("strategy profiles must contain the same nonempty infosets")

    distances = []
    for info in candidate:
        candidate_row = candidate[info]
        reference_row = reference[info]
        if not candidate_row or candidate_row.keys() != reference_row.keys():
            raise ValueError("strategy rows must have matching legal actions")
        for row in (candidate_row, reference_row):
            values = tuple(float(value) for value in row.values())
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(
                    "strategy probabilities must be finite and nonnegative"
                )
            if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
                raise ValueError("strategy probabilities must sum to one")
        distances.append(
            0.5
            * sum(
                abs(float(candidate_row[action]) - float(reference_row[action]))
                for action in candidate_row
            )
        )
    return sum(distances) / len(distances)
