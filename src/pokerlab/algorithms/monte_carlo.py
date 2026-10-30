"""Exact and Monte Carlo heads-up Hold'em equity estimation."""

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from pokerlab.engine.cards import SUITS, Card
from pokerlab.engine.hand_evaluator import evaluate_hand


@dataclass(frozen=True, slots=True)
class EquityEstimate:
    win_probability: float
    tie_probability: float
    loss_probability: float
    equity: float
    simulations: int
    confidence_interval: tuple[float, float]
    runtime_seconds: float
    method: str


def _validate_known_cards(
    hole_cards: Sequence[Card], community_cards: Sequence[Card]
) -> tuple[Card, ...]:
    if len(hole_cards) != 2:
        raise ValueError("exactly two hole cards are required")
    if len(community_cards) not in (0, 3, 4, 5):
        raise ValueError("community cards must contain 0, 3, 4, or 5 cards")
    known = (*hole_cards, *community_cards)
    if len(set(known)) != len(known):
        raise ValueError("known cards must be unique")
    return tuple(known)


def _estimate_outcomes(
    hole: tuple[Card, Card], board: tuple[Card, ...], opponent: tuple[Card, Card]
) -> int:
    hero_rank = evaluate_hand((*hole, *board))
    villain_rank = evaluate_hand((*opponent, *board))
    return 1 if hero_rank > villain_rank else -1 if hero_rank < villain_rank else 0


def estimate_equity(
    hole_cards: Sequence[Card],
    community_cards: Sequence[Card] = (),
    *,
    simulations: int = 10_000,
    seed: int | None = None,
    method: str = "auto",
    exact_threshold: int = 100_000,
    confidence: float = 0.95,
) -> EquityEstimate:
    """Estimate win/tie/loss shares against uniform random opponent cards.

    `auto` exactly enumerates when opponent-and-runout outcomes are at most
    `exact_threshold`; otherwise it samples without replacement. Monte Carlo
    intervals use the distribution-free Hoeffding bound for outcomes in [0, 1].
    Exact enumeration has a zero-width interval. All unknown holdings are uniformly
    weighted; no opponent model or future betting is represented.
    """
    known = _validate_known_cards(hole_cards, community_cards)
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if method not in {"auto", "exact", "monte_carlo"}:
        raise ValueError("method must be 'auto', 'exact', or 'monte_carlo'")
    if exact_threshold < 0 or not 0 < confidence < 1:
        raise ValueError("invalid exact threshold or confidence level")
    available = tuple(
        Card(rank, suit)
        for suit in SUITS
        for rank in range(2, 15)
        if Card(rank, suit) not in known
    )
    missing = 5 - len(community_cards)
    outcome_count = math.comb(len(available), 2) * math.comb(
        len(available) - 2, missing
    )
    exact = method == "exact" or (method == "auto" and outcome_count <= exact_threshold)
    if exact and outcome_count > 2_000_000:
        raise ValueError("exact enumeration exceeds the 2,000,000 outcome safety limit")

    start = time.perf_counter()
    wins = ties = losses = 0
    hero = (hole_cards[0], hole_cards[1])
    board = tuple(community_cards)
    if exact:
        # Group by final board so the hero's hand is evaluated once per runout,
        # rather than once for every opponent holding on that runout.
        for runout in combinations(available, missing):
            final_board = (*board, *runout)
            hero_rank = evaluate_hand((*hero, *final_board))
            remaining = tuple(card for card in available if card not in runout)
            for opponent in combinations(remaining, 2):
                villain_rank = evaluate_hand((*opponent, *final_board))
                outcome = (
                    1
                    if hero_rank > villain_rank
                    else -1
                    if hero_rank < villain_rank
                    else 0
                )
                wins += outcome == 1
                ties += outcome == 0
                losses += outcome == -1
        count = wins + ties + losses
    else:
        rng = random.Random(seed)
        for _ in range(simulations):
            draw = rng.sample(available, 2 + missing)
            opponent = (draw[0], draw[1])
            final_board = (*board, *draw[2:])
            outcome = _estimate_outcomes(hero, final_board, opponent)
            wins += outcome == 1
            ties += outcome == 0
            losses += outcome == -1
        count = simulations

    win_probability = wins / count
    tie_probability = ties / count
    loss_probability = losses / count
    equity = win_probability + tie_probability / 2
    if exact:
        interval = (equity, equity)
    else:
        epsilon = math.sqrt(math.log(2 / (1 - confidence)) / (2 * count))
        interval = (max(0.0, equity - epsilon), min(1.0, equity + epsilon))
    return EquityEstimate(
        win_probability,
        tie_probability,
        loss_probability,
        equity,
        count,
        interval,
        time.perf_counter() - start,
        "exact" if exact else "monte_carlo",
    )
