"""Deterministic 169-class preflop and sampled-strength postflop buckets."""

from __future__ import annotations

import math
import random
from functools import lru_cache
from itertools import combinations

from pokerlab.engine import Card, HandCategory, evaluate_hand
from pokerlab.research.nlhe.config import HUNLGameConfig

RANK_LABELS = "23456789TJQKA"
SUIT_INDEX = {"clubs": 0, "diamonds": 1, "hearts": 2, "spades": 3}
SUITS = ("clubs", "diamonds", "hearts", "spades")
PREMIUM_CLASSES = frozenset(
    {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "KQs"}
)


def preflop_class(hole_cards: tuple[Card, Card]) -> str:
    """Map any concrete starting hand to its standard 169 canonical class."""
    first, second = sorted(
        hole_cards, key=lambda card: (card.rank, card.suit), reverse=True
    )
    high, low = first.rank, second.rank
    if high == low:
        return RANK_LABELS[high - 2] * 2
    suited = first.suit == second.suit
    return f"{RANK_LABELS[high - 2]}{RANK_LABELS[low - 2]}{'s' if suited else 'o'}"


def _stable_seed(cards: tuple[Card, ...]) -> int:
    value = 2_166_136_261
    for card in cards:
        value = (value ^ (card.rank * 5 + SUIT_INDEX[card.suit] + 1)) * 16_777_619
        value &= 0xFFFFFFFF
    return value


def percentile_bucket(percentile: float, bucket_count: int) -> int:
    """Map a closed-unit-interval percentile to a zero-based bucket."""
    if bucket_count < 1 or not math.isfinite(percentile) or not 0 <= percentile <= 1:
        raise ValueError("percentile must be in [0, 1] and bucket_count positive")
    return min(bucket_count - 1, int(percentile * bucket_count))


def _has_straight(ranks: set[int]) -> bool:
    values = set(ranks)
    if 14 in values:
        values.add(1)
    return any(
        all(rank in values for rank in range(high - 4, high + 1))
        for high in range(5, 15)
    )


def _draw_features(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
    available: tuple[Card, ...],
    made_category: int,
    legacy_combined_flush_outs: bool = False,
) -> tuple[bool, int, int]:
    """Return legacy flush flag, richer flush-draw code, and straight-out bucket."""
    counts: dict[str, int] = {}
    for card in (*hole, *board):
        counts[card.suit] = counts.get(card.suit, 0) + 1
    maximum_suit_count = max(counts.values(), default=0)
    legacy_flush_draw = maximum_suit_count == 4
    flush_code = (
        2
        if maximum_suit_count >= 4 and len(board) < 5
        else 1
        if len(board) == 3 and maximum_suit_count == 3
        else 0
    )
    current_ranks = {card.rank for card in (*hole, *board)}
    if len(board) == 5 or made_category >= int(HandCategory.STRAIGHT):
        return legacy_flush_draw, flush_code, 0
    flush_suit = next((suit for suit, count in counts.items() if count == 4), None)
    outs = sum(
        1
        for card in available
        if _has_straight(current_ranks | {card.rank})
        or (legacy_combined_flush_outs and card.suit == flush_suit)
    )
    straight_bucket = 0 if outs == 0 else 1 if outs <= 4 else 2 if outs <= 8 else 3
    return legacy_flush_draw, flush_code, straight_bucket


@lru_cache(maxsize=100_000)
def _postflop_features(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
    bucket_count: int,
    opponent_samples: int,
) -> tuple[int, int, bool, int]:
    known = set(hole) | set(board)
    hero_rank = evaluate_hand((*hole, *board))
    available = tuple(
        Card(rank, suit)
        for suit in ("clubs", "diamonds", "hearts", "spades")
        for rank in range(2, 15)
        if Card(rank, suit) not in known
    )
    possible_opponents = tuple(combinations(available, 2))
    sample_count = min(opponent_samples, len(possible_opponents))
    seed = _stable_seed((*hole, *board))
    if sample_count == len(possible_opponents):
        sampled_indices = range(sample_count)
    else:
        sampled_indices = random.Random(seed).sample(
            range(len(possible_opponents)), sample_count
        )
    wins = ties = 0
    for index in sampled_indices:
        opponent_rank = evaluate_hand((*possible_opponents[index], *board))
        if hero_rank > opponent_rank:
            wins += 1
        elif hero_rank == opponent_rank:
            ties += 1
    percentile = (wins + 0.5 * ties) / sample_count
    strength_bucket = percentile_bucket(percentile, bucket_count)

    category = int(hero_rank.category)
    flush_draw, _flush_code, straight_draw_bucket = _draw_features(
        hole, board, available, category, legacy_combined_flush_outs=True
    )
    return category, strength_bucket, flush_draw, straight_draw_bucket


@lru_cache(maxsize=100_000)
def _sampled_equity_features(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
    bucket_count: int,
    sample_count: int,
    opponent_range: str,
) -> tuple[int, int, int, int, str]:
    """Estimate showdown equity using sampled legal holdings and runouts."""
    known = set(hole) | set(board)
    available = tuple(
        Card(rank, suit)
        for suit in SUITS
        for rank in range(2, 15)
        if Card(rank, suit) not in known
    )
    opponent_hands = tuple(combinations(available, 2))
    if opponent_range == "premium":
        filtered = tuple(
            hand for hand in opponent_hands if preflop_class(hand) in PREMIUM_CLASSES
        )
        if filtered:
            opponent_hands = filtered
    seed = _stable_seed((*hole, *board))
    seed ^= sample_count * 0x9E3779B1
    seed ^= 0xA5A5A5A5 if opponent_range == "premium" else 0
    rng = random.Random(seed & 0xFFFFFFFF)
    wins = ties = 0
    future_cards = 5 - len(board)
    for _ in range(sample_count):
        opponent = rng.choice(opponent_hands)
        unavailable = known | set(opponent)
        runout_pool = tuple(card for card in available if card not in unavailable)
        runout = tuple(rng.sample(runout_pool, future_cards)) if future_cards else ()
        final_board = (*board, *runout)
        hero_rank = evaluate_hand((*hole, *final_board))
        opponent_rank = evaluate_hand((*opponent, *final_board))
        if hero_rank > opponent_rank:
            wins += 1
        elif hero_rank == opponent_rank:
            ties += 1
    equity = (wins + 0.5 * ties) / sample_count
    equity_bucket = percentile_bucket(equity, bucket_count)
    hero_category = int(evaluate_hand((*hole, *board)).category)
    _legacy_flush, flush_code, straight_bucket = _draw_features(
        hole, board, available, hero_category
    )
    if hero_category >= int(HandCategory.FLUSH):
        flush_code = 0
    rank_counts: dict[int, int] = {}
    suit_counts: dict[str, int] = {}
    for card in board:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
    pairedness = "unpaired" if max(rank_counts.values(), default=0) < 2 else "paired"
    max_suit = max(suit_counts.values(), default=0)
    suitedness = (
        "monotone" if max_suit >= 3 else "two_tone" if max_suit == 2 else "rainbow"
    )
    unique_ranks = set(rank_counts)
    if 14 in unique_ranks:
        unique_ranks.add(1)
    span = max(unique_ranks, default=0) - min(unique_ranks, default=0)
    connectedness = (
        "connected" if span <= 4 else "semi_connected" if span <= 6 else "disconnected"
    )
    texture = f"{pairedness}:{suitedness}:{connectedness}"
    return hero_category, equity_bucket, flush_code, straight_bucket, texture


def _suit_signature(
    suit: str, hole: tuple[Card, ...], board: tuple[Card, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return a suit-name-independent and flop-order-independent rank signature."""
    return (
        tuple(sorted((c.rank for c in hole if c.suit == suit), reverse=True)),
        tuple(sorted((c.rank for c in board[:3] if c.suit == suit), reverse=True)),
        tuple(c.rank for c in board[3:4] if c.suit == suit),
        tuple(c.rank for c in board[4:5] if c.suit == suit),
    )


def _canonicalize_suits(
    hole: tuple[Card, Card], board: tuple[Card, ...]
) -> tuple[tuple[Card, Card], tuple[Card, ...]]:
    """Canonicalize global suit labels and flop order, preserving street boundaries."""
    present_suits: list[str] = []
    for card in (*hole, *board):
        if card.suit not in present_suits:
            present_suits.append(card.suit)
    present_suits.sort(
        key=lambda s: _suit_signature(s, hole, board),
        reverse=True,
    )
    mapping = {suit: SUITS[idx] for idx, suit in enumerate(present_suits)}
    converted_hole = tuple(
        sorted(
            (Card(card.rank, mapping[card.suit]) for card in hole),
            key=lambda c: (c.rank, SUIT_INDEX[c.suit]),
        )
    )
    flop = tuple(
        sorted(
            (Card(card.rank, mapping[card.suit]) for card in board[:3]),
            key=lambda c: (-c.rank, SUIT_INDEX[c.suit]),
        )
    )
    turn_river = tuple(Card(card.rank, mapping[card.suit]) for card in board[3:])
    return converted_hole, (*flop, *turn_river)


def private_hand_bucket(
    hole_cards: tuple[Card, Card],
    board: tuple[Card, ...],
    config: HUNLGameConfig,
) -> str:
    """Return a deterministic private bucket without consulting opponent cards.

    The preflop class is retained postflop to preserve the player's own
    information and action history (perfect recall) under abstraction.
    Both postflop methods canonicalize suits and flop card order first so
    that private buckets are strictly invariant under all 24 suit permutations.
    """
    starting_class = preflop_class(hole_cards)
    if not board:
        return f"{starting_class}|preflop"
    canonical_hole, canonical_board = _canonicalize_suits(
        tuple(hole_cards), tuple(board)
    )
    if config.postflop_method == "sampled_equity":
        category, equity, flush_code, straight_draw, texture = _sampled_equity_features(
            canonical_hole,
            canonical_board,
            config.postflop_buckets,
            config.equity_samples,
            config.opponent_range,
        )
        return (
            f"{starting_class}|equity:{config.postflop_buckets}:"
            f"{config.opponent_range}:{equity}|made:{category}|"
            f"flush_draw:{flush_code}|straight_draw:{straight_draw}|"
            f"texture:{texture}"
        )
    category, strength, flush_draw, straight_draw = _postflop_features(
        canonical_hole,
        canonical_board,
        config.postflop_buckets,
        config.opponent_hand_samples,
    )
    return (
        f"{starting_class}|made:{category}|relative:{strength}|"
        f"flush_draw:{int(flush_draw)}|straight_draw:{straight_draw}"
    )


def clear_postflop_cache() -> None:
    """Clear cached hand features for bounded-memory experiments/tests."""
    _postflop_features.cache_clear()
    _sampled_equity_features.cache_clear()


def postflop_cache_info():
    return _postflop_features.cache_info()


def equity_cache_info():
    return _sampled_equity_features.cache_info()
