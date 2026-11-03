"""Bayesian Opponent Hand Range Model for Heads-Up No-Limit Texas Hold'em.

Mathematical Foundation:
    P(H | A_1, ..., A_t, B) propto P(H | B) * prod_{k=1}^t P(A_k | H, C_k, theta_opp)

Where:
- H in C(52, 2) is an unordered two-card opponent holding.
- B is the set of known dead cards (hero's hole cards union public board cards).
  Any holding H intersecting B has P(H | B) = 0 (exact card removal).
- C_k = (street, position, facing_bet, facing_raise_count, pot_ratio) is the public
  betting context at action k.
- theta_opp combines the opponent's smoothed behavioral statistics (VPIP, PFR, AFq,
  3-bet rate) with empirical showdown calibration parameters learned from legitimately
  revealed showdown hands.

Statistical Distinction & Assumptions:
1. Range vs. Tendencies:
   - `posterior_weights[H]` represents what hand H the opponent likely holds in the
     *current* hand given their actions so far.
   - `ShowdownCalibration` + `OpponentStatsSnapshot` represent *how* the opponent
     tends to act with various hand strengths across hands.
2. Non-circularity:
   - Each intra-hand action event is keyed by `(hand_number, event_index)` in
     `_applied_events` and is applied at most once to the current hand's posterior.
   - Showdown calibration updates occur at most once per completed showdown hand
     (keyed by `hand_number` in `_calibrated_hands`) and update cross-hand likelihood
     parameters without retroactively mutating past hand posteriors.
3. Fold Handling:
   - When the opponent folds, `is_folded` is set to `True` and the pre-fold posterior
     distribution is preserved (never pretending the folded hand was revealed).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any

from pokerlab.engine import ActionType, Card, HandCategory
from pokerlab.opponent_model.models import (
    ObservedActionEvent,
    OpponentStatsSnapshot,
    RangeClassEstimate,
    RangeComboEstimate,
    RangeSnapshot,
    RangeUpdateStep,
)
from pokerlab.research.nlhe.abstraction import SUIT_INDEX, SUITS, preflop_class

RANK_CHAR = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
SUIT_CHAR = {"clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s"}


def _card_code(card: Card) -> str:
    return f"{RANK_CHAR[card.rank]}{SUIT_CHAR[card.suit]}"


def _preflop_combo_strength(hand: tuple[Card, Card]) -> float:
    """Calibrated preflop strength percentile in [0.05, 0.99] for a 2-card holding."""
    c1, c2 = hand
    high, low = (c1.rank, c2.rank) if c1.rank >= c2.rank else (c2.rank, c1.rank)
    if high == low:
        # Pocket pairs: 22 (~0.52) up to AA (~0.99)
        return min(0.99, 0.52 + (high - 2) * (0.47 / 12.0))
    strength = 0.12 + (high - 2) * 0.032 + (low - 2) * 0.018
    if c1.suit == c2.suit:
        strength += 0.065
    gap = high - low
    if gap == 1:
        strength += 0.045
    elif gap == 2:
        strength += 0.025
    elif gap == 3:
        strength += 0.01
    if high == 14:
        strength += 0.06
    elif high >= 12 and low >= 10:
        strength += 0.045
    return max(0.05, min(0.92, strength))


ALL_CARDS: tuple[Card, ...] = tuple(
    Card(rank, suit) for suit in SUITS for rank in range(2, 15)
)

ALL_COMBOS: tuple[tuple[Card, Card], ...] = tuple(
    tuple(
        sorted(
            comb,
            key=lambda c: (c.rank, SUIT_INDEX[c.suit]),
            reverse=True,
        )
    )
    for comb in combinations(ALL_CARDS, 2)
)

COMBO_TO_CLASS: dict[tuple[Card, Card], str] = {
    combo: preflop_class(combo) for combo in ALL_COMBOS
}

COMBO_PREFLOP_STRENGTH: dict[tuple[Card, Card], float] = {
    combo: _preflop_combo_strength(combo) for combo in ALL_COMBOS
}

COMBO_CODES: dict[tuple[Card, Card], tuple[str, str]] = {
    combo: (_card_code(combo[0]), _card_code(combo[1])) for combo in ALL_COMBOS
}

COMBO_INDEX: dict[tuple[Card, Card], int] = {
    combo: idx for idx, combo in enumerate(ALL_COMBOS)
}

PREFLOP_STRENGTH_VECTOR: tuple[float, ...] = tuple(
    COMBO_PREFLOP_STRENGTH[combo] for combo in ALL_COMBOS
)


_STRAIGHT_WINDOWS: tuple[tuple[int, int], ...] = tuple(
    (
        high,
        (1 << high)
        | (1 << (high - 1))
        | (1 << (high - 2))
        | (1 << (high - 3))
        | (1 << (high - 4)),
    )
    for high in range(14, 4, -1)
)


def _best_straight_high(mask: int) -> int:
    for high, win in _STRAIGHT_WINDOWS:
        if (mask & win) == win:
            return high
    return 0


def fast_hand_rank_value(cards: Sequence[Card]) -> tuple[int, ...]:
    """Compute exact 5-card `HandRank.value` from 5, 6, or 7 cards without enumerating combinations."""
    rc: dict[int, int] = {}
    sc: dict[str, list[int]] = {}
    rank_mask = 0
    for c in cards:
        r = c.rank
        rc[r] = rc.get(r, 0) + 1
        s_list = sc.get(c.suit)
        if s_list is None:
            sc[c.suit] = [r]
        else:
            s_list.append(r)
        rank_mask |= 1 << r
        if r == 14:
            rank_mask |= 1 << 1

    flush_ranks: list[int] | None = None
    for s_list in sc.values():
        if len(s_list) >= 5:
            flush_ranks = sorted(s_list, reverse=True)
            break

    if flush_ranks is not None:
        flush_mask = 0
        for r in flush_ranks:
            flush_mask |= 1 << r
            if r == 14:
                flush_mask |= 1 << 1
        sf_high = _best_straight_high(flush_mask)
        if sf_high == 14:
            return (int(HandCategory.ROYAL_FLUSH), 14)
        if sf_high > 0:
            return (int(HandCategory.STRAIGHT_FLUSH), sf_high)

    quads: list[int] = []
    trips: list[int] = []
    pairs: list[int] = []
    for r, cnt in rc.items():
        if cnt == 4:
            quads.append(r)
        elif cnt == 3:
            trips.append(r)
        elif cnt == 2:
            pairs.append(r)

    all_ranks = sorted(rc.keys(), reverse=True)
    if len(trips) > 1:
        trips.sort(reverse=True)
    if len(pairs) > 1:
        pairs.sort(reverse=True)

    if quads:
        q = quads[0]
        kicker = next(r for r in all_ranks if r != q)
        return (int(HandCategory.FOUR_OF_A_KIND), q, kicker)

    if trips:
        if len(trips) >= 2:
            return (int(HandCategory.FULL_HOUSE), trips[0], trips[1])
        if pairs:
            return (int(HandCategory.FULL_HOUSE), trips[0], pairs[0])

    if flush_ranks is not None:
        return (int(HandCategory.FLUSH), *flush_ranks[:5])

    st_high = _best_straight_high(rank_mask)
    if st_high > 0:
        return (int(HandCategory.STRAIGHT), st_high)

    if trips:
        t = trips[0]
        kickers = [r for r in all_ranks if r != t][:2]
        return (int(HandCategory.THREE_OF_A_KIND), t, *kickers)

    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = next(r for r in all_ranks if r != p1 and r != p2)
        return (int(HandCategory.TWO_PAIR), p1, p2, kicker)

    if pairs:
        p = pairs[0]
        kickers = [r for r in all_ranks if r != p][:3]
        return (int(HandCategory.ONE_PAIR), p, *kickers)

    return (int(HandCategory.HIGH_CARD), *all_ranks[:5])


@lru_cache(maxsize=65_536)
def postflop_combo_strength(hand: tuple[Card, Card], board: tuple[Card, ...]) -> float:
    """Fast deterministic postflop hand strength in [0.05, 0.999] on `board`."""
    if not board:
        return COMBO_PREFLOP_STRENGTH[hand]
    cards = (*hand, *board)

    rc: dict[int, int] = {}
    sc: dict[str, int] = {}
    rank_mask = 0
    for c in cards:
        r = c.rank
        rc[r] = rc.get(r, 0) + 1
        sc[c.suit] = sc.get(c.suit, 0) + 1
        rank_mask |= 1 << r
        if r == 14:
            rank_mask |= 1 << 1

    flush_suit: str | None = None
    for s, cnt in sc.items():
        if cnt >= 5:
            flush_suit = s
            break

    quads_rank = 0
    trips_ranks: list[int] = []
    pair_ranks: list[int] = []
    max_rank = 2
    for r, cnt in rc.items():
        max_rank = max(max_rank, r)
        if cnt == 4:
            quads_rank = r
        elif cnt == 3:
            trips_ranks.append(r)
        elif cnt == 2:
            pair_ranks.append(r)

    cat = HandCategory.HIGH_CARD
    primary_tb = max_rank

    if flush_suit is not None:
        flush_mask = 0
        flush_high = 2
        for c in cards:
            if c.suit == flush_suit:
                r = c.rank
                flush_high = max(flush_high, r)
                flush_mask |= 1 << r
                if r == 14:
                    flush_mask |= 1 << 1
        sf_high = _best_straight_high(flush_mask)
        if sf_high == 14:
            cat = HandCategory.ROYAL_FLUSH
            primary_tb = 14
        elif sf_high > 0:
            cat = HandCategory.STRAIGHT_FLUSH
            primary_tb = sf_high
        elif quads_rank > 0:
            cat = HandCategory.FOUR_OF_A_KIND
            primary_tb = quads_rank
        elif trips_ranks and (len(trips_ranks) >= 2 or pair_ranks):
            cat = HandCategory.FULL_HOUSE
            primary_tb = max(trips_ranks)
        else:
            cat = HandCategory.FLUSH
            primary_tb = flush_high
    elif quads_rank > 0:
        cat = HandCategory.FOUR_OF_A_KIND
        primary_tb = quads_rank
    elif trips_ranks and (len(trips_ranks) >= 2 or pair_ranks):
        cat = HandCategory.FULL_HOUSE
        primary_tb = max(trips_ranks)
    else:
        st_high = _best_straight_high(rank_mask)
        if st_high > 0:
            cat = HandCategory.STRAIGHT
            primary_tb = st_high
        elif trips_ranks:
            cat = HandCategory.THREE_OF_A_KIND
            primary_tb = max(trips_ranks)
        elif len(pair_ranks) >= 2:
            cat = HandCategory.TWO_PAIR
            primary_tb = max(pair_ranks)
        elif pair_ranks:
            cat = HandCategory.ONE_PAIR
            primary_tb = pair_ranks[0]

    base_bands = {
        HandCategory.HIGH_CARD: 0.18,
        HandCategory.ONE_PAIR: 0.48,
        HandCategory.TWO_PAIR: 0.74,
        HandCategory.THREE_OF_A_KIND: 0.85,
        HandCategory.STRAIGHT: 0.91,
        HandCategory.FLUSH: 0.94,
        HandCategory.FULL_HOUSE: 0.975,
        HandCategory.FOUR_OF_A_KIND: 0.992,
        HandCategory.STRAIGHT_FLUSH: 0.998,
        HandCategory.ROYAL_FLUSH: 0.999,
    }
    base = base_bands.get(cat, 0.25)
    kicker_bonus = ((primary_tb - 2) / 12.0) * 0.14

    # Distinguish top pair / overpair from bottom pair on board
    if cat == HandCategory.ONE_PAIR:
        top_board = max((c.rank for c in board), default=2)
        if primary_tb > top_board:
            kicker_bonus += 0.14  # Overpair
        elif primary_tb == top_board and (
            hand[0].rank == top_board or hand[1].rank == top_board
        ):
            kicker_bonus += 0.09  # Top pair held in hole cards

    draw_bonus = 0.0
    if len(board) < 5 and cat < HandCategory.STRAIGHT:
        if max(sc.values(), default=0) >= 4:
            draw_bonus += 0.11
        for _, win in _STRAIGHT_WINDOWS:
            if (rank_mask & win).bit_count() == 4:
                draw_bonus += 0.06
                break

    return max(0.05, min(0.999, base + kicker_bonus + draw_bonus))


@lru_cache(maxsize=512)
def _board_strength_vector(board: tuple[Card, ...]) -> tuple[float, ...]:
    """Precompute postflop strength for all 1,326 combos on `board` once per unique board."""
    if not board:
        return PREFLOP_STRENGTH_VECTOR
    b_set = set(board)
    return tuple(
        0.0
        if (combo[0] in b_set or combo[1] in b_set)
        else postflop_combo_strength(combo, board)
        for combo in ALL_COMBOS
    )


@dataclass(slots=True)
class ShowdownCalibration:
    """Empirical calibration of opponent action likelihoods learned at showdown."""

    showdowns_recorded: int = 0
    aggressive_showdowns: int = 0
    passive_showdowns: int = 0
    bluff_showdowns: int = 0  # Aggressive river/postflop line with weak hand (< 0.42)
    value_showdowns: int = 0  # Aggressive line with strong hand (>= 0.62)
    trap_showdowns: int = 0  # Passive line with monster hand (>= 0.80)
    aggressive_strength_sum: float = 0.0
    passive_strength_sum: float = 0.0

    @property
    def bluff_frequency_at_showdown(self) -> float:
        # Prior: 20% of aggressive showdowns are bluffs (strength 4)
        return (0.8 + self.bluff_showdowns) / (4.0 + self.aggressive_showdowns)

    @property
    def mean_aggressive_showdown_strength(self) -> float:
        # Prior mean: 0.68 (strength 4)
        return (4.0 * 0.68 + self.aggressive_strength_sum) / (
            4.0 + self.aggressive_showdowns
        )

    @property
    def mean_passive_showdown_strength(self) -> float:
        # Prior mean: 0.46 (strength 4)
        return (4.0 * 0.46 + self.passive_strength_sum) / (4.0 + self.passive_showdowns)

    @property
    def trap_frequency(self) -> float:
        return (0.3 + self.trap_showdowns) / (4.0 + self.passive_showdowns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "showdowns_recorded": self.showdowns_recorded,
            "aggressive_showdowns": self.aggressive_showdowns,
            "passive_showdowns": self.passive_showdowns,
            "bluff_showdowns": self.bluff_showdowns,
            "value_showdowns": self.value_showdowns,
            "trap_showdowns": self.trap_showdowns,
            "aggressive_strength_sum": round(self.aggressive_strength_sum, 6),
            "passive_strength_sum": round(self.passive_strength_sum, 6),
            "bluff_frequency_at_showdown": round(self.bluff_frequency_at_showdown, 6),
            "mean_aggressive_showdown_strength": round(
                self.mean_aggressive_showdown_strength, 6
            ),
            "mean_passive_showdown_strength": round(
                self.mean_passive_showdown_strength, 6
            ),
            "trap_frequency": round(self.trap_frequency, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShowdownCalibration:
        return cls(
            showdowns_recorded=int(data.get("showdowns_recorded", 0)),
            aggressive_showdowns=int(data.get("aggressive_showdowns", 0)),
            passive_showdowns=int(data.get("passive_showdowns", 0)),
            bluff_showdowns=int(data.get("bluff_showdowns", 0)),
            value_showdowns=int(data.get("value_showdowns", 0)),
            trap_showdowns=int(data.get("trap_showdowns", 0)),
            aggressive_strength_sum=float(data.get("aggressive_strength_sum", 0.0)),
            passive_strength_sum=float(data.get("passive_strength_sum", 0.0)),
        )


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(40.0, x))
        return 1.0 / (1.0 + z)
    z = math.exp(max(-40.0, x))
    return z / (1.0 + z)


class BayesianRangeModel:
    """Maintains and updates P(H | A_1, ..., A_t, B) over all 1,326 Hold'em holdings."""

    def __init__(
        self,
        *,
        calibration: ShowdownCalibration | None = None,
    ) -> None:
        self.calibration = calibration or ShowdownCalibration()
        self.hand_number: int = 0
        self.street: str = "preflop"
        self.hero_hole: tuple[Card, ...] = ()
        self.community_cards: tuple[Card, ...] = ()
        self.dead_cards: set[Card] = set()
        self.is_folded: bool = False

        self.prior_weights: dict[tuple[Card, Card], float] = {}
        self.posterior_weights: dict[tuple[Card, Card], float] = {}
        self.street_entry_weights: dict[tuple[Card, Card], float] = {}
        self._applied_events: set[tuple[int, int]] = set()
        self._calibrated_hands: set[int] = set()
        self._update_steps: list[RangeUpdateStep] = []

        self.start_hand(hand_number=0)

    def reset_all(self) -> None:
        """Reset both current-hand beliefs and cross-hand showdown calibration."""
        self.calibration = ShowdownCalibration()
        self._calibrated_hands.clear()
        self.start_hand(hand_number=0)

    def start_hand(
        self,
        *,
        hand_number: int = 0,
        hero_hole: Sequence[Card] = (),
        community_cards: Sequence[Card] = (),
    ) -> None:
        """Initialize a uniform prior over all non-dead 2-card holdings for a new hand."""
        self.hand_number = int(hand_number)
        self.street = "preflop"
        self.is_folded = False
        self._applied_events.clear()
        self._update_steps.clear()
        self.hero_hole = tuple(hero_hole)
        self.community_cards = tuple(community_cards)
        self.dead_cards = set(self.hero_hole) | set(self.community_cards)

        valid_count = sum(
            1
            for combo in ALL_COMBOS
            if combo[0] not in self.dead_cards and combo[1] not in self.dead_cards
        )
        uniform_p = 1.0 / max(1, valid_count)
        for combo in ALL_COMBOS:
            if combo[0] in self.dead_cards or combo[1] in self.dead_cards:
                self.prior_weights[combo] = 0.0
                self.posterior_weights[combo] = 0.0
                self.street_entry_weights[combo] = 0.0
            else:
                self.prior_weights[combo] = uniform_p
                self.posterior_weights[combo] = uniform_p
                self.street_entry_weights[combo] = uniform_p

    def condition_on_cards(
        self,
        *,
        hero_hole: Sequence[Card] | None = None,
        community_cards: Sequence[Card] | None = None,
    ) -> None:
        """Apply exact card removal for newly observed hero hole cards or board cards."""
        prev_street = self.street
        if hero_hole is not None:
            self.hero_hole = tuple(hero_hole)
        if community_cards is not None:
            self.community_cards = tuple(community_cards)
            n_board = len(self.community_cards)
            if n_board == 3:
                self.street = "flop"
            elif n_board == 4:
                self.street = "turn"
            elif n_board >= 5:
                self.street = "river"

        new_dead = set(self.hero_hole) | set(self.community_cards)
        street_changed = self.street != prev_street
        if new_dead == self.dead_cards and not street_changed:
            return
        self.dead_cards = new_dead

        for combo in ALL_COMBOS:
            if combo[0] in self.dead_cards or combo[1] in self.dead_cards:
                self.prior_weights[combo] = 0.0
                self.posterior_weights[combo] = 0.0
                self.street_entry_weights[combo] = 0.0

        self._normalize_dict(self.prior_weights)
        self._normalize_dict(self.posterior_weights)
        self._normalize_dict(self.street_entry_weights)

    def _normalize_dict(self, weights: dict[tuple[Card, Card], float]) -> None:
        total = sum(weights.values())
        if total > 0.0:
            inv = 1.0 / total
            for k in weights:
                weights[k] *= inv
        else:
            # Recover uniform over all non-dead combinations
            valid = [
                c
                for c in ALL_COMBOS
                if c[0] not in self.dead_cards and c[1] not in self.dead_cards
            ]
            u = 1.0 / max(1, len(valid))
            for c in ALL_COMBOS:
                weights[c] = (
                    u
                    if (c[0] not in self.dead_cards and c[1] not in self.dead_cards)
                    else 0.0
                )

    def action_likelihood(
        self,
        hand_strength: float,
        event: ObservedActionEvent,
        stats: OpponentStatsSnapshot | None = None,
    ) -> float:
        """Compute P(A | H, context, theta_opp) in [0.02, 1.0] for a holding of strength s."""
        s = max(0.01, min(0.999, hand_strength))
        kind = event.action_type
        street = event.street
        pot_ratio = max(0.0, min(4.0, event.pot_ratio))

        # Extract behavioral parameters (with sensible equilibrium defaults)
        if stats is not None:
            vpip = (
                stats.vpip_button.smoothed_value
                if event.position == "button"
                else stats.vpip_big_blind.smoothed_value
            )
            pfr = (
                stats.pfr_button.smoothed_value
                if event.position == "button"
                else stats.pfr_big_blind.smoothed_value
            )
            three_bet = stats.three_bet_freq.smoothed_value
            afq = stats.aggression_frequency.smoothed_value
        else:
            vpip = 0.50 if event.position == "button" else 0.38
            pfr = 0.34 if event.position == "button" else 0.18
            three_bet = 0.10
            afq = 0.38

        bluff_rate = self.calibration.bluff_frequency_at_showdown
        trap_rate = self.calibration.trap_frequency
        agg_center = self.calibration.mean_aggressive_showdown_strength

        if street == "preflop":
            # Thresholds in preflop strength space [0.05..0.99]:
            # A player with VPIP v plays roughly the top v fraction of hands
            vpip_thresh = max(0.12, min(0.82, 0.90 - 0.72 * vpip))
            pfr_thresh = max(
                vpip_thresh + 0.04,
                min(0.88, 0.92 - 0.68 * pfr),
            )
            if event.facing_raise_count >= 1:
                # 3-bet or 4-bet threshold
                raise_thresh = max(
                    0.50,
                    min(0.92, 0.94 - 0.75 * three_bet + 0.04 * pot_ratio),
                )
            else:
                raise_thresh = pfr_thresh + 0.03 * min(2.0, pot_ratio)

            if kind in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            ):
                size_steepness = 9.0 + 3.0 * min(2.0, pot_ratio)
                value_prob = _sigmoid(size_steepness * (s - raise_thresh))
                if event.facing_raise_count >= 1:
                    # 3-bet / 4-bet: tighter bluff distribution governed by 3-bet frequency
                    bluff_floor = max(
                        0.04,
                        min(0.70, 0.60 * three_bet + 0.25 * bluff_rate),
                    )
                else:
                    # Open raise / unraised pot: scale bluff floor with PFR and VPIP
                    bluff_floor = max(
                        0.06,
                        min(0.85, 0.60 * pfr + 0.25 * vpip + 0.15 * bluff_rate),
                    )
                return max(0.04, (1.0 - bluff_floor) * value_prob + bluff_floor)

            if kind == ActionType.CALL.value:
                # Calling preflop: above VPIP threshold, but capped if opponent usually raises top hands
                enter_prob = _sigmoid(10.0 * (s - vpip_thresh))
                raise_freq = min(0.75, 1.2 * pfr)
                not_3bet_prob = 1.0 - (1.0 - trap_rate) * raise_freq * _sigmoid(
                    10.0 * (s - (raise_thresh + 0.06))
                )
                return max(0.05, enter_prob * not_3bet_prob)

            if kind == ActionType.CHECK.value:
                # Checking big blind option: did not raise preflop
                not_raise = 1.0 - 0.75 * _sigmoid(9.0 * (s - pfr_thresh))
                return max(0.06, not_raise)

            if kind == ActionType.FOLD.value:
                fold_prob = _sigmoid(10.0 * (vpip_thresh - s))
                return max(0.04, fold_prob)

        # Postflop (flop, turn, river)
        if kind in (
            ActionType.BET.value,
            ActionType.RAISE.value,
            ActionType.ALL_IN.value,
        ):
            # Aggressive postflop action: polarized + value-weighted by bet size and AFq
            # Higher AFq lowers the value threshold; larger bet size increases polarization/strength
            value_thresh = max(
                0.35,
                min(
                    0.88,
                    agg_center - 0.30 * (afq - 0.38) + 0.06 * min(2.0, pot_ratio),
                ),
            )
            if kind in (ActionType.RAISE.value, ActionType.ALL_IN.value):
                value_thresh = min(0.90, value_thresh + 0.07)

            value_component = _sigmoid(10.0 * (s - value_thresh))
            # Polarized bluff component on weak hands / draws when opponent is aggressive
            agg_floor = max(0.05, min(0.80, afq - 0.15))
            bluff_weight = max(
                0.06,
                min(
                    0.80,
                    0.50 * afq + 0.35 * bluff_rate + 0.04 * min(1.5, pot_ratio),
                ),
            )
            weak_bluff_region = _sigmoid(8.0 * (0.42 - s))
            pattern_likelihood = (
                1.0 - bluff_weight
            ) * value_component + bluff_weight * (
                0.50 * weak_bluff_region + 0.50 * value_component
            )
            likelihood = (1.0 - agg_floor) * pattern_likelihood + agg_floor
            return max(0.05, min(1.0, likelihood))

        if kind == ActionType.CALL.value:
            # Calling a bet postflop: medium-to-strong bluff-catchers, draws, or slowplayed monsters
            call_thresh = max(
                0.22,
                min(0.75, 0.38 + 0.14 * min(1.5, pot_ratio) - 0.18 * (vpip - 0.45)),
            )
            above_floor = _sigmoid(9.5 * (s - call_thresh))
            call_floor = max(0.05, min(0.40, 0.35 * max(0.0, vpip - 0.35)))
            effective_above = (1.0 - call_floor) * above_floor + call_floor
            # Very strong hands raise some fraction of the time unless trapping or passive
            raise_freq = min(0.65, 1.2 * afq)
            raise_cap = 1.0 - (1.0 - trap_rate) * raise_freq * _sigmoid(
                10.0 * (s - 0.86)
            )
            return max(0.05, effective_above * raise_cap)

        if kind == ActionType.CHECK.value:
            # Checking postflop: weak/medium showdown value or occasional slowplay trap
            not_bet = 1.0 - (1.0 - trap_rate) * 0.72 * _sigmoid(
                9.0 * (s - (0.68 - 0.20 * (afq - 0.38)))
            )
            return max(0.05, not_bet)

        if kind == ActionType.FOLD.value:
            fold_thresh = max(0.25, min(0.70, 0.40 + 0.12 * min(1.5, pot_ratio)))
            return max(0.04, _sigmoid(9.0 * (fold_thresh - s)))

        return 0.5

    def update_from_action(
        self,
        event: ObservedActionEvent,
        *,
        stats: OpponentStatsSnapshot | None = None,
        community_cards: Sequence[Card] | None = None,
    ) -> bool:
        """Apply a single observed opponent action to the posterior distribution.

        Returns `True` if the update was applied, or `False` if the event was
        not an opponent action or had already been applied (preventing duplicate updates).
        """
        if not event.is_opponent:
            return False

        event_key = (event.hand_number, event.event_index)
        if event_key in self._applied_events:
            return False
        self._applied_events.add(event_key)

        if community_cards is not None:
            self.condition_on_cards(community_cards=community_cards)
        if event.street != self.street:
            self.street_entry_weights = dict(self.posterior_weights)
            self.street = event.street

        board = self.community_cards if event.street != "preflop" else ()
        strength_vec = _board_strength_vector(board)

        if event.action_type == ActionType.FOLD.value:
            prior_mean = self.mean_strength(use_prior=False)
            # Handle fold without pretending the folded hand is known:
            # retain pre-fold posterior weights and mark range as folded.
            self.is_folded = True
            top_after = self.top_class_probabilities(limit=5)
            self._update_steps.append(
                RangeUpdateStep(
                    step_index=len(self._update_steps) + 1,
                    street=event.street,
                    action_type=event.action_type,
                    amount=event.amount,
                    pot_ratio=event.pot_ratio,
                    prior_mean_strength=prior_mean,
                    posterior_mean_strength=prior_mean,
                    l1_shift=0.0,
                    top_classes_after=top_after,
                    description=(
                        f"Opponent folded on {event.street}; pre-fold posterior range "
                        "retained without assuming folded hole cards are known."
                    ),
                )
            )
            return True

        before_weights = dict(self.posterior_weights)
        prior_mean = 0.0
        new_total = 0.0
        weighted_strength_sum = 0.0

        for idx, combo in enumerate(ALL_COMBOS):
            w_prev = self.posterior_weights[combo]
            if w_prev <= 0.0:
                continue
            strength = strength_vec[idx]
            prior_mean += w_prev * strength
            lik = self.action_likelihood(strength, event, stats)
            w_new = w_prev * lik
            self.posterior_weights[combo] = w_new
            new_total += w_new
            weighted_strength_sum += w_new * strength

        if new_total > 0.0:
            inv = 1.0 / new_total
            for combo in self.posterior_weights:
                self.posterior_weights[combo] *= inv
            post_mean = weighted_strength_sum * inv
        else:
            self.posterior_weights = before_weights
            post_mean = prior_mean

        l1_shift = 0.5 * sum(
            abs(self.posterior_weights[c] - before_weights[c]) for c in ALL_COMBOS
        )
        top_after = self.top_class_probabilities(limit=5)
        size_note = (
            f" ({event.pot_ratio:.2f}x pot)"
            if event.action_type
            in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
            and event.pot_ratio > 0
            else ""
        )
        desc = (
            f"{event.street.capitalize()} {event.action_type}{size_note} in {event.position}: "
            f"mean range strength {prior_mean:.3f} -> {post_mean:.3f} (TV shift {l1_shift:.3f})."
        )
        self._update_steps.append(
            RangeUpdateStep(
                step_index=len(self._update_steps) + 1,
                street=event.street,
                action_type=event.action_type,
                amount=event.amount,
                pot_ratio=event.pot_ratio,
                prior_mean_strength=prior_mean,
                posterior_mean_strength=post_mean,
                l1_shift=l1_shift,
                top_classes_after=top_after,
                description=desc,
            )
        )
        return True

    def record_showdown(
        self,
        *,
        hand_number: int,
        revealed_cards: tuple[Card, Card],
        community_cards: Sequence[Card],
        opponent_actions: Sequence[ObservedActionEvent],
    ) -> bool:
        """Update behavioral likelihood calibration from a legitimate showdown.

        Idempotent per `hand_number`: returns False if `hand_number` was already
        calibrated.
        """
        if hand_number in self._calibrated_hands:
            return False
        if len(revealed_cards) != 2:
            raise ValueError("revealed_cards must contain exactly 2 cards")

        self._calibrated_hands.add(int(hand_number))
        combo = tuple(
            sorted(
                revealed_cards,
                key=lambda c: (c.rank, SUIT_INDEX[c.suit]),
                reverse=True,
            )
        )
        board = tuple(community_cards)
        final_strength = (
            postflop_combo_strength(combo, board)
            if board
            else COMBO_PREFLOP_STRENGTH[combo]
        )

        postflop_agg = any(
            a.is_opponent
            and a.street != "preflop"
            and a.action_type
            in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
            for a in opponent_actions
        )
        any_agg = postflop_agg or any(
            a.is_opponent
            and a.action_type
            in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
            for a in opponent_actions
        )

        self.calibration.showdowns_recorded += 1
        if any_agg:
            self.calibration.aggressive_showdowns += 1
            self.calibration.aggressive_strength_sum += final_strength
            if final_strength < 0.42:
                self.calibration.bluff_showdowns += 1
            elif final_strength >= 0.62:
                self.calibration.value_showdowns += 1
        else:
            self.calibration.passive_showdowns += 1
            self.calibration.passive_strength_sum += final_strength
            if final_strength >= 0.80:
                self.calibration.trap_showdowns += 1

        return True

    def mean_strength(self, *, use_prior: bool = False) -> float:
        weights = self.prior_weights if use_prior else self.posterior_weights
        strength_vec = _board_strength_vector(self.community_cards)
        total = 0.0
        for idx, combo in enumerate(ALL_COMBOS):
            w = weights[combo]
            if w > 0.0:
                total += w * strength_vec[idx]
        return total

    def position_prior_weights(
        self,
        position: str = "button",
        stats: OpponentStatsSnapshot | None = None,
    ) -> dict[tuple[Card, Card], float]:
        """Construct a position-conditioned range prior over non-dead holdings."""
        if stats is not None:
            vpip = (
                stats.vpip_button.smoothed_value
                if position == "button"
                else stats.vpip_big_blind.smoothed_value
            )
        else:
            vpip = 0.54 if position == "button" else 0.36
        vpip_thresh = max(0.12, min(0.82, 0.90 - 0.72 * vpip))

        weights: dict[tuple[Card, Card], float] = {}
        total = 0.0
        for idx, combo in enumerate(ALL_COMBOS):
            if combo[0] in self.dead_cards or combo[1] in self.dead_cards:
                weights[combo] = 0.0
                continue
            s = PREFLOP_STRENGTH_VECTOR[idx]
            w = max(0.05, _sigmoid(8.0 * (s - vpip_thresh)))
            weights[combo] = w
            total += w
        if total > 0.0:
            inv = 1.0 / total
            for c in weights:
                weights[c] *= inv
        else:
            return dict(self.prior_weights)
        return weights

    def top_class_probabilities(
        self, *, limit: int = 10
    ) -> tuple[tuple[str, float], ...]:
        class_mass: dict[str, float] = {}
        for combo, w in self.posterior_weights.items():
            if w <= 0.0:
                continue
            cls_name = COMBO_TO_CLASS[combo]
            class_mass[cls_name] = class_mass.get(cls_name, 0.0) + w
        ranked = sorted(class_mass.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return tuple(ranked)

    def estimate_equity_against_range(
        self,
        hero_hole: tuple[Card, Card],
        community_cards: Sequence[Card],
        *,
        simulations: int = 48,
        rng: random.Random | None = None,
    ) -> tuple[float, float]:
        """Compute `(range_weighted_equity, uniform_equity)` via paired Monte Carlo rollouts."""
        self.condition_on_cards(hero_hole=hero_hole, community_cards=community_cards)
        board = tuple(community_cards)
        active_combos = [c for c, w in self.posterior_weights.items() if w > 0.0]
        if not active_combos:
            return 0.5, 0.5

        local_rng = rng or random.Random(42)
        needed = max(0, 5 - len(board))
        dead_base = set(hero_hole) | set(board)
        deck_pool = [c for c in ALL_CARDS if c not in dead_base]

        n_active = len(active_combos)
        post_probs = [self.posterior_weights[c] for c in active_combos]

        k = min(max(16, int(simulations)), 120)
        chosen_indices = local_rng.choices(range(n_active), weights=post_probs, k=k)
        uniform_indices = local_rng.choices(range(n_active), k=k)

        range_wins = 0.0
        static_h_rank = (
            fast_hand_rank_value((*hero_hole, *board)) if needed == 0 else ()
        )
        for idx in chosen_indices:
            opp = active_combos[idx]
            if needed == 0:
                h_rank = static_h_rank
                o_rank = fast_hand_rank_value((*opp, *board))
            else:
                rem = [c for c in deck_pool if c != opp[0] and c != opp[1]]
                runout = local_rng.sample(rem, needed)
                full_board = (*board, *runout)
                h_rank = fast_hand_rank_value((*hero_hole, *full_board))
                o_rank = fast_hand_rank_value((*opp, *full_board))
            if h_rank > o_rank:
                range_wins += 1.0
            elif h_rank == o_rank:
                range_wins += 0.5

        uniform_wins = 0.0
        for idx in uniform_indices:
            opp = active_combos[idx]
            if needed == 0:
                h_rank = static_h_rank
                o_rank = fast_hand_rank_value((*opp, *board))
            else:
                rem = [c for c in deck_pool if c != opp[0] and c != opp[1]]
                runout = local_rng.sample(rem, needed)
                full_board = (*board, *runout)
                h_rank = fast_hand_rank_value((*hero_hole, *full_board))
                o_rank = fast_hand_rank_value((*opp, *full_board))
            if h_rank > o_rank:
                uniform_wins += 1.0
            elif h_rank == o_rank:
                uniform_wins += 0.5

        return range_wins / k, uniform_wins / k

    def evaluate_ground_truth_hand(
        self,
        true_hole_cards: tuple[Card, Card],
        *,
        position: str = "button",
        stats: OpponentStatsSnapshot | None = None,
    ) -> dict[str, float]:
        """Evaluate posterior quality on a revealed ground-truth hand against multiple baselines."""
        combo = tuple(
            sorted(
                true_hole_cards,
                key=lambda c: (c.rank, SUIT_INDEX[c.suit]),
                reverse=True,
            )
        )
        true_cls = COMBO_TO_CLASS[combo]
        p_post = self.posterior_weights.get(combo, 0.0)
        p_prior = self.prior_weights.get(combo, 0.0)
        p_prev = self.street_entry_weights.get(combo, p_prior)
        pos_weights = self.position_prior_weights(position=position, stats=stats)
        p_pos = pos_weights.get(combo, p_prior)
        ratio = (p_post / p_prior) if p_prior > 0 else 1.0

        cls_post: dict[str, float] = {}
        cls_prior: dict[str, float] = {}
        cls_pos: dict[str, float] = {}
        cls_prev: dict[str, float] = {}
        for c, w in self.posterior_weights.items():
            k = COMBO_TO_CLASS[c]
            cls_post[k] = cls_post.get(k, 0.0) + w
            cls_prior[k] = cls_prior.get(k, 0.0) + self.prior_weights.get(c, 0.0)
            cls_pos[k] = cls_pos.get(k, 0.0) + pos_weights.get(c, 0.0)
            cls_prev[k] = cls_prev.get(k, 0.0) + self.street_entry_weights.get(c, 0.0)

        brier_post = sum(
            (prob - (1.0 if k == true_cls else 0.0)) ** 2
            for k, prob in cls_post.items()
        )
        brier_prior = sum(
            (prob - (1.0 if k == true_cls else 0.0)) ** 2
            for k, prob in cls_prior.items()
        )
        brier_pos = sum(
            (prob - (1.0 if k == true_cls else 0.0)) ** 2 for k, prob in cls_pos.items()
        )
        brier_prev = sum(
            (prob - (1.0 if k == true_cls else 0.0)) ** 2
            for k, prob in cls_prev.items()
        )

        eps = 1e-6
        ll_post = -math.log(max(eps, cls_post.get(true_cls, 0.0)))
        ll_prior = -math.log(max(eps, cls_prior.get(true_cls, 0.0)))
        ll_pos = -math.log(max(eps, cls_pos.get(true_cls, 0.0)))
        ll_prev = -math.log(max(eps, cls_prev.get(true_cls, 0.0)))

        ranked_post = [
            k for k, _ in sorted(cls_post.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        ranked_prior = [
            k for k, _ in sorted(cls_prior.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        true_rank_post = (
            ranked_post.index(true_cls) + 1 if true_cls in ranked_post else 169
        )
        true_rank_prior = (
            ranked_prior.index(true_cls) + 1 if true_cls in ranked_prior else 169
        )

        return {
            "posterior_prob": p_post,
            "prior_prob": p_prior,
            "position_prior_prob": p_pos,
            "prev_street_prob": p_prev,
            "likelihood_ratio": ratio,
            "class_posterior_prob": cls_post.get(true_cls, 0.0),
            "class_prior_prob": cls_prior.get(true_cls, 0.0),
            "class_pos_prior_prob": cls_pos.get(true_cls, 0.0),
            "class_prev_street_prob": cls_prev.get(true_cls, 0.0),
            "brier_class_score": brier_post,
            "brier_prior_score": brier_prior,
            "brier_pos_prior_score": brier_pos,
            "brier_prev_street_score": brier_prev,
            "log_loss_post": ll_post,
            "log_loss_prior": ll_prior,
            "log_loss_pos_prior": ll_pos,
            "log_loss_prev_street": ll_prev,
            "true_class_rank": float(true_rank_post),
            "prior_class_rank": float(true_rank_prior),
            "top_1_hit": 1.0 if true_rank_post <= 1 else 0.0,
            "top_3_hit": 1.0 if true_rank_post <= 3 else 0.0,
            "top_5_hit": 1.0 if true_rank_post <= 5 else 0.0,
            "top_10_hit": 1.0 if true_rank_post <= 10 else 0.0,
            "top_25_hit": 1.0 if true_rank_post <= 25 else 0.0,
            "prior_top_10_hit": 1.0 if true_rank_prior <= 10 else 0.0,
            "prior_top_25_hit": 1.0 if true_rank_prior <= 25 else 0.0,
        }

    def snapshot(
        self, *, top_class_limit: int = 12, top_combo_limit: int = 10
    ) -> RangeSnapshot:
        """Build a rich `RangeSnapshot` for Policy Inspector visualization."""
        board = self.community_cards
        strength_vec = _board_strength_vector(board)
        active_combos = [c for c, w in self.posterior_weights.items() if w > 0.0]
        n_active = len(active_combos)
        n_eliminated = len(ALL_COMBOS) - n_active

        prior_mean = self.mean_strength(use_prior=True)
        post_mean = self.mean_strength(use_prior=False)
        l1_shift = 0.5 * sum(
            abs(self.posterior_weights[c] - self.prior_weights[c]) for c in ALL_COMBOS
        )

        entropy_bits = -sum(
            w * math.log2(w) for w in self.posterior_weights.values() if w > 0.0
        )
        max_entropy = math.log2(n_active) if n_active > 1 else 0.0
        concentration = (
            max(0.0, min(1.0, 1.0 - (entropy_bits / max_entropy)))
            if max_entropy > 0.0
            else 0.0
        )

        # Group by 169 canonical hand classes
        cls_counts: dict[str, int] = {}
        cls_prior: dict[str, float] = {}
        cls_post: dict[str, float] = {}
        cls_strength_sum: dict[str, float] = {}

        tier_prior = {
            "premium_strong": 0.0,
            "medium_playable": 0.0,
            "speculative_weak": 0.0,
        }
        tier_post = {
            "premium_strong": 0.0,
            "medium_playable": 0.0,
            "speculative_weak": 0.0,
        }

        for combo in active_combos:
            w_pr = self.prior_weights[combo]
            w_po = self.posterior_weights[combo]
            s = strength_vec[COMBO_INDEX[combo]]
            cls_name = COMBO_TO_CLASS[combo]
            cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1
            cls_prior[cls_name] = cls_prior.get(cls_name, 0.0) + w_pr
            cls_post[cls_name] = cls_post.get(cls_name, 0.0) + w_po
            cls_strength_sum[cls_name] = cls_strength_sum.get(cls_name, 0.0) + s

            tier_key = (
                "premium_strong"
                if s >= 0.68
                else ("medium_playable" if s >= 0.44 else "speculative_weak")
            )
            tier_prior[tier_key] += w_pr
            tier_post[tier_key] += w_po

        top_classes_sorted = sorted(
            cls_post.keys(),
            key=lambda k: (-cls_post[k], k),
        )[:top_class_limit]

        top_classes = tuple(
            RangeClassEstimate(
                hand_class=k,
                combo_count=cls_counts[k],
                prior_probability=cls_prior[k],
                posterior_probability=cls_post[k],
                probability_delta=cls_post[k] - cls_prior[k],
                mean_strength=cls_strength_sum[k] / max(1, cls_counts[k]),
            )
            for k in top_classes_sorted
        )

        top_combos_sorted = sorted(
            active_combos,
            key=lambda c: (-self.posterior_weights[c], COMBO_CODES[c]),
        )[:top_combo_limit]

        top_combos = tuple(
            RangeComboEstimate(
                cards=COMBO_CODES[c],
                hand_class=COMBO_TO_CLASS[c],
                prior_probability=self.prior_weights[c],
                posterior_probability=self.posterior_weights[c],
                strength=strength_vec[COMBO_INDEX[c]],
            )
            for c in top_combos_sorted
        )

        tier_breakdown = {
            t: {
                "prior": tier_prior[t],
                "posterior": tier_post[t],
                "delta": tier_post[t] - tier_prior[t],
            }
            for t in ("premium_strong", "medium_playable", "speculative_weak")
        }

        return RangeSnapshot(
            street=self.street,
            active_combo_count=n_active,
            eliminated_combo_count=n_eliminated,
            is_folded=self.is_folded,
            prior_mean_strength=prior_mean,
            posterior_mean_strength=post_mean,
            l1_shift_from_prior=l1_shift,
            entropy_bits=entropy_bits,
            max_entropy_bits=max_entropy,
            concentration=concentration,
            top_classes=top_classes,
            top_combos=top_combos,
            tier_breakdown=tier_breakdown,
            action_updates=tuple(self._update_steps),
            showdown_calibration=self.calibration.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration.to_dict(),
            "calibrated_hands": sorted(self._calibrated_hands),
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        if "calibration" in data and isinstance(data["calibration"], dict):
            self.calibration = ShowdownCalibration.from_dict(data["calibration"])
        self._calibrated_hands = {int(x) for x in data.get("calibrated_hands", [])}
