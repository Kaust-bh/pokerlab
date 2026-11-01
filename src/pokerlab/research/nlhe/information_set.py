"""Stable player-visible information-set keys for the HUNL abstraction."""

from __future__ import annotations

from dataclasses import dataclass

from pokerlab.agents.base import Observation
from pokerlab.research.nlhe.abstraction import _suit_signature, private_hand_bucket
from pokerlab.research.nlhe.actions import (
    AbstractAction,
    abstract_action_specs_from_observation,
)
from pokerlab.research.nlhe.config import HUNLGameConfig


@dataclass(frozen=True, slots=True)
class HUNLInformationSet:
    player: int
    street: str
    private_bucket: str
    board: tuple[tuple[int, str], ...]
    pot: int
    effective_stack: int
    street_contributions: tuple[int, int]
    history: tuple[tuple[str, bool, str, int | None], ...]
    legal_actions: tuple[AbstractAction, ...]

    def generalized_key(self) -> tuple:
        """Coarser key ignoring exact board cards to allow cross-board generalization."""
        return (
            self.player,
            self.street,
            self.private_bucket,
            self.history,
            self.legal_actions,
        )

    def _street_action_pattern(self) -> tuple:
        """Normalized betting sequence on the current street plus prior-street aggression."""
        prior_raises = sum(
            1
            for st, _btn, act_type, _amt in self.history
            if st != self.street and act_type in ("bet", "raise", "all_in")
        )
        current_actions = tuple(
            (btn, act_type)
            for st, btn, act_type, _amt in self.history
            if st == self.street
        )
        return (min(prior_raises, 3), current_actions)

    def _primary_postflop_bucket(self) -> str:
        """Extract core postflop bucket without the 169-class preflop prefix."""
        parts = self.private_bucket.split("|")
        if len(parts) <= 1:
            return self.private_bucket
        for part in parts[1:]:
            if part.startswith(("equity:", "relative:")):
                return part
        return parts[1]

    def coarse_generalized_key(self) -> tuple:
        """Board-agnostic bucket-and-betting-context key for hierarchical generalization."""
        if self.street == "preflop":
            return (
                self.player,
                self.street,
                self.private_bucket,
                self._street_action_pattern(),
                self.legal_actions,
            )
        return (
            self.player,
            self.street,
            self._primary_postflop_bucket(),
            self._street_action_pattern(),
            self.legal_actions,
        )

    def postflop_abstraction_key(self, board_mode: str = "exact") -> tuple:
        """Abstract postflop equivalence key for cross-deal regret sharing."""
        if self.street == "preflop":
            return (self,)
        if board_mode == "isomorphic":
            board_feature: tuple = (
                "iso",
                any(suit == "s0" for _rank, suit in self.board),
            )
        elif board_mode == "texture":
            board_feature = ("tex", self.board)
        else:
            board_feature = (
                "exact",
                self.board[0][1] if self.board else "",
            )
        return (
            self.player,
            self.street,
            self._primary_postflop_bucket(),
            board_feature,
            self._street_action_pattern(),
            self.legal_actions,
        )


def canonicalize_board_representation(
    cards: tuple, mode: str = "exact", hole_cards: tuple = ()
) -> tuple[tuple[int, str], ...]:
    """Map board cards into exact, suit-isomorphic, or texture-abstracted tuples.

    - ``exact``: preserves exact ``(rank, suit)`` tuples in deal order.
    - ``isomorphic``: canonicalizes suits jointly across ``hole_cards`` and ``cards``
      using rank-only street signatures (strictly invariant under all 24 suit
      permutations), sorts the 3 flop cards by ``(-rank, canonical_suit)``, and
      preserves turn (index 3) and river (index 4) street order.
    - ``texture``: coarse lossy compression into ``((street_len, texture_tag),)``
      recording high-card rank, pairedness, and max suit count.
    """
    if not cards:
        return ()
    if mode == "exact":
        return tuple((card.rank, card.suit) for card in cards)
    if mode == "isomorphic":
        present_suits: list[str] = []
        for card in (*hole_cards, *cards):
            if card.suit not in present_suits:
                present_suits.append(card.suit)
        present_suits.sort(
            key=lambda s: _suit_signature(s, tuple(hole_cards), tuple(cards)),
            reverse=True,
        )
        suit_order = {suit: f"s{idx}" for idx, suit in enumerate(present_suits)}
        flop = tuple(
            sorted(
                ((card.rank, suit_order[card.suit]) for card in cards[:3]),
                key=lambda item: (-item[0], item[1]),
            )
        )
        turn_river = tuple((card.rank, suit_order[card.suit]) for card in cards[3:])
        return (*flop, *turn_river)
    if mode == "texture":
        ranks = sorted([card.rank for card in cards], reverse=True)
        high = ranks[0]
        suit_counts: dict[str, int] = {}
        for c in cards:
            suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
        max_suit = max(suit_counts.values()) if suit_counts else 0
        is_paired = 1 if len(ranks) != len(set(ranks)) else 0
        texture_tag = f"n{len(cards)}_h{high}_p{is_paired}_f{max_suit}"
        return ((high, texture_tag),)
    return tuple((card.rank, card.suit) for card in cards)


def information_set_from_observation(
    observation: Observation,
    history: tuple[tuple[str, bool, str, int | None], ...],
    config: HUNLGameConfig,
) -> HUNLInformationSet:
    """Construct a key without reading GameState or opponent private cards."""
    own, opponent = (
        observation.street_contribution,
        observation.opponent_street_contribution,
    )
    contributions = (own, opponent) if observation.is_button else (opponent, own)
    specs = abstract_action_specs_from_observation(observation)
    board_mode = getattr(config, "board_abstraction", "exact")
    return HUNLInformationSet(
        # Role (button or big blind), not process-local player identity.
        player=0 if observation.is_button else 1,
        street=observation.street,
        private_bucket=private_hand_bucket(
            observation.hole_cards, observation.community_cards, config
        ),
        board=canonicalize_board_representation(
            observation.community_cards,
            mode=board_mode,
            hole_cards=observation.hole_cards,
        ),
        pot=observation.pot,
        effective_stack=observation.effective_stack,
        street_contributions=contributions,
        history=history,
        legal_actions=tuple(spec.abstract for spec in specs),
    )
