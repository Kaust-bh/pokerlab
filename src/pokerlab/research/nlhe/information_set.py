"""Stable player-visible information-set keys for the HUNL abstraction."""

from __future__ import annotations

from dataclasses import dataclass

from pokerlab.agents.base import Observation
from pokerlab.research.nlhe.abstraction import private_hand_bucket
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
    return HUNLInformationSet(
        # Role (button or big blind), not process-local player identity.
        player=0 if observation.is_button else 1,
        street=observation.street,
        private_bucket=private_hand_bucket(
            observation.hole_cards, observation.community_cards, config
        ),
        board=tuple((card.rank, card.suit) for card in observation.community_cards),
        pot=observation.pot,
        effective_stack=observation.effective_stack,
        street_contributions=contributions,
        history=history,
        legal_actions=tuple(spec.abstract for spec in specs),
    )
