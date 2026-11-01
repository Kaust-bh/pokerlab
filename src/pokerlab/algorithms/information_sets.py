"""Protocols and interfaces for information-set abstraction and extraction."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pokerlab.agents.base import Observation


@runtime_checkable
class InformationSetCodec(Protocol):
    """Encodes game observations into stable information set keys."""

    def encode(self, observation: Observation) -> Any:
        """Map a player-visible observation to an information set key."""
        ...
