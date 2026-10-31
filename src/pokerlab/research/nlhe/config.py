"""Game and abstraction configuration for the abstracted HUNL prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pokerlab.agents.action_profiles import ACTION_PROFILES


@dataclass(frozen=True, slots=True)
class HUNLGameConfig:
    """One-hand HU game configuration.

    Chips are integer half-big-blinds by default, so the 100 BB stack and
    0.5/1 BB blinds are represented exactly as 200/1/2 chips.
    """

    starting_stack_bb: float = 100.0
    small_blind_bb: float = 0.5
    big_blind_bb: float = 1.0
    chips_per_bb: int = 2
    postflop_buckets: int = 20
    opponent_hand_samples: int = 128
    postflop_method: str = "sampled_strength"
    equity_samples: int = 64
    opponent_range: str = "uniform"
    action_profile: str = "legacy"

    def __post_init__(self) -> None:
        if self.chips_per_bb < 1:
            raise ValueError("chips_per_bb must be positive")
        if self.starting_stack_bb <= 0:
            raise ValueError("starting stack must be positive")
        if self.small_blind_bb <= 0 or self.big_blind_bb < self.small_blind_bb:
            raise ValueError("blinds must be positive and BB >= SB")
        for amount in (
            self.starting_stack_bb,
            self.small_blind_bb,
            self.big_blind_bb,
        ):
            if not math.isfinite(amount) or not math.isclose(
                amount * self.chips_per_bb,
                round(amount * self.chips_per_bb),
                abs_tol=1e-9,
            ):
                raise ValueError("stack and blinds must be representable in chips")
        if self.postflop_buckets < 2:
            raise ValueError("postflop_buckets must be at least two")
        if self.opponent_hand_samples < 1:
            raise ValueError("opponent_hand_samples must be positive")
        if self.postflop_method not in {"sampled_strength", "sampled_equity"}:
            raise ValueError(
                "postflop_method must be sampled_strength or sampled_equity"
            )
        if self.equity_samples < 1:
            raise ValueError("equity_samples must be positive")
        if self.opponent_range not in {"uniform", "premium"}:
            raise ValueError("opponent_range must be uniform or premium")
        if self.action_profile not in ACTION_PROFILES:
            raise ValueError(
                f"action_profile must be one of {', '.join(ACTION_PROFILES)}"
            )

    @property
    def starting_stack_chips(self) -> int:
        return round(self.starting_stack_bb * self.chips_per_bb)

    @property
    def small_blind_chips(self) -> int:
        return round(self.small_blind_bb * self.chips_per_bb)

    @property
    def big_blind_chips(self) -> int:
        return round(self.big_blind_bb * self.chips_per_bb)
