"""Contribution tracking and side-pot construction/distribution."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PotLayer:
    amount: int
    eligible: frozenset[str]


class Pot:
    """Builds pots from total contributions. Ties award odd chips by seat order."""

    def __init__(self) -> None:
        self.contributions: dict[str, int] = {}
        self.folded: set[str] = set()

    def contribute(self, player: str, amount: int) -> None:
        if amount < 0:
            raise ValueError("contribution cannot be negative")
        self.contributions[player] = self.contributions.get(player, 0) + amount

    def fold(self, player: str) -> None:
        if player not in self.contributions:
            raise ValueError("unknown player")
        self.folded.add(player)

    @property
    def total(self) -> int:
        return sum(self.contributions.values())

    def layers(self) -> list[PotLayer]:
        levels = sorted(set(self.contributions.values()) - {0})
        layers: list[PotLayer] = []
        previous = 0
        for level in levels:
            contributors = {
                p for p, amount in self.contributions.items() if amount >= level
            }
            amount = (level - previous) * len(contributors)
            eligible = frozenset(contributors - self.folded)
            if amount:
                layers.append(PotLayer(amount, eligible))
            previous = level
        return layers

    def distribute(
        self, winners: Mapping[str, object], seat_order: Sequence[str]
    ) -> dict[str, int]:
        """Award each layer to its strongest eligible rank; earlier seats get odd chips."""
        payouts = {p: 0 for p in self.contributions}
        order = {p: i for i, p in enumerate(seat_order)}
        for layer in self.layers():
            candidates = layer.eligible.intersection(winners)
            if not candidates:
                raise ValueError("pot layer has no eligible winner")
            best = max(winners[p] for p in candidates)
            tied = sorted(
                (p for p in candidates if winners[p] == best),
                key=lambda p: order.get(p, len(order)),
            )
            share, odd = divmod(layer.amount, len(tied))
            for index, player in enumerate(tied):
                payouts[player] += share + (index < odd)
        return payouts
