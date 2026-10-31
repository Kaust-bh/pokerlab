"""Bounded reproducible reservoir memories used by Deep CFR."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class AdvantageSample:
    player: int
    features: tuple[float, ...]
    targets: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    iteration: int
    sample_weight: float


@dataclass(frozen=True, slots=True)
class StrategySample:
    player: int
    features: tuple[float, ...]
    targets: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    iteration: int
    sample_weight: float


class ReservoirMemory[Sample: (AdvantageSample, StrategySample)]:
    """Uniform reservoir sample over the complete stream seen so far."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.seen = 0
        self.replacements = 0
        self.discarded = 0
        self.counter_baseline = 0
        self._items: list[Sample] = []

    def add(self, sample: Sample) -> None:
        self.seen += 1
        if len(self._items) < self.capacity:
            self._items.append(sample)
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self._items[index] = sample
            self.replacements += 1
        else:
            self.discarded += 1

    def extend(self, samples) -> None:
        for sample in samples:
            self.add(sample)

    def sample_batch(self, batch_size: int) -> list[Sample]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._items:
            raise ValueError("cannot sample an empty reservoir")
        return self.rng.sample(self._items, min(batch_size, len(self._items)))

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "seen": self.seen,
            "replacements": self.replacements,
            "discarded": self.discarded,
            "counter_baseline": self.counter_baseline,
            "rng_state": self.rng.getstate(),
            "items": [asdict(item) for item in self._items],
            "sample_type": self._items[0].__class__.__name__ if self._items else None,
        }

    def load_state_dict(self, state: dict, sample_type: type[Sample]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("reservoir capacity does not match checkpoint")
        if state.get("sample_type") not in (None, sample_type.__name__):
            raise ValueError("reservoir sample type does not match checkpoint")
        self.seen = int(state["seen"])
        self.replacements = int(state.get("replacements", 0))
        self.discarded = int(state.get("discarded", 0))
        self.counter_baseline = int(
            state.get(
                "counter_baseline",
                int(state["seen"])
                - len(state["items"])
                - self.replacements
                - self.discarded,
            )
        )
        if self.seen < len(state["items"]) or len(state["items"]) > self.capacity:
            raise ValueError("invalid reservoir checkpoint counts")
        if (
            self.replacements < 0
            or self.discarded < 0
            or self.counter_baseline < 0
            or self.replacements + self.discarded + self.counter_baseline
            != self.seen - len(state["items"])
        ):
            raise ValueError("invalid reservoir replacement counters")
        self._items = [
            sample_type(
                player=int(item["player"]),
                features=tuple(float(value) for value in item["features"]),
                targets=tuple(float(value) for value in item["targets"]),
                legal_mask=tuple(bool(value) for value in item["legal_mask"]),
                iteration=int(item["iteration"]),
                sample_weight=float(item["sample_weight"]),
            )
            for item in state["items"]
        ]
        self.rng.setstate(_tuple_tree(state["rng_state"]))

    @property
    def items(self) -> tuple[Sample, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)


class AdvantageMemory(ReservoirMemory[AdvantageSample]):
    pass


class StrategyMemory(ReservoirMemory[StrategySample]):
    pass


def _tuple_tree(value):
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value
