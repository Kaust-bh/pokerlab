"""Validated Deep CFR network and traversal configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeepCFRConfig:
    iterations: int = 100
    traversals_per_player: int = 1
    advantage_memory_capacity: int = 50_000
    strategy_memory_capacity: int = 50_000
    hidden_layers: tuple[int, ...] = (64, 64)
    strategy_hidden_layers: tuple[int, ...] | None = None
    activation: str = "relu"
    learning_rate: float = 1e-3
    strategy_learning_rate: float | None = None
    batch_size: int = 128
    strategy_batch_size: int | None = None
    training_epochs: int = 1
    strategy_training_epochs: int | None = None
    optimizer: str = "adam"
    strategy_optimizer: str | None = None
    gradient_clip_norm: float = 1.0
    device: str = "cpu"
    iteration_weighting: str = "linear"
    evaluation_interval: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.traversals_per_player < 1:
            raise ValueError("iterations and traversals_per_player must be positive")
        if self.advantage_memory_capacity < 1 or self.strategy_memory_capacity < 1:
            raise ValueError("memory capacities must be positive")
        if not self.hidden_layers or any(width < 1 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        if self.strategy_hidden_layers is not None and (
            not self.strategy_hidden_layers
            or any(width < 1 for width in self.strategy_hidden_layers)
        ):
            raise ValueError("strategy_hidden_layers must contain positive widths")
        if self.activation not in {"relu", "tanh", "leaky_relu"}:
            raise ValueError("unsupported activation")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.strategy_learning_rate is not None and (
            not math.isfinite(self.strategy_learning_rate)
            or self.strategy_learning_rate <= 0
        ):
            raise ValueError("strategy_learning_rate must be positive")
        if self.batch_size < 1 or self.training_epochs < 1:
            raise ValueError("batch_size and training_epochs must be positive")
        if self.strategy_batch_size is not None and self.strategy_batch_size < 1:
            raise ValueError("strategy_batch_size must be positive")
        if (
            self.strategy_training_epochs is not None
            and self.strategy_training_epochs < 1
        ):
            raise ValueError("strategy_training_epochs must be positive")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if self.strategy_optimizer is not None and self.strategy_optimizer not in {
            "adam",
            "sgd",
        }:
            raise ValueError("strategy_optimizer must be 'adam' or 'sgd'")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.iteration_weighting not in {"linear", "uniform"}:
            raise ValueError("iteration_weighting must be 'linear' or 'uniform'")
        if self.evaluation_interval < 1:
            raise ValueError("evaluation_interval must be positive")

    def sample_weight(self, iteration: int) -> float:
        return float(iteration if self.iteration_weighting == "linear" else 1)
