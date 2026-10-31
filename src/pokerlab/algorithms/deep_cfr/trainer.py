"""Deep CFR training loop with external-sampling traversals and reservoirs."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from pokerlab.algorithms.deep_cfr.config import DeepCFRConfig
from pokerlab.algorithms.deep_cfr.encoding import GameCodec
from pokerlab.algorithms.deep_cfr.memory import (
    AdvantageMemory,
    AdvantageSample,
    StrategyMemory,
    StrategySample,
    _tuple_tree,
)
from pokerlab.algorithms.deep_cfr.networks import (
    AdvantageNetwork,
    StrategyNetwork,
    require_torch,
)
from pokerlab.algorithms.deep_cfr.traverser import (
    ExternalSamplingTraverser,
    TraversalStats,
)


@dataclass(frozen=True, slots=True)
class DeepCFRTrainingRecord:
    iteration: int
    player0_advantage_loss: float
    player1_advantage_loss: float
    strategy_loss: float
    advantage_memory_sizes: tuple[int, int]
    strategy_memory_size: int
    traversal_count: int
    sampled_nodes: int
    runtime_seconds: float
    iterations_per_second: float
    advantage_memory_seen: tuple[int, int] = (0, 0)
    strategy_memory_seen: int = 0
    advantage_memory_replacements: tuple[int, int] = (0, 0)
    strategy_memory_replacements: int = 0


class DeepCFRTrainer:
    """Two-player Deep CFR using the paper's external-sampling data flow.

    Advantage and strategy models are refit from scratch each iteration, as in
    Brown et al. (2019). Advantage memories are player-specific; the strategy
    memory is shared and its samples include player identity in the encoding.
    """

    def __init__(self, codec: GameCodec, config: DeepCFRConfig | None = None) -> None:
        torch = require_torch()
        self.codec = codec
        self.config = config or DeepCFRConfig()
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        mps_backend = getattr(torch.backends, "mps", None)
        if self.config.device.startswith("mps") and not (
            mps_backend is not None and mps_backend.is_available()
        ):
            raise ValueError("MPS was requested but is unavailable")
        self.device = torch.device(self.config.device)
        self.rng = random.Random(self.config.seed)
        self.iteration = 0
        self.traversal_count = 0
        self.records: list[DeepCFRTrainingRecord] = []
        self.advantage_memories = [
            AdvantageMemory(self.config.advantage_memory_capacity, self.config.seed + p)
            for p in (0, 1)
        ]
        self.strategy_memory = StrategyMemory(
            self.config.strategy_memory_capacity, self.config.seed + 17
        )
        torch.manual_seed(self.config.seed)
        self.advantage_networks = [self._new_network(AdvantageNetwork) for _ in (0, 1)]
        self.strategy_network = self._new_network(
            StrategyNetwork,
            self.config.strategy_hidden_layers or self.config.hidden_layers,
        )

    def _new_network(self, network_type, hidden_layers=None):
        return network_type(
            self.codec.feature_dim,
            len(self.codec.action_vocabulary),
            self.config.hidden_layers if hidden_layers is None else hidden_layers,
            self.config.activation,
        ).to(self.device)

    def _policy(
        self, player: int, features: tuple[float, ...], mask: tuple[bool, ...]
    ) -> tuple[float, ...]:
        torch = require_torch()
        network = self.advantage_networks[player]
        network.eval()
        with torch.inference_mode():
            inputs = torch.tensor(
                features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            values = network(inputs)[0].detach().cpu().tolist()
        if len(values) != len(mask) or not any(mask):
            raise ValueError("advantage output and legal-action mask are incompatible")
        legal_values = [
            max(0.0, value) if legal else 0.0
            for value, legal in zip(values, mask, strict=True)
        ]
        total = sum(legal_values)
        if total > 0 and math.isfinite(total):
            return tuple(value / total for value in legal_values)
        legal_count = sum(mask)
        return tuple((1.0 / legal_count) if legal else 0.0 for legal in mask)

    def _fit(
        self, network, memory, *, player: int, iteration: int, strategy: bool
    ) -> float:
        torch = require_torch()
        if not memory:
            raise RuntimeError("cannot fit a Deep CFR network without samples")
        learning_rate = (
            self.config.strategy_learning_rate
            if strategy and self.config.strategy_learning_rate is not None
            else self.config.learning_rate
        )
        batch_size = (
            self.config.strategy_batch_size
            if strategy and self.config.strategy_batch_size is not None
            else self.config.batch_size
        )
        training_epochs = (
            self.config.strategy_training_epochs
            if strategy and self.config.strategy_training_epochs is not None
            else self.config.training_epochs
        )
        optimizer_name = (
            self.config.strategy_optimizer
            if strategy and self.config.strategy_optimizer is not None
            else self.config.optimizer
        )
        seed_offset = 100_003 if strategy else 10_007 * (player + 1)
        torch.manual_seed(self.config.seed + iteration * 1_000_003 + seed_offset)
        network.reset_parameters()
        network.train()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

        samples = memory.items
        features = torch.tensor(
            [s.features for s in samples], dtype=torch.float32, device=self.device
        )
        targets = torch.tensor(
            [s.targets for s in samples], dtype=torch.float32, device=self.device
        )
        masks = torch.tensor(
            [s.legal_mask for s in samples], dtype=torch.bool, device=self.device
        )
        weights = torch.tensor(
            [s.sample_weight for s in samples], dtype=torch.float32, device=self.device
        )
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("Deep CFR sample weights must be finite and positive")

        order_rng = random.Random(
            self.config.seed + iteration * 2_000_003 + seed_offset
        )
        losses: list[tuple[float, int]] = []
        for _epoch in range(training_epochs):
            order = list(range(len(samples)))
            order_rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                indices = torch.tensor(
                    order[start : start + batch_size], device=self.device
                )
                batch_features = features[indices]
                batch_targets = targets[indices]
                batch_mask = masks[indices]
                batch_weights = weights[indices]
                if strategy:
                    predictions = network.masked_probabilities(
                        batch_features, batch_mask
                    )
                else:
                    predictions = network(batch_features)
                per_action = (predictions - batch_targets).square() * batch_mask
                per_sample = per_action.sum(dim=-1) / batch_mask.sum(dim=-1).clamp_min(
                    1
                )
                loss = (per_sample * batch_weights).sum() / batch_weights.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.config.gradient_clip_norm
                )
                optimizer.step()
                losses.append((float(loss.detach().cpu()), len(indices)))
        network.eval()
        total_rows = sum(count for _, count in losses)
        return sum(loss * count for loss, count in losses) / total_rows

    def run_iteration(self) -> DeepCFRTrainingRecord:
        iteration = self.iteration + 1
        weight = self.config.sample_weight(iteration)
        started = time.perf_counter()
        stats_total = TraversalStats()
        advantage_losses = [0.0, 0.0]
        for player in (0, 1):
            traverser = ExternalSamplingTraverser(self.codec, self._policy, self.rng)
            for _ in range(self.config.traversals_per_player):
                initial = self.codec.initial_state(self.rng)
                # The initial private-card deal is a sampled chance event for
                # both Kuhn and Leduc, even where the game API omits a root node.
                stats_total.chance_samples += 1
                _value, stats = traverser.traverse(
                    initial,
                    player,
                    iteration=iteration,
                    sample_weight=weight,
                    advantage_memory=self.advantage_memories[player],
                    strategy_memory=self.strategy_memory,
                )
                _sum_stats(stats_total, stats)
                self.traversal_count += 1
            advantage_losses[player] = self._fit(
                self.advantage_networks[player],
                self.advantage_memories[player],
                player=player,
                iteration=iteration,
                strategy=False,
            )

        strategy_loss = self._fit(
            self.strategy_network,
            self.strategy_memory,
            player=0,
            iteration=iteration,
            strategy=True,
        )
        self.iteration = iteration
        elapsed = time.perf_counter() - started
        record = DeepCFRTrainingRecord(
            iteration=iteration,
            player0_advantage_loss=advantage_losses[0],
            player1_advantage_loss=advantage_losses[1],
            strategy_loss=strategy_loss,
            advantage_memory_sizes=tuple(
                len(memory) for memory in self.advantage_memories
            ),
            strategy_memory_size=len(self.strategy_memory),
            traversal_count=self.traversal_count,
            sampled_nodes=stats_total.nodes,
            runtime_seconds=elapsed,
            iterations_per_second=1.0 / elapsed if elapsed else 0.0,
            advantage_memory_seen=tuple(
                memory.seen for memory in self.advantage_memories
            ),
            strategy_memory_seen=self.strategy_memory.seen,
            advantage_memory_replacements=tuple(
                memory.replacements for memory in self.advantage_memories
            ),
            strategy_memory_replacements=self.strategy_memory.replacements,
        )
        self.records.append(record)
        return record

    def train(
        self,
        iterations: int | None = None,
        *,
        callback: Callable[[DeepCFRTrainingRecord], None] | None = None,
    ) -> tuple[DeepCFRTrainingRecord, ...]:
        count = self.config.iterations if iterations is None else iterations
        if count < 1:
            raise ValueError("iterations must be positive")
        before = len(self.records)
        for _ in range(count):
            record = self.run_iteration()
            if callback is not None and (
                record.iteration % self.config.evaluation_interval == 0
                or record.iteration == self.config.iterations
            ):
                callback(record)
        return tuple(self.records[before:])

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "game": self.codec.game_name,
            "config": asdict(self.config),
            "iteration": self.iteration,
            "traversal_count": self.traversal_count,
            "rng_state": self.rng.getstate(),
            "advantage_memories": [
                memory.state_dict() for memory in self.advantage_memories
            ],
            "strategy_memory": self.strategy_memory.state_dict(),
            "advantage_networks": [
                {
                    key: value.detach().cpu()
                    for key, value in network.state_dict().items()
                }
                for network in self.advantage_networks
            ],
            "strategy_network": {
                key: value.detach().cpu()
                for key, value in self.strategy_network.state_dict().items()
            },
            "records": [asdict(record) for record in self.records],
        }

    def load_state_dict(self, state: dict) -> None:
        torch = require_torch()
        if (
            state.get("schema_version") != 1
            or state.get("game") != self.codec.game_name
        ):
            raise ValueError("checkpoint schema or game does not match")
        saved = dict(state["config"])
        current = asdict(self.config)
        # Iterations and evaluation cadence are run controls; all mathematical,
        # sampling, memory, and network settings must match for continuation.
        for field in ("iterations", "evaluation_interval"):
            saved.pop(field, None)
            current.pop(field, None)
        if saved != current:
            raise ValueError("checkpoint Deep CFR configuration does not match")
        self.iteration = int(state["iteration"])
        self.traversal_count = int(state["traversal_count"])
        self.rng.setstate(_tuple_tree(state["rng_state"]))
        for memory, memory_state in zip(
            self.advantage_memories, state["advantage_memories"], strict=True
        ):
            memory.load_state_dict(memory_state, AdvantageSample)
        self.strategy_memory.load_state_dict(state["strategy_memory"], StrategySample)
        for network, network_state in zip(
            self.advantage_networks, state["advantage_networks"], strict=True
        ):
            network.load_state_dict(network_state)
        self.strategy_network.load_state_dict(state["strategy_network"])
        self.records = [DeepCFRTrainingRecord(**record) for record in state["records"]]
        torch.manual_seed(self.config.seed + self.iteration * 1_000_003)

    def save_checkpoint(self, path: str | Path, *, overwrite: bool = False) -> Path:
        torch = require_torch()
        target = Path(path)
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        torch = require_torch()
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.load_state_dict(state)

    def policy(self, *, seed: int | None = None):
        from pokerlab.algorithms.deep_cfr.inference import DeepCFRPolicy

        return DeepCFRPolicy(
            self.codec,
            self.strategy_network,
            device=self.config.device,
            seed=self.config.seed if seed is None else seed,
        )


def _sum_stats(target: TraversalStats, source: TraversalStats) -> None:
    for name in (
        "nodes",
        "chance_samples",
        "opponent_samples",
        "traverser_decisions",
        "advantage_samples",
        "strategy_samples",
    ):
        setattr(target, name, getattr(target, name) + getattr(source, name))
