from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

try:
    import torch
except ImportError:  # optional deep-cfr extra
    torch = None

from pokerlab.algorithms.deep_cfr import (
    DeepCFRConfig,
    DeepCFRTrainer,
    KuhnCodec,
    LeducCodec,
)
from pokerlab.algorithms.deep_cfr.memory import (
    AdvantageMemory,
    AdvantageSample,
    StrategyMemory,
    StrategySample,
)
from pokerlab.algorithms.deep_cfr.metrics import mean_information_set_total_variation
from pokerlab.algorithms.deep_cfr.networks import AdvantageNetwork, StrategyNetwork
from pokerlab.algorithms.deep_cfr.traverser import ExternalSamplingTraverser
from pokerlab.research.kuhn.evaluation import evaluate_nash
from pokerlab.research.kuhn.game import KuhnGame
from pokerlab.research.kuhn.information_sets import InformationSet as KuhnInfo
from pokerlab.research.leduc.betting import Action as LeducAction
from pokerlab.research.leduc.cards import DECK
from pokerlab.research.leduc.game import LeducGame
from pokerlab.research.leduc.information_sets import InformationSet as LeducInfo


def _sample(index: int) -> AdvantageSample:
    return AdvantageSample(0, (float(index),), (1.0, -1.0), (True, True), 1, 1.0)


def test_reservoir_capacity_reproducibility_and_uniform_replacement():
    first = AdvantageMemory(capacity=4, seed=15)
    second = AdvantageMemory(capacity=4, seed=15)
    for index in range(100):
        first.add(_sample(index))
        second.add(_sample(index))
    assert len(first) == 4
    assert first.items == second.items
    assert first.seen == second.seen == 100
    assert len(first.sample_batch(3)) == 3
    assert len(first.sample_batch(99)) == 4
    counts = [0] * 10
    for seed in range(2_000):
        sample = AdvantageMemory(capacity=1, seed=seed)
        for index in range(10):
            sample.add(_sample(index))
        counts[int(sample.items[0].features[0])] += 1
    assert all(150 < count < 250 for count in counts)


def test_reservoir_empty_batch_and_checkpoint_state():
    memory = StrategyMemory(capacity=3, seed=21)
    with pytest.raises(ValueError, match="empty"):
        memory.sample_batch(1)
    nonempty = StrategyMemory(capacity=1, seed=3)
    nonempty.add(StrategySample(1, (0.0,), (0.5, 0.5), (True, True), 1, 1.0))
    with pytest.raises(ValueError, match="positive"):
        nonempty.sample_batch(0)
    for index in range(10):
        memory.add(
            StrategySample(1, (float(index),), (0.25, 0.75), (True, True), 2, 2.0)
        )
    state = memory.state_dict()
    restored = StrategyMemory(capacity=3, seed=0)
    restored.load_state_dict(state, StrategySample)
    assert restored.items == memory.items
    assert restored.seen == memory.seen
    assert restored.sample_batch(3) == memory.sample_batch(3)


def test_reservoir_replacement_and_discard_counts_survive_resume():
    memory = AdvantageMemory(capacity=2, seed=3)
    for index in range(20):
        memory.add(_sample(index))
    assert memory.seen == 20
    assert memory.replacements + memory.discarded == 18
    restored = AdvantageMemory(capacity=2, seed=99)
    restored.load_state_dict(memory.state_dict(), AdvantageSample)
    assert restored.replacements == memory.replacements
    assert restored.discarded == memory.discarded
    restored.add(_sample(20))
    assert restored.replacements + restored.discarded == 19


def test_strategy_distance_is_mean_per_infoset_total_variation():
    candidate = {
        "a": {"check": 0.75, "bet": 0.25},
        "b": {"fold": 0.0, "call": 1.0},
    }
    reference = {
        "a": {"check": 0.25, "bet": 0.75},
        "b": {"fold": 0.5, "call": 0.5},
    }
    assert mean_information_set_total_variation(candidate, reference) == pytest.approx(
        0.5
    )
    with pytest.raises(ValueError, match="same nonempty infosets"):
        mean_information_set_total_variation(candidate, {"a": reference["a"]})


def test_kuhn_and_leduc_encodings_are_deterministic_and_observation_complete():
    kuhn = KuhnCodec()
    deal = next(iter(KuhnGame.deals()))
    state = KuhnGame.initial_state(deal)
    info = KuhnInfo.from_state(state)
    assert len(kuhn.encode(info)) == kuhn.feature_dim == 17
    assert kuhn.encode(info) == kuhn.encode(info)
    alternate_deal = next(
        pair for pair in KuhnGame.deals() if pair[0] == deal[0] and pair[1] != deal[1]
    )
    assert kuhn.encode(
        KuhnInfo.from_state(KuhnGame.initial_state(alternate_deal))
    ) == kuhn.encode(info)

    leduc = LeducCodec()
    deal = LeducGame.deals()[0]
    leduc_state = LeducGame.initial_state(deal)
    leduc_info = LeducInfo.from_state(leduc_state)
    encoded = leduc.encode(leduc_info)
    assert len(encoded) == leduc.feature_dim == 57
    assert encoded == leduc.encode(leduc_info)
    alternative_hidden_deal = next(
        pair for pair in LeducGame.deals() if pair[0] == deal[0] and pair[1] != deal[1]
    )
    assert (
        leduc.encode(
            LeducInfo.from_state(LeducGame.initial_state(alternative_hidden_deal))
        )
        == encoded
    )

    card = next(card for card in DECK if card not in deal)
    board_state = leduc_state.act(LeducAction.CHECK).act(LeducAction.CHECK)
    board_state = board_state.deal_public(card)
    board_info = LeducInfo.from_state(board_state)
    assert sum(leduc.encode(board_info)[2 : 2 + len(DECK) + 1]) == 1


@pytest.mark.skipif(torch is None, reason="requires optional PyTorch dependency")
def test_network_dimensions_masking_updates_and_probability_normalization():
    advantage = AdvantageNetwork(3, 4, (8,), "tanh")
    features = torch.tensor([[0.2, 0.4, -0.1]], dtype=torch.float32)
    mask = torch.tensor([[True, False, True, False]])
    assert advantage(features).shape == (1, 4)
    strategy = StrategyNetwork(3, 4, (8, 8))
    probabilities = strategy.masked_probabilities(features, mask)
    assert probabilities[0, 1].item() == 0
    assert probabilities[0, 3].item() == 0
    assert probabilities.sum().item() == pytest.approx(1.0)
    optimizer = torch.optim.Adam(strategy.parameters(), lr=0.01)
    before = {
        key: value.detach().clone() for key, value in strategy.state_dict().items()
    }
    loss = (strategy.masked_probabilities(features, mask)[0, 0] - 0.9).square()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(before[key], value)
        for key, value in strategy.state_dict().items()
    )


@dataclass(frozen=True)
class _ToyState:
    player: int | None = None
    info: str = "root"
    children: tuple[tuple[str, _ToyState], ...] = ()
    outcomes: tuple[tuple[str, float], ...] = ()
    payoff: float | None = None
    chance: bool = False


class _ToyCodec:
    game_name = "toy"
    action_vocabulary = ("left", "right")
    feature_dim = 1

    def initial_state(self, rng):
        return self.root

    def information_set(self, state):
        return state.info

    def encode(self, info):
        return (float(getattr(self, "feature_map", {}).get(info, 0.0)),)

    def is_terminal(self, state):
        return state.payoff is not None

    def is_chance_node(self, state):
        return state.chance

    def chance_outcomes(self, state):
        return state.outcomes

    def deal(self, state, outcome):
        return dict(state.children)[outcome]

    def current_player(self, state):
        assert state.player is not None
        return state.player

    def legal_actions(self, state):
        return tuple(action for action, _child in state.children)

    def act(self, state, action):
        return dict(state.children)[action]

    def utility(self, state, player):
        assert state.payoff is not None
        return state.payoff if player == 0 else -state.payoff

    def action_index(self, action):
        return self.action_vocabulary.index(action)


def test_external_sampling_chance_estimator_matches_independent_expectation():
    first = _ToyState(
        0, "shared", (("left", _ToyState(payoff=4)), ("right", _ToyState(payoff=0)))
    )
    second = _ToyState(
        0, "shared", (("left", _ToyState(payoff=0)), ("right", _ToyState(payoff=2)))
    )
    codec = _ToyCodec()
    codec.root = _ToyState(
        children=(("A", first), ("B", second)),
        outcomes=(("A", 0.25), ("B", 0.75)),
        chance=True,
    )
    policy = lambda _player, _features, _mask: (0.5, 0.5)
    traverser = ExternalSamplingTraverser(codec, policy, random.Random(101))
    advantages = AdvantageMemory(20_000, seed=1)
    strategies = StrategyMemory(20_000, seed=2)
    for _ in range(20_000):
        traverser.traverse(
            codec.root,
            0,
            iteration=1,
            sample_weight=1.0,
            advantage_memory=advantages,
            strategy_memory=strategies,
        )
    left_mean = sum(sample.targets[0] for sample in advantages.items) / len(advantages)
    right_mean = sum(sample.targets[1] for sample in advantages.items) / len(advantages)
    # A: regrets (2,-2); B: regrets (-1,1). Chance expectation = (-.25,.25).
    assert left_mean == pytest.approx(-0.25, abs=0.025)
    assert right_mean == pytest.approx(0.25, abs=0.025)
    assert all(sample.legal_mask == (True, True) for sample in advantages.items)


def test_player_one_advantage_uses_its_own_zero_sum_utility():
    codec = _ToyCodec()
    codec.root = _ToyState(
        1,
        "player-one",
        (("left", _ToyState(payoff=3)), ("right", _ToyState(payoff=1))),
    )
    traverser = ExternalSamplingTraverser(
        codec,
        lambda _player, _features, _mask: (0.5, 0.5),
        random.Random(51),
    )
    memory = AdvantageMemory(1, seed=6)
    traverser.traverse(
        codec.root,
        1,
        iteration=1,
        sample_weight=1.0,
        advantage_memory=memory,
        strategy_memory=StrategyMemory(1, seed=7),
    )
    assert memory.items[0].player == 1
    assert memory.items[0].targets == (-1.0, 1.0)


def test_traverser_advantages_exclude_own_reach_at_later_infosets():
    left_later = _ToyState(
        0,
        "later-left",
        (("left", _ToyState(payoff=3)), ("right", _ToyState(payoff=1))),
    )
    right_later = _ToyState(
        0,
        "later-right",
        (("left", _ToyState(payoff=2)), ("right", _ToyState(payoff=0))),
    )
    codec = _ToyCodec()
    codec.root = _ToyState(
        0,
        "traverser-root",
        (("left", left_later), ("right", right_later)),
    )
    traverser = ExternalSamplingTraverser(
        codec,
        lambda _player, _features, _mask: (0.5, 0.5),
        random.Random(41),
    )
    memory = AdvantageMemory(10, seed=4)
    traverser.traverse(
        codec.root,
        0,
        iteration=1,
        sample_weight=1.0,
        advantage_memory=memory,
        strategy_memory=StrategyMemory(10, seed=5),
    )
    later_targets = [
        sample.targets for sample in memory.items if sample.features == (0.0,)
    ]
    # Each later information set has action values differing by two. The
    # probability of the preceding root action must not scale its advantage.
    assert later_targets[:2] == [(1.0, -1.0), (1.0, -1.0)]


def test_external_sampling_opponent_policy_estimator_and_strategy_samples():
    left_opponent = _ToyState(
        1,
        "opponent-after-left",
        (("left", _ToyState(payoff=4)), ("right", _ToyState(payoff=0))),
    )
    right_opponent = _ToyState(
        1,
        "opponent-after-right",
        (("left", _ToyState(payoff=0)), ("right", _ToyState(payoff=2))),
    )
    codec = _ToyCodec()
    codec.root = _ToyState(
        0,
        "traverser-root",
        (("left", left_opponent), ("right", right_opponent)),
    )

    def policy(player, _features, _mask):
        return (0.5, 0.5) if player == 0 else (0.25, 0.75)

    traverser = ExternalSamplingTraverser(codec, policy, random.Random(202))
    advantages = AdvantageMemory(10_000, seed=4)
    strategies = StrategyMemory(10_000, seed=5)
    for _ in range(10_000):
        traverser.traverse(
            codec.root,
            0,
            iteration=3,
            sample_weight=3.0,
            advantage_memory=advantages,
            strategy_memory=strategies,
        )
    means = tuple(
        sum(sample.targets[index] for sample in advantages.items) / len(advantages)
        for index in (0, 1)
    )
    # Expected action values are 1 and 1.5; the uniform root value is 1.25.
    assert means == pytest.approx((-0.25, 0.25), abs=0.04)
    assert len(strategies) == 10_000
    assert strategies.seen == 20_000
    assert all(sample.targets == (0.25, 0.75) for sample in strategies.items)
    # Both root actions are enumerated at probability 1/2, so each opponent
    # strategy row receives iteration weight 3 times traverser reach 1/2.
    assert all(sample.sample_weight == 1.5 for sample in strategies.items)


def test_strategy_memory_weights_enumerated_branches_by_traverser_reach():
    after_left = _ToyState(1, "after-left", (("left", _ToyState(payoff=1)),))
    after_right = _ToyState(1, "after-right", (("left", _ToyState(payoff=1)),))
    codec = _ToyCodec()
    codec.feature_map = {"after-left": 1.0, "after-right": 2.0}
    codec.root = _ToyState(
        0, "traverser-root", (("left", after_left), ("right", after_right))
    )

    def policy(player, _features, mask):
        return (0.25, 0.75) if player == 0 else (1.0, 0.0)

    traverser = ExternalSamplingTraverser(codec, policy, random.Random(303))
    strategies = StrategyMemory(10, seed=10)
    traverser.traverse(
        codec.root,
        0,
        iteration=1,
        sample_weight=1.0,
        advantage_memory=AdvantageMemory(10, seed=9),
        strategy_memory=strategies,
    )
    weights = {sample.features: sample.sample_weight for sample in strategies.items}
    # Traverser actions are both enumerated, so visitation alone gives each
    # branch equal multiplicity. Multiplying by the traverser's own reach
    # restores the 1/4 : 3/4 contribution required by the average strategy.
    assert weights == {(1.0,): 0.25, (2.0,): 0.75}


@pytest.mark.skipif(torch is None, reason="requires optional PyTorch dependency")
def test_deep_cfr_training_exact_kuhn_evaluation_and_reproducibility():
    config = DeepCFRConfig(
        iterations=4,
        seed=51,
        hidden_layers=(16,),
        batch_size=16,
        advantage_memory_capacity=1_000,
        strategy_memory_capacity=1_000,
        iteration_weighting="uniform",
    )
    first = DeepCFRTrainer(KuhnCodec(), config)
    second = DeepCFRTrainer(KuhnCodec(), config)
    assert first.advantage_networks[0] is not first.advantage_networks[1]
    assert first.advantage_memories[0] is not first.advantage_memories[1]
    first.train()
    second.train()
    profile = first.policy(seed=7).exact_strategy_profile()
    assert profile == second.policy(seed=7).exact_strategy_profile()
    evaluation = evaluate_nash(profile)
    assert evaluation.exploitability >= 0
    assert len(profile) == 12
    assert all(
        sum(distribution.values()) == pytest.approx(1.0)
        for distribution in profile.values()
    )
    assert all(
        record.traversal_count == 2 * (index + 1)
        for index, record in enumerate(first.records)
    )


@pytest.mark.skipif(torch is None, reason="requires optional PyTorch dependency")
def test_training_updates_strategy_network_and_can_play_a_complete_kuhn_game():
    trainer = DeepCFRTrainer(
        KuhnCodec(),
        DeepCFRConfig(
            iterations=2,
            seed=61,
            hidden_layers=(8,),
            batch_size=16,
            advantage_memory_capacity=500,
            strategy_memory_capacity=500,
        ),
    )
    before = {
        key: value.detach().clone()
        for key, value in trainer.strategy_network.state_dict().items()
    }
    records = trainer.train()
    assert len(records) == 2
    assert records[-1].strategy_loss >= 0
    assert any(
        not torch.equal(before[key], value)
        for key, value in trainer.strategy_network.state_dict().items()
    )
    policy = trainer.policy(seed=123)
    state = KuhnGame.initial_state(next(iter(KuhnGame.deals())))
    while not state.is_terminal:
        action = policy.action_for_state(state)
        assert action in state.legal_actions
        state = state.act(action)
    assert sum(state.utilities()) == 0


@pytest.mark.skipif(torch is None, reason="requires optional PyTorch dependency")
def test_strategy_network_has_independent_architecture_and_training_budget():
    trainer = DeepCFRTrainer(
        KuhnCodec(),
        DeepCFRConfig(
            iterations=1,
            seed=62,
            hidden_layers=(16,),
            strategy_hidden_layers=(4, 4),
            strategy_training_epochs=2,
            strategy_learning_rate=5e-4,
            batch_size=16,
            advantage_memory_capacity=500,
            strategy_memory_capacity=500,
        ),
    )
    assert trainer.advantage_networks[0].hidden[0].out_features == 16
    assert trainer.strategy_network.hidden[0].out_features == 4
    assert trainer.config.strategy_training_epochs == 2
    assert trainer.train()[0].strategy_loss >= 0


@pytest.mark.skipif(torch is None, reason="requires optional PyTorch dependency")
def test_leduc_deep_cfr_training_and_legal_policy_inference():
    trainer = DeepCFRTrainer(
        LeducCodec(),
        DeepCFRConfig(
            iterations=2,
            seed=63,
            hidden_layers=(16,),
            batch_size=32,
            advantage_memory_capacity=1_000,
            strategy_memory_capacity=1_000,
        ),
    )
    trainer.train()
    policy = trainer.policy(seed=64)
    profile = policy.exact_strategy_profile()
    assert profile
    state = LeducGame.initial_state(LeducGame.deals()[0])
    while not state.is_terminal:
        if state.is_chance_node:
            state = state.deal_public(state.chance_outcomes[0][0])
            continue
        action = policy.action_for_state(state, deterministic=True)
        assert action in state.legal_actions
        state = state.act(action)
    assert sum(state.utilities()) == pytest.approx(0.0)


@pytest.mark.skipif(torch is None, reason="requires optional PyTorch dependency")
def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    config = DeepCFRConfig(
        iterations=4,
        seed=72,
        hidden_layers=(8,),
        batch_size=16,
        advantage_memory_capacity=1_000,
        strategy_memory_capacity=1_000,
        iteration_weighting="uniform",
    )
    uninterrupted = DeepCFRTrainer(KuhnCodec(), config)
    uninterrupted.train(4)
    partial = DeepCFRTrainer(KuhnCodec(), config)
    partial.train(2)
    checkpoint = partial.save_checkpoint(tmp_path / "deep-cfr.pt")
    resumed = DeepCFRTrainer(KuhnCodec(), config)
    resumed.load_checkpoint(checkpoint)
    resumed.train(2)
    assert resumed.iteration == uninterrupted.iteration == 4
    for resumed_record, full_record in zip(
        resumed.records, uninterrupted.records, strict=True
    ):
        assert resumed_record.iteration == full_record.iteration
        assert resumed_record.player0_advantage_loss == pytest.approx(
            full_record.player0_advantage_loss
        )
        assert resumed_record.player1_advantage_loss == pytest.approx(
            full_record.player1_advantage_loss
        )
        assert resumed_record.strategy_loss == pytest.approx(full_record.strategy_loss)
        assert (
            resumed_record.advantage_memory_sizes == full_record.advantage_memory_sizes
        )
        assert resumed_record.strategy_memory_size == full_record.strategy_memory_size
    assert (
        resumed.policy(seed=5).exact_strategy_profile()
        == uninterrupted.policy(seed=5).exact_strategy_profile()
    )
    for first, second in zip(
        resumed.advantage_networks, uninterrupted.advantage_networks, strict=True
    ):
        for key in first.state_dict():
            assert torch.equal(first.state_dict()[key], second.state_dict()[key])
