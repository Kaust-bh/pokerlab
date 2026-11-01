import json
import random
from dataclasses import asdict, replace
from itertools import combinations

import pytest

from pokerlab.agents.action_profiles import ACTION_PROFILES
from pokerlab.agents.base import observe
from pokerlab.engine import (
    ActionType,
    Card,
    Deck,
    GameState,
    HandCategory,
    evaluate_hand,
)
from pokerlab.research.nlhe.abstraction import (
    _draw_features,
    percentile_bucket,
    preflop_class,
    private_hand_bucket,
)
from pokerlab.research.nlhe.actions import (
    AbstractAction,
    abstract_action_specs,
    abstract_action_specs_from_observation,
)
from pokerlab.research.nlhe.agent import deserialize_strategy, serialize_strategy
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.game import AbstractHUNLState
from pokerlab.research.nlhe.information_set import information_set_from_observation
from pokerlab.research.nlhe.mccfr import (
    ExternalSamplingMCCFR,
    HUNLMCCFRConfig,
    average_strategy_importance_weight,
)


def test_preflop_abstraction_has_exactly_169_classes_and_canonical_suits():
    deck = tuple(
        Card(rank, suit)
        for suit in ("clubs", "diamonds", "hearts", "spades")
        for rank in range(2, 15)
    )
    classes = {preflop_class(hand) for hand in combinations(deck, 2)}
    assert len(classes) == 169
    assert preflop_class((Card(14, "clubs"), Card(13, "clubs"))) == "AKs"
    assert preflop_class((Card(14, "clubs"), Card(13, "spades"))) == "AKo"
    assert preflop_class((Card(13, "hearts"), Card(14, "hearts"))) == "AKs"


def test_postflop_bucket_is_deterministic_and_does_not_accept_opponent_cards():
    config = HUNLGameConfig(postflop_buckets=8, opponent_hand_samples=16)
    hand = (Card(14, "spades"), Card(13, "spades"))
    board = (Card(2, "spades"), Card(8, "spades"), Card(10, "clubs"))
    assert private_hand_bucket(hand, board, config) == private_hand_bucket(
        hand, board, config
    )


def test_bucket_boundaries_and_optimized_draw_features():
    assert [percentile_bucket(p, 4) for p in (0, 0.249, 0.25, 0.999, 1)] == [
        0,
        0,
        1,
        3,
        3,
    ]
    hand = (Card(9, "spades"), Card(8, "clubs"))
    board = (Card(7, "hearts"), Card(6, "diamonds"), Card(2, "clubs"))
    known = set(hand) | set(board)
    available = tuple(
        Card(rank, suit)
        for suit in ("clubs", "diamonds", "hearts", "spades")
        for rank in range(2, 15)
        if Card(rank, suit) not in known
    )
    _flush, _code, bucket = _draw_features(hand, board, available, 0)
    assert bucket == 2  # Eight physical outs complete the open-ended straight.


def test_sampled_equity_is_deterministic_suit_invariant_and_range_configurable():
    common = {
        "postflop_method": "sampled_equity",
        "postflop_buckets": 10,
        "equity_samples": 32,
    }
    uniform = HUNLGameConfig(**common, opponent_range="uniform")
    premium = HUNLGameConfig(**common, opponent_range="premium")
    hand = (Card(14, "spades"), Card(13, "spades"))
    board = (Card(2, "spades"), Card(8, "spades"), Card(10, "clubs"))
    permuted_hand = (Card(14, "hearts"), Card(13, "hearts"))
    permuted_board = (
        Card(2, "hearts"),
        Card(8, "hearts"),
        Card(10, "diamonds"),
    )
    bucket = private_hand_bucket(hand, board, uniform)
    assert bucket == private_hand_bucket(hand, board, uniform)
    assert bucket == private_hand_bucket(permuted_hand, permuted_board, uniform)
    assert "equity:10:premium:" in private_hand_bucket(hand, board, premium)


def test_action_profiles_are_configurable_and_observation_legal():
    config = HUNLGameConfig(starting_stack_bb=20, action_profile="wide")
    state = AbstractHUNLState.new_hand(config, random.Random(71))
    button = state.engine.current_player
    state = state.act(AbstractAction.CALL)
    state = state.act(AbstractAction.CHECK)
    state = state.act(AbstractAction.BET_POT)
    state = state.act(AbstractAction.CALL)
    observation = observe(
        state.engine,
        state.engine.current_player,
        action_profile="wide",
    )
    observed = {spec.abstract: spec.engine for spec in state.legal_specs}
    assert AbstractAction.BET_MIN in observed
    assert AbstractAction.BET_THIRD_POT in observed
    assert AbstractAction.BET_TWO_POT in observed
    assert all(spec.engine in observation.legal_actions for spec in state.legal_specs)
    assert button in state.engine.players


def test_all_action_profiles_match_agent_legal_menus_and_engine_transitions():
    rng = random.Random(819)
    for profile in ACTION_PROFILES:
        config = HUNLGameConfig(
            starting_stack_bb=5,
            postflop_buckets=4,
            opponent_hand_samples=4,
            action_profile=profile,
        )
        for hand_index in range(8):
            state = AbstractHUNLState.new_hand(config, rng, button_index=hand_index % 2)
            while not state.terminal:
                specs = state.legal_specs
                observation = observe(
                    state.engine,
                    state.engine.current_player,
                    action_profile=profile,
                )
                visible_specs = abstract_action_specs_from_observation(observation)
                assert [(item.abstract, item.engine) for item in specs] == [
                    (item.abstract, item.engine) for item in visible_specs
                ]
                assert all(item.engine in observation.legal_actions for item in specs)
                state = state.act(rng.choice(specs).abstract)


def test_abstraction_configuration_round_trips_through_json():
    config = HUNLGameConfig(
        starting_stack_bb=25,
        postflop_method="sampled_equity",
        postflop_buckets=12,
        opponent_hand_samples=64,
        equity_samples=48,
        opponent_range="premium",
        action_profile="wide",
    )
    payload = json.loads(json.dumps(asdict(config)))
    assert HUNLGameConfig(**payload) == config


def test_optimized_legacy_straight_draw_bucket_matches_evaluator_reference():
    rng = random.Random(423)
    deck = tuple(
        Card(rank, suit)
        for suit in ("clubs", "diamonds", "hearts", "spades")
        for rank in range(2, 15)
    )
    for board_size in (3, 4):
        for _ in range(8):
            chosen = rng.sample(deck, 2 + board_size)
            hole, board = tuple(chosen[:2]), tuple(chosen[2:])
            known = set(chosen)
            available = tuple(card for card in deck if card not in known)
            current = evaluate_hand((*hole, *board))
            outs = 0
            if current.category < HandCategory.STRAIGHT:
                outs = sum(
                    evaluate_hand((*hole, *board, card)).category
                    >= HandCategory.STRAIGHT
                    for card in available
                )
            expected = 0 if outs == 0 else 1 if outs <= 4 else 2 if outs <= 8 else 3
            assert (
                _draw_features(
                    hole,
                    board,
                    available,
                    int(current.category),
                    legacy_combined_flush_outs=True,
                )[2]
                == expected
            )


def test_abstract_raise_targets_and_short_all_in_respect_engine_rules():
    deep = GameState({"p0": 200, "p1": 200}, deck=Deck(seed=5))
    specs = abstract_action_specs(deep, deep.current_player)
    targets = {spec.abstract: spec.engine for spec in specs}
    assert targets[AbstractAction.RAISE_HALF_POT].amount == 4
    assert targets[AbstractAction.RAISE_POT].amount == 6

    short = GameState({"p0": 3, "p1": 20}, deck=Deck(seed=6), button="p0")
    short_specs = abstract_action_specs(short, "p0")
    all_in = next(
        spec for spec in short_specs if spec.abstract == AbstractAction.ALL_IN
    )
    assert all_in.engine.type == ActionType.ALL_IN
    short.act(all_in.engine, player="p0")
    assert short.betting.contributions["p0"] == 3
    assert short.betting.current_bet == 3


def test_engine_backed_state_completes_hands_and_preserves_chips():
    config = HUNLGameConfig(starting_stack_bb=5, opponent_hand_samples=8)
    rng = random.Random(12)
    for button in (0, 1):
        state = AbstractHUNLState.new_hand(config, rng, button_index=button)
        while not state.terminal:
            specs = state.legal_specs
            action = rng.choice(specs).abstract
            state = state.act(action)
        assert sum(state.engine.stacks.values()) == 2 * config.starting_stack_chips
        assert abs(state.utility(0) + state.utility(1)) < 1e-9


def test_traversal_branches_clone_mutable_engine_state_and_deck():
    config = HUNLGameConfig(starting_stack_bb=10)
    state = AbstractHUNLState.new_hand(config, random.Random(77))
    original_stacks = state.engine.stacks.copy()
    original_deck = tuple(state.engine.deck.cards)
    original_history = state.history
    called = state.act(AbstractAction.CALL)
    checked = called.act(AbstractAction.CHECK)
    assert not state.engine.community_cards
    assert state.engine.stacks == original_stacks
    assert tuple(state.engine.deck.cards) == original_deck
    assert state.history == original_history
    assert len(checked.engine.community_cards) == 3
    assert checked.engine.deck.cards is not state.engine.deck.cards


def test_information_set_is_independent_of_opponent_private_cards():
    config = HUNLGameConfig(starting_stack_bb=10, opponent_hand_samples=8)
    state = AbstractHUNLState.new_hand(config, random.Random(4))
    player = state.current_player
    before = state.information_set(player)
    opponent = state.engine.players[1 - player]
    own_cards = set(state.engine.hole_cards[state.engine.players[player]])
    replacement = tuple(
        card for card in state.engine.deck.cards if card not in own_cards
    )[:2]
    state.engine.hole_cards[opponent] = replacement
    after = state.information_set(player)
    assert before == after


def test_suit_equivalent_preflop_hands_share_information_set_key():
    config = HUNLGameConfig(starting_stack_bb=10, opponent_hand_samples=8)
    state = AbstractHUNLState.new_hand(config, random.Random(24))
    observation = observe(state.engine, state.engine.current_player)
    first = replace(observation, hole_cards=(Card(14, "clubs"), Card(13, "clubs")))
    second = replace(observation, hole_cards=(Card(14, "hearts"), Card(13, "hearts")))
    key_a = information_set_from_observation(first, (), config)
    key_b = information_set_from_observation(second, (), config)
    assert key_a == key_b


def test_importance_weighted_average_matches_exact_own_reach_reference():
    # Controlled two-outcome chance and two-action opponent proposal. For each
    # sampled external outcome z, this estimator contributes pi_i*sigma/q(z).
    # Summing p(z) times that update cancels both sampling probabilities and
    # recovers the vanilla own-reach numerator. The fixed numeric result is
    # calculated here independently from the production traversal.
    chance = (0.25, 0.75)
    opponent_q = ((0.5, 0.5), (0.25, 0.75))
    own_reach = 0.4
    sigma = (0.3, 0.7)
    estimate = [0.0, 0.0]
    for p_chance, q_actions in zip(chance, opponent_q, strict=True):
        for q in q_actions:
            weight = average_strategy_importance_weight(own_reach, p_chance, q)
            for action_index in range(2):
                estimate[action_index] += p_chance * q * weight * sigma[action_index]
    # Four concrete external histories map to this controlled abstract infoset
    # (two chance outcomes by two proposal outcomes), so the raw masses carry
    # that constant multiplicity. Policy normalization must remove it.
    assert estimate == pytest.approx([0.48, 1.12])
    assert [value / sum(estimate) for value in estimate] == [0.3, 0.7]


def test_mccfr_seed_reproduces_updates_and_changes_trajectories_by_seed():
    game = HUNLGameConfig(
        starting_stack_bb=4, postflop_buckets=4, opponent_hand_samples=8
    )
    first = ExternalSamplingMCCFR(HUNLMCCFRConfig(2, 22, game))
    second = ExternalSamplingMCCFR(HUNLMCCFRConfig(2, 22, game))
    other = ExternalSamplingMCCFR(HUNLMCCFRConfig(2, 23, game))
    first.train()
    second.train()
    other.train()
    assert first.regrets == second.regrets
    assert first.strategy_sums == second.strategy_sums
    assert first.regrets != other.regrets


def test_checkpoint_resume_matches_uninterrupted_rng_and_tables():
    game = HUNLGameConfig(
        starting_stack_bb=4,
        postflop_buckets=4,
        opponent_hand_samples=8,
        action_profile="standard",
    )
    config = HUNLMCCFRConfig(4, 77, game)
    uninterrupted = ExternalSamplingMCCFR(config)
    uninterrupted.train(4)

    partial = ExternalSamplingMCCFR(config)
    partial.train(2)
    serialized = json.loads(json.dumps(partial.checkpoint_state()))
    resumed = ExternalSamplingMCCFR(config)
    resumed.load_checkpoint(serialized)
    resumed.train(2)

    assert resumed.regrets == uninterrupted.regrets
    assert resumed.strategy_sums == uninterrupted.strategy_sums
    assert resumed.rng.getstate() == uninterrupted.rng.getstate()
    assert resumed.iterations == uninterrupted.iterations == 4
    assert resumed.traversals == uninterrupted.traversals == 16
    diagnostics = resumed.diagnostics()
    assert diagnostics["information_sets"] == len(resumed.regrets)
    assert diagnostics["positive_cumulative_regret"] >= 0
    assert diagnostics["negative_cumulative_regret_abs"] >= 0


def test_checkpoint_rejects_mismatched_abstraction_or_seed():
    solver = ExternalSamplingMCCFR(HUNLMCCFRConfig(1, 18, HUNLGameConfig()))
    checkpoint = solver.checkpoint_state()
    wrong_game = ExternalSamplingMCCFR(
        HUNLMCCFRConfig(1, 18, HUNLGameConfig(action_profile="wide"))
    )
    with pytest.raises(ValueError, match="configuration"):
        wrong_game.load_checkpoint(checkpoint)
    wrong_seed = ExternalSamplingMCCFR(HUNLMCCFRConfig(1, 19, HUNLGameConfig()))
    with pytest.raises(ValueError, match="seed"):
        wrong_seed.load_checkpoint(checkpoint)


def test_strategy_snapshot_round_trips_stable_information_set_keys():
    config = HUNLGameConfig(
        starting_stack_bb=4, postflop_buckets=4, opponent_hand_samples=8
    )
    solver = ExternalSamplingMCCFR(HUNLMCCFRConfig(1, 31, config))
    solver.train()
    strategy = {info: solver.average_strategy(info) for info in solver.strategy_sums}
    restored = deserialize_strategy(serialize_strategy(strategy))
    assert restored == strategy


class _ToyState:
    """Tiny controlled extensive-form state used only as a regret oracle."""

    def __init__(self, info, player=None, transitions=None, payoff=None):
        self._info = info
        self.current_player = player
        self.transitions = transitions or {}
        self.payoff = payoff

    @property
    def terminal(self):
        return self.payoff is not None

    def information_set(self, player):
        assert player == self.current_player
        return self._info

    def act(self, action):
        return self.transitions[action]

    def utility(self, player):
        return self.payoff if player == 0 else -self.payoff


def test_sampled_regret_increment_matches_independent_exact_two_chance_reference():
    from pokerlab.research.nlhe.information_set import HUNLInformationSet

    actions = (AbstractAction.CHECK, AbstractAction.FOLD)
    solver = ExternalSamplingMCCFR(HUNLMCCFRConfig(iterations=1, seed=1))

    # In hidden outcome A, action utilities are (3, 1); in B they are (1, -1).
    # The independently enumerated exact action values under equal chance are
    # (2, 0), giving node value 0.25*2 + 0.75*0 = 0.5 and regrets (1.5,-0.5).
    expected = {actions[0]: 1.5, actions[1]: -0.5}
    observed = []
    for traverser in (0, 1):
        player_info = HUNLInformationSet(
            traverser, "preflop", "AA|preflop", (), 3, 20, (1, 2), (), actions
        )
        player_profile = {player_info: {actions[0]: 0.25, actions[1]: 0.75}}
        rows = []
        for payoffs in ((3.0, 1.0), (1.0, -1.0)):
            if traverser == 1:
                payoffs = tuple(-value for value in payoffs)
            state = _ToyState(
                player_info,
                player=traverser,
                transitions={
                    actions[0]: _ToyState(player_info, payoff=payoffs[0]),
                    actions[1]: _ToyState(player_info, payoff=payoffs[1]),
                },
            )
            delta = {}
            solver._regret_traverse(state, traverser, player_profile.copy(), delta, [0])
            rows.append(delta[player_info])
        observed.append(
            {action: sum(row[action] for row in rows) / 2 for action in actions}
        )
    assert observed == [expected, expected]


def test_opponent_sampling_regrets_match_exact_fixed_profile_expectation():
    from pokerlab.research.nlhe.information_set import HUNLInformationSet

    own_actions = (AbstractAction.CHECK, AbstractAction.FOLD)
    opponent_actions = (AbstractAction.CALL, AbstractAction.ALL_IN)
    own_info = HUNLInformationSet(
        0, "preflop", "AKs|preflop", (), 5, 20, (1, 2), (), own_actions
    )
    opponent_infos = {
        action: HUNLInformationSet(
            1,
            "preflop",
            "72o|preflop",
            (),
            5,
            20,
            (1, 2),
            (("preflop", True, action.value, None),),
            opponent_actions,
        )
        for action in own_actions
    }
    profile = {
        own_info: {own_actions[0]: 0.5, own_actions[1]: 0.5},
        **{
            info: {
                opponent_actions[0]: 0.25,
                opponent_actions[1]: 0.75,
            }
            for info in opponent_infos.values()
        },
    }
    # Exact action values: CHECK -> .25*0 + .75*4 = 3; FOLD ->
    # .25*2 + .75*0 = .5. Node value is 1.75; regrets are +/-1.25.
    expected = {own_actions[0]: 1.25, own_actions[1]: -1.25}
    solver = ExternalSamplingMCCFR(HUNLMCCFRConfig(iterations=1, seed=983))
    total = dict.fromkeys(own_actions, 0.0)
    samples = 8_000
    for _ in range(samples):
        transitions = {}
        payoffs = ((0.0, 4.0), (2.0, 0.0))
        for own_index, own_action in enumerate(own_actions):
            opp_info = opponent_infos[own_action]
            transitions[own_action] = _ToyState(
                opp_info,
                player=1,
                transitions={
                    opponent_actions[0]: _ToyState(
                        opp_info, payoff=payoffs[own_index][0]
                    ),
                    opponent_actions[1]: _ToyState(
                        opp_info, payoff=payoffs[own_index][1]
                    ),
                },
            )
        state = _ToyState(own_info, player=0, transitions=transitions)
        delta = {}
        solver._regret_traverse(state, 0, profile.copy(), delta, [0])
        for action in own_actions:
            total[action] += delta[own_info][action]
    mean = {action: total[action] / samples for action in own_actions}
    # For either action's regret, variance is 0.25*(3.0 + 0.75)=0.9375;
    # the standard error at 8,000 samples is about 0.0109. The 0.08 bound is
    # over seven standard errors, leaving room for this fixed-seed realization.
    assert mean[own_actions[0]] == pytest.approx(expected[own_actions[0]], abs=0.08)
    assert mean[own_actions[1]] == pytest.approx(expected[own_actions[1]], abs=0.08)


def test_reach_weighted_average_strategy_matches_exact_own_reach_on_small_tree():
    """Validate derived reach_weighted estimator against exact sequence-form own-reach.

    Tree structure:
    - Chance deals outcome z in {0, 1} with p_c = (0.3, 0.7).
    - Player 0 (root info I0) chooses a0 in {CHECK, BET_POT}.
    - After BET_POT, Player 1 (info I1_z depending on z) chooses a1 in {CALL, RAISE_POT}.
    - After RAISE_POT, Player 0 reaches child info I2 (shared across z) and chooses
      a2 in {CALL, FOLD}.

    Across two iterations t=1, 2:
    - Iteration 1:
      sigma_0^1(I0, BET_POT) = 0.8, sigma_0^1(I2) = {CALL: 0.9, FOLD: 0.1}
      sigma_1^1(I1_0, RAISE_POT) = 0.1, sigma_1^1(I1_1, RAISE_POT) = 0.1
    - Iteration 2:
      sigma_0^2(I0, BET_POT) = 0.4, sigma_0^2(I2) = {CALL: 0.2, FOLD: 0.8}
      sigma_1^2(I1_0, RAISE_POT) = 0.9, sigma_1^2(I1_1, RAISE_POT) = 0.9

    Exact sequence-form own-reach accumulation at I2:
      Iteration 1 own reach: pi_0^1(I2) = 0.8 -> mass {CALL: 0.72, FOLD: 0.08}
      Iteration 2 own reach: pi_0^2(I2) = 0.4 -> mass {CALL: 0.08, FOLD: 0.32}
      Total own-reach mass: {CALL: 0.80, FOLD: 0.40} -> normalized {CALL: 2/3, FOLD: 1/3}.
    (Note: A biased full-reach estimator weighting by pi_{-i}^t would weight Iteration 1
    by 0.8*0.1=0.08 and Iteration 2 by 0.4*0.9=0.36, yielding CALL = (0.072+0.072)/0.44 = 0.327
    instead of 2/3 = 0.667!)
    """
    from pokerlab.research.nlhe.information_set import HUNLInformationSet
    from pokerlab.research.nlhe.mccfr import average_strategy_reach_weight

    a_root = (AbstractAction.CHECK, AbstractAction.BET_POT)
    a_opp = (AbstractAction.CALL, AbstractAction.RAISE_POT)
    a_leaf = (AbstractAction.CALL, AbstractAction.FOLD)

    info_root = HUNLInformationSet(
        0, "preflop", "AA|preflop", (), 3, 20, (1, 2), (), a_root
    )
    info_opp_0 = HUNLInformationSet(
        1, "preflop", "KK|preflop", (), 6, 20, (4, 2), (), a_opp
    )
    info_opp_1 = HUNLInformationSet(
        1, "preflop", "QQ|preflop", (), 6, 20, (4, 2), (), a_opp
    )
    info_leaf = HUNLInformationSet(
        0, "preflop", "AA|preflop", (), 12, 20, (4, 8), (), a_leaf
    )

    profiles = (
        {
            info_root: {AbstractAction.CHECK: 0.2, AbstractAction.BET_POT: 0.8},
            info_opp_0: {AbstractAction.CALL: 0.9, AbstractAction.RAISE_POT: 0.1},
            info_opp_1: {AbstractAction.CALL: 0.9, AbstractAction.RAISE_POT: 0.1},
            info_leaf: {AbstractAction.CALL: 0.9, AbstractAction.FOLD: 0.1},
        },
        {
            info_root: {AbstractAction.CHECK: 0.6, AbstractAction.BET_POT: 0.4},
            info_opp_0: {AbstractAction.CALL: 0.1, AbstractAction.RAISE_POT: 0.9},
            info_opp_1: {AbstractAction.CALL: 0.1, AbstractAction.RAISE_POT: 0.9},
            info_leaf: {AbstractAction.CALL: 0.2, AbstractAction.FOLD: 0.8},
        },
    )
    chance_probs = ((info_opp_0, 0.3), (info_opp_1, 0.7))

    # 1. Exact analytical expectation of average_strategy_reach_weight over chance & uniform q_{-i}
    analytical_mass = {AbstractAction.CALL: 0.0, AbstractAction.FOLD: 0.0}
    for prof in profiles:
        own_reach = prof[info_root][AbstractAction.BET_POT]
        for _opp_info, p_c in chance_probs:
            q_opp = 0.5  # uniform proposal over 2 opponent actions
            visit_prob = p_c * q_opp
            w = average_strategy_reach_weight(own_reach, q_opp)
            for act in a_leaf:
                analytical_mass[act] += visit_prob * w * prof[info_leaf][act]

    norm_analytical = {
        act: analytical_mass[act] / sum(analytical_mass.values()) for act in a_leaf
    }
    assert norm_analytical[AbstractAction.CALL] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert norm_analytical[AbstractAction.FOLD] == pytest.approx(1.0 / 3.0, abs=1e-12)

    # 2. Validate production _average_traverse Monte Carlo convergence on the same tree
    solver = ExternalSamplingMCCFR(
        HUNLMCCFRConfig(
            iterations=1,
            seed=42,
            game=HUNLGameConfig(average_strategy_mode="reach_weighted"),
        )
    )

    def make_tree(opp_info):
        leaf_node = _ToyState(
            info_leaf,
            player=0,
            transitions={
                AbstractAction.CALL: _ToyState(info_leaf, payoff=1.0),
                AbstractAction.FOLD: _ToyState(info_leaf, payoff=-1.0),
            },
        )
        opp_node = _ToyState(
            opp_info,
            player=1,
            transitions={
                AbstractAction.CALL: _ToyState(opp_info, payoff=0.0),
                AbstractAction.RAISE_POT: leaf_node,
            },
        )
        return _ToyState(
            info_root,
            player=0,
            transitions={
                AbstractAction.CHECK: _ToyState(info_root, payoff=0.0),
                AbstractAction.BET_POT: opp_node,
            },
        )

    mc_delta: dict = {}
    trials = 6_000
    for _ in range(trials):
        opp_info = info_opp_0 if solver.rng.random() < 0.3 else info_opp_1
        tree = make_tree(opp_info)
        for prof in profiles:
            solver._average_traverse(tree, 0, prof.copy(), mc_delta, 1.0, 1.0, [0])

    mc_leaf = mc_delta[info_leaf]
    mc_norm = {act: mc_leaf[act] / sum(mc_leaf.values()) for act in a_leaf}
    assert mc_norm[AbstractAction.CALL] == pytest.approx(2.0 / 3.0, abs=0.015)
    assert mc_norm[AbstractAction.FOLD] == pytest.approx(1.0 / 3.0, abs=0.015)


def test_all_24_suit_permutations_and_street_order_invariance():
    from itertools import permutations

    from pokerlab.research.nlhe.abstraction import SUITS
    from pokerlab.research.nlhe.information_set import canonicalize_board_representation

    base_hole_suited = (Card(14, "spades"), Card(13, "spades"))
    base_hole_pair = (Card(14, "clubs"), Card(14, "spades"))
    base_board = (
        Card(10, "clubs"),
        Card(8, "spades"),
        Card(2, "diamonds"),
        Card(5, "hearts"),
    )

    for method in ("sampled_strength", "sampled_equity"):
        cfg = HUNLGameConfig(
            postflop_method=method,
            postflop_buckets=12,
            opponent_hand_samples=16,
            equity_samples=16,
            board_abstraction="isomorphic",
        )
        for base_hole in (base_hole_suited, base_hole_pair):
            ref_bucket = private_hand_bucket(base_hole, base_board, cfg)
            ref_board = canonicalize_board_representation(
                base_board, mode="isomorphic", hole_cards=base_hole
            )
            for perm in permutations(SUITS):
                suit_map = dict(zip(SUITS, perm, strict=True))
                perm_hole = tuple(Card(c.rank, suit_map[c.suit]) for c in base_hole)
                perm_board = tuple(Card(c.rank, suit_map[c.suit]) for c in base_board)
                assert private_hand_bucket(perm_hole, perm_board, cfg) == ref_bucket
                assert (
                    canonicalize_board_representation(
                        perm_board, mode="isomorphic", hole_cards=perm_hole
                    )
                    == ref_board
                )

    # Street order: Flop (2c 3d 4h) + Turn (As) != Flop (As 4h 3d) + Turn (2c)
    board_a = (
        Card(2, "clubs"),
        Card(3, "diamonds"),
        Card(4, "hearts"),
        Card(14, "spades"),
    )
    board_b = (
        Card(14, "spades"),
        Card(4, "hearts"),
        Card(3, "diamonds"),
        Card(2, "clubs"),
    )
    assert canonicalize_board_representation(
        board_a, mode="isomorphic"
    ) != canonicalize_board_representation(board_b, mode="isomorphic")

    # Private-to-public suit relation: Ac Kc on Qc 7d 2h vs Ah Kh on Qc 7d 2s
    flop = (Card(12, "clubs"), Card(7, "diamonds"), Card(2, "hearts"))
    flop_alt = (Card(12, "clubs"), Card(7, "diamonds"), Card(2, "spades"))
    iso_with_flush_blocker = canonicalize_board_representation(
        flop, mode="isomorphic", hole_cards=(Card(14, "clubs"), Card(13, "clubs"))
    )
    iso_without_flush_blocker = canonicalize_board_representation(
        flop_alt, mode="isomorphic", hole_cards=(Card(14, "hearts"), Card(13, "hearts"))
    )
    assert iso_with_flush_blocker != iso_without_flush_blocker


def test_card_aware_tactical_fallback_and_lookup_telemetry():
    from pokerlab.agents.base import Action, Observation
    from pokerlab.research.nlhe.agent import AbstractHUNLAgent

    cfg = HUNLGameConfig(postflop_method="sampled_equity", postflop_buckets=20)
    agent = AbstractHUNLAgent({}, cfg, seed=1)

    dry_board = (
        Card(13, "spades"),
        Card(12, "hearts"),
        Card(11, "diamonds"),
        Card(9, "clubs"),
        Card(4, "spades"),
    )
    # 1. 7-2 offsuit facing a half-pot bet (pot=20, to_call=10 -> pot_odds=0.333) must FOLD
    weak_obs = Observation(
        player="p0",
        hole_cards=(Card(7, "clubs"), Card(2, "diamonds")),
        community_cards=dry_board,
        street="river",
        in_position=True,
        stack=100,
        opponent_stack=100,
        pot=20,
        to_call=10,
        big_blind=2,
        min_raise=10,
        legal_actions=(
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
            Action(ActionType.ALL_IN),
        ),
        street_contribution=0,
        opponent_street_contribution=10,
        is_button=True,
    )
    assert agent.act(weak_obs) == Action(ActionType.FOLD)

    # 2. Nut straight (Ace-Ten) facing a pot-sized bet (pot=40, to_call=20 -> pot_odds=0.333 or pot=20, to_call=20 -> pot_odds=0.50) must NOT fold
    nut_obs = replace(
        weak_obs,
        hole_cards=(Card(14, "clubs"), Card(10, "diamonds")),
        pot=20,
        to_call=20,
        opponent_street_contribution=20,
    )
    assert agent.act(nut_obs) == Action(ActionType.CALL)

    rates = agent.lookup_rates()
    assert rates["decisions"] == 2
    assert rates["fallback_hits"] == 2
    assert rates["fallback_rate"] == 1.0
