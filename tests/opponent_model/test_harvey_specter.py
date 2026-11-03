"""Comprehensive test suite for Phase 10: HarveySpecter Opponent Intelligence & Exploitative Strategy."""

from __future__ import annotations

from pathlib import Path

import pytest

from pokerlab.agents import ActionRecord, Observation, observe
from pokerlab.arena.server import ArenaApp
from pokerlab.arena.session import ArenaConfig, ArenaSession
from pokerlab.engine import Action, ActionType, Card, Deck, GameState
from pokerlab.opponent_model import (
    ALL_COMBOS,
    ArchetypeClassifier,
    BayesianRangeModel,
    BehaviouralProfiler,
    CallingStationAgent,
    ExploitativeStrategyEngine,
    HarveyBenchmarkConfig,
    HarveySpecterAgent,
    HarveySpecterConfig,
    ManiacAgent,
    NitAgent,
    ObservedActionEvent,
    OpponentDataCollector,
    OpponentProfile,
    OpponentProfileStore,
    ProfileStorageError,
    run_harvey_specter_benchmark,
)
from pokerlab.research.nlhe.agent import AbstractHUNLAgent
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.mccfr import (
    ExternalSamplingMCCFR,
    HUNLMCCFRConfig,
)


@pytest.fixture(scope="module")
def trained_mccfr_agent() -> AbstractHUNLAgent:
    cfg = HUNLGameConfig(
        starting_stack_bb=20.0,
        small_blind_bb=0.5,
        big_blind_bb=1.0,
        chips_per_bb=2,
        postflop_method="sampled_equity",
        postflop_buckets=8,
        equity_samples=8,
        board_abstraction="isomorphic",
    )
    solver = ExternalSamplingMCCFR(HUNLMCCFRConfig(iterations=4, seed=42, game=cfg))
    solver.train(4)
    strategy = solver.average_strategy_profile()
    strategy_sums = {k: dict(v) for k, v in solver.strategy_sums.items()}
    return AbstractHUNLAgent(
        strategy=strategy,
        config=cfg,
        seed=42,
        strategy_sums=strategy_sums,
    )


# ============================================================================
# 1. Opponent Data Collector & Hidden-Card Privacy Tests
# ============================================================================


def test_collector_action_recording_positions_and_hand_association():
    collector = OpponentDataCollector(default_opponent_id="Villain")
    hand1 = collector.start_hand(
        opponent_id="Villain",
        hand_number=1,
        opponent_is_button=True,
        starting_stack=200,
        effective_stack=200,
        big_blind=2,
    )
    assert hand1.opponent_position == "button"
    assert hand1.opponent_is_button is True

    # Villain (button) raises to 6 preflop; Hero calls; Flop Villains bets 8
    history = (
        ActionRecord("Villain", Action(ActionType.RAISE, 6), "preflop"),
        ActionRecord("Hero", Action(ActionType.CALL), "preflop"),
        ActionRecord("Hero", Action(ActionType.CHECK), "flop"),
        ActionRecord("Villain", Action(ActionType.BET, 8), "flop"),
    )
    obs = Observation(
        player="Hero",
        hole_cards=(Card(14, "spades"), Card(13, "spades")),
        community_cards=(
            Card(12, "hearts"),
            Card(7, "diamonds"),
            Card(2, "clubs"),
        ),
        street="flop",
        in_position=False,
        stack=194,
        opponent_stack=186,
        pot=20,
        to_call=8,
        big_blind=2,
        min_raise=8,
        legal_actions=(Action(ActionType.FOLD), Action(ActionType.CALL)),
        history=history,
        street_contribution=0,
        opponent_street_contribution=8,
        is_button=False,
    )

    rec, new_opp = collector.ingest_observation(
        obs, opponent_id="Villain", hand_number=1
    )
    assert rec.hand_number == 1
    assert rec.opponent_id == "Villain"
    assert rec.saw_flop is True
    assert len(rec.preflop_actions) == 2
    assert len(rec.flop_actions) == 2
    assert len(new_opp) == 2
    assert new_opp[0].street == "preflop"
    assert new_opp[0].action_type == "raise"
    assert new_opp[0].position == "button"
    assert new_opp[0].in_position is False  # Button acts first preflop
    assert new_opp[1].street == "flop"
    assert new_opp[1].action_type == "bet"
    assert new_opp[1].in_position is True  # Button acts last postflop

    # Ingesting the exact same observation again must NOT duplicate actions
    _, duplicate_opp = collector.ingest_observation(
        obs, opponent_id="Villain", hand_number=1
    )
    assert duplicate_opp == []
    assert len(rec.actions) == 4


def test_collector_showdown_folded_incomplete_and_privacy_enforcement():
    collector = OpponentDataCollector(default_opponent_id="Villain")
    collector.start_hand(
        opponent_id="Villain",
        hand_number=1,
        opponent_is_button=False,
        starting_stack=200,
    )
    # Attempting to pass opponent hole cards when reached_showdown=False MUST raise ValueError
    with pytest.raises(ValueError, match="Privacy violation"):
        collector.finalize_hand(
            opponent_id="Villain",
            hand_number=1,
            reached_showdown=False,
            revealed_opponent_cards=(Card(14, "hearts"), Card(14, "clubs")),
        )

    # Folded hand cannot be marked as reached_showdown=True
    collector.record_action(
        opponent_id="Villain",
        hand_number=1,
        actor="Villain",
        is_opponent=True,
        action=Action(ActionType.FOLD),
        street="preflop",
        pot_before=3,
        to_call_before=1,
        stack_before=198,
        effective_stack_before=198,
        opponent_is_button=False,
    )
    with pytest.raises(ValueError, match="fold"):
        collector.finalize_hand(
            opponent_id="Villain",
            hand_number=1,
            reached_showdown=True,
        )

    folded_rec = collector.finalize_hand(
        opponent_id="Villain",
        hand_number=1,
        reached_showdown=False,
        opponent_net_chips=-2,
    )
    assert folded_rec.opponent_folded is True
    assert folded_rec.reached_showdown is False
    assert folded_rec.revealed_hole_cards is None

    # Legitimate showdown hand records revealed cards
    collector.start_hand(
        opponent_id="Villain",
        hand_number=2,
        opponent_is_button=True,
        starting_stack=200,
    )
    sd_cards = (Card(13, "hearts"), Card(13, "diamonds"))
    sd_rec = collector.finalize_hand(
        opponent_id="Villain",
        hand_number=2,
        reached_showdown=True,
        revealed_opponent_cards=sd_cards,
        opponent_net_chips=20,
        opponent_won=True,
    )
    assert sd_rec.reached_showdown is True
    assert sd_rec.revealed_hole_cards == sd_cards


# ============================================================================
# 2. Behavioural Profiler Tests
# ============================================================================


def test_behavioural_profiler_zero_denominators_and_priors():
    profiler = BehaviouralProfiler()
    empty_snap = profiler.compute_snapshot([], opponent_id="ColdStart")

    assert empty_snap.hands_observed == 0
    assert empty_snap.vpip.trials == 0
    assert empty_snap.vpip.raw_value is None
    assert 0.0 < empty_snap.vpip.smoothed_value < 1.0
    assert empty_snap.vpip.confidence == 0.0
    assert empty_snap.vpip.uncertainty == 1.0
    assert empty_snap.raw_aggression_factor is None
    assert empty_snap.aggression_factor == pytest.approx(1.0)
    assert empty_snap.fold_to_river_bet.raw_value is None


def test_behavioural_profiler_metrics_smoothing_and_confidence():
    collector = OpponentDataCollector(default_opponent_id="Aggro")
    profiler = BehaviouralProfiler(recency_decay=1.0)

    # Hand 1: Opponent on button raises preflop, Hero 3-bets, Opponent calls;
    # Flop Hero c-bets, Opponent raises; River Opponent bets and wins at showdown.
    collector.start_hand(
        opponent_id="Aggro",
        hand_number=1,
        opponent_is_button=True,
        starting_stack=200,
        big_blind=2,
    )
    h1_history = (
        ActionRecord("Aggro", Action(ActionType.RAISE, 6), "preflop"),
        ActionRecord("Hero", Action(ActionType.RAISE, 18), "preflop"),
        ActionRecord("Aggro", Action(ActionType.CALL), "preflop"),
        ActionRecord("Hero", Action(ActionType.BET, 16), "flop"),
        ActionRecord("Aggro", Action(ActionType.RAISE, 40), "flop"),
        ActionRecord("Hero", Action(ActionType.CALL), "flop"),
        ActionRecord("Hero", Action(ActionType.CHECK), "turn"),
        ActionRecord("Aggro", Action(ActionType.CHECK), "turn"),
        ActionRecord("Hero", Action(ActionType.BET, 30), "river"),
        ActionRecord("Aggro", Action(ActionType.FOLD), "river"),
    )
    collector.finalize_hand(
        opponent_id="Aggro",
        hand_number=1,
        history=h1_history,
        hero_player="Hero",
        community_cards=(
            Card(10, "spades"),
            Card(9, "hearts"),
            Card(2, "clubs"),
            Card(4, "diamonds"),
            Card(14, "clubs"),
        ),
        reached_showdown=False,
        opponent_net_chips=-58,
    )

    snap1 = profiler.compute_snapshot(
        collector.list_hands("Aggro"), opponent_id="Aggro"
    )
    # Raw VPIP is 1/1 = 1.0, but Beta smoothing prevents jumping to 100% after 1 hand!
    assert snap1.vpip.raw_value == pytest.approx(1.0)
    assert snap1.vpip.smoothed_value < 0.70
    assert snap1.pfr.raw_value == pytest.approx(1.0)
    assert snap1.pfr.smoothed_value < 0.60
    assert snap1.fold_to_three_bet.trials == 1
    assert snap1.fold_to_three_bet.successes == 0
    assert snap1.fold_to_cbet.trials == 1
    assert snap1.fold_to_cbet.successes == 0
    assert snap1.fold_to_river_bet.trials == 1
    assert snap1.fold_to_river_bet.successes == 1
    assert snap1.fold_to_river_bet.smoothed_value < 0.75
    assert snap1.vpip_button.trials == 1
    assert snap1.vpip_big_blind.trials == 0
    assert snap1.vpip.ci95_low <= snap1.vpip.smoothed_value <= snap1.vpip.ci95_high


# ============================================================================
# 3. Bayesian Opponent Range Model Tests
# ============================================================================


def test_bayesian_range_model_prior_card_removal_and_normalization():
    model = BayesianRangeModel()
    assert len(ALL_COMBOS) == 1326

    hero = (Card(14, "spades"), Card(14, "hearts"))
    flop = (Card(14, "clubs"), Card(13, "spades"), Card(2, "diamonds"))
    model.start_hand(hand_number=1, hero_hole=hero, community_cards=flop)

    snap = model.snapshot()
    # 52 - 5 known cards = 47 remaining -> C(47, 2) = 1081 valid combinations
    assert snap.active_combo_count == 1081
    assert snap.eliminated_combo_count == 1326 - 1081
    assert sum(model.prior_weights.values()) == pytest.approx(1.0)
    assert sum(model.posterior_weights.values()) == pytest.approx(1.0)

    # Any combination containing a dead card must have exact 0 probability
    for combo, weight in model.posterior_weights.items():
        if combo[0] in set(hero) | set(flop) or combo[1] in set(hero) | set(flop):
            assert weight == 0.0
        else:
            assert weight > 0.0


def test_bayesian_range_action_updates_fold_handling_and_deduplication():
    model = BayesianRangeModel()
    hero = (Card(8, "clubs"), Card(7, "clubs"))
    model.start_hand(hand_number=1, hero_hole=hero)
    prior_mean = model.mean_strength(use_prior=True)

    # Opponent 3-bets preflop
    raise_event = ObservedActionEvent(
        hand_number=1,
        event_index=1,
        opponent_id="Villain",
        actor="Villain",
        is_opponent=True,
        street="preflop",
        action_type="raise",
        amount=18,
        paid_amount=16,
        pot_before=9,
        to_call_before=4,
        stack_before=198,
        effective_stack_before=198,
        position="big_blind",
        in_position=False,
        pot_ratio=1.1,
        facing_bet=True,
        facing_raise_count=1,
    )
    applied = model.update_from_action(raise_event)
    assert applied is True
    assert sum(model.posterior_weights.values()) == pytest.approx(1.0)

    post_mean = model.mean_strength(use_prior=False)
    assert post_mean > prior_mean

    # Premium class AA should have much higher posterior probability than 72o
    top_classes = dict(model.top_class_probabilities(limit=169))
    assert top_classes["AA"] > top_classes["72o"] * 3.0

    # Calling update_from_action again with the same (hand_number, event_index) is a no-op
    weights_after_first = dict(model.posterior_weights)
    applied_again = model.update_from_action(raise_event)
    assert applied_again is False
    assert model.posterior_weights == weights_after_first

    # Fold handling: retains pre-fold posterior without pretending folded hand is known
    fold_event = ObservedActionEvent(
        hand_number=1,
        event_index=3,
        opponent_id="Villain",
        actor="Villain",
        is_opponent=True,
        street="flop",
        action_type="fold",
        amount=None,
        paid_amount=0,
        pot_before=36,
        to_call_before=24,
        stack_before=182,
        effective_stack_before=182,
        position="big_blind",
        in_position=False,
        pot_ratio=0.0,
        facing_bet=True,
        facing_raise_count=1,
    )
    model.update_from_action(fold_event)
    assert model.is_folded is True
    assert sum(model.posterior_weights.values()) == pytest.approx(1.0)
    assert model.posterior_weights == weights_after_first

    # Showdown calibration update is idempotent per hand_number
    sd_updated = model.record_showdown(
        hand_number=2,
        revealed_cards=(Card(2, "hearts"), Card(7, "diamonds")),
        community_cards=(
            Card(14, "spades"),
            Card(13, "hearts"),
            Card(12, "clubs"),
            Card(10, "diamonds"),
            Card(3, "spades"),
        ),
        opponent_actions=(raise_event,),
    )
    assert sd_updated is True
    assert model.calibration.showdowns_recorded == 1
    assert model.calibration.bluff_showdowns == 1
    assert (
        model.record_showdown(
            hand_number=2,
            revealed_cards=(Card(2, "hearts"), Card(7, "diamonds")),
            community_cards=(),
            opponent_actions=(raise_event,),
        )
        is False
    )
    assert model.calibration.showdowns_recorded == 1


# ============================================================================
# 4. Opponent Archetype Classifier Tests
# ============================================================================


def _simulate_hands_into_profile(
    opponent_agent,
    *,
    hands: int = 30,
    seed: int = 100,
    profile: OpponentProfile | None = None,
) -> OpponentProfile:
    prof = profile or OpponentProfile(opponent_id="TestOpp")
    caller = CallingStationAgent(seed=seed + 1)
    import random

    rng = random.Random(seed)
    start_idx = prof.last_hand_number
    for idx in range(hands):
        hand_num = start_idx + idx + 1
        button = "Hero" if idx % 2 == 0 else "Villain"
        game = GameState(
            {"Hero": 200, "Villain": 200},
            small_blind=1,
            big_blind=2,
            button=button,
            deck=Deck(seed=rng.randrange(2**32)),
        )
        prof.start_hand(
            hand_number=hand_num,
            opponent_is_button=(button == "Villain"),
            starting_stack=200,
            big_blind=2,
            hero_hole=game.hole_cards["Hero"],
        )
        history: list[ActionRecord] = []
        while not game.complete:
            actor = game.current_player
            assert actor is not None
            obs = observe(game, actor, tuple(history))
            act = caller.act(obs) if actor == "Hero" else opponent_agent.act(obs)
            street = game.street.value
            game.act(act, player=actor)
            history.append(ActionRecord(actor, act, street))
        reached_sd = bool(game.hand_ranks and "Villain" in game.hand_ranks)
        prof.finalize_hand(
            hand_number=hand_num,
            history=tuple(history),
            hero_player="Hero",
            community_cards=tuple(game.community_cards),
            reached_showdown=reached_sd,
            revealed_opponent_cards=(
                game.hole_cards["Villain"] if reached_sd else None
            ),
            opponent_net_chips=game.stacks["Villain"] - 200,
            final_pot=game.current_pot,
        )
    return prof


def test_archetype_classifier_insufficient_data_calling_station_nit_maniac_and_drift():
    classifier = ArchetypeClassifier()

    # 1. Insufficient data (2 hands) -> balanced_unknown with low confidence
    sparse_prof = _simulate_hands_into_profile(
        CallingStationAgent(seed=1), hands=2, seed=10
    )
    sparse_arch = classifier.classify(sparse_prof.stats_snapshot())
    assert sparse_arch.most_likely == "balanced_unknown"
    assert sparse_arch.confidence < 0.30

    # 2. Calling Station (35 hands) -> calling_station
    cs_prof = _simulate_hands_into_profile(
        CallingStationAgent(seed=2), hands=35, seed=20
    )
    cs_arch = classifier.classify(cs_prof.stats_snapshot())
    assert cs_arch.most_likely == "calling_station"
    assert cs_arch.probabilities["calling_station"] > 0.35

    # 3. Nit (35 hands) -> nit
    nit_prof = _simulate_hands_into_profile(NitAgent(seed=3), hands=35, seed=30)
    nit_arch = classifier.classify(nit_prof.stats_snapshot())
    assert nit_arch.most_likely in ("nit", "tight_passive")
    assert (
        nit_arch.probabilities["nit"] + nit_arch.probabilities["tight_passive"] > 0.50
    )

    # 4. Maniac (35 hands) -> maniac
    maniac_prof = _simulate_hands_into_profile(ManiacAgent(seed=4), hands=35, seed=40)
    maniac_arch = classifier.classify(maniac_prof.stats_snapshot())
    assert maniac_arch.most_likely in ("maniac", "loose_aggressive")
    assert (
        maniac_arch.probabilities["maniac"]
        + maniac_arch.probabilities["loose_aggressive"]
        > 0.50
    )

    # 5. Behaviour shift: Nit for 20 hands, then Maniac for 25 hands
    shift_prof = OpponentProfile(opponent_id="Shifter", recency_decay=0.92)
    _simulate_hands_into_profile(
        NitAgent(seed=5), hands=20, seed=50, profile=shift_prof
    )
    before_shift = shift_prof.archetype_distribution()
    assert before_shift.most_likely in ("nit", "tight_passive")

    _simulate_hands_into_profile(
        ManiacAgent(seed=6), hands=30, seed=60, profile=shift_prof
    )
    after_shift = shift_prof.archetype_distribution()
    # Aggressive probabilities must increase dramatically after switching to Maniac
    assert (
        after_shift.probabilities["maniac"]
        + after_shift.probabilities["loose_aggressive"]
    ) > (
        before_shift.probabilities["maniac"]
        + before_shift.probabilities["loose_aggressive"]
    )


# ============================================================================
# 5. Adaptive Strategy & Exploitative Engine Tests
# ============================================================================


def test_adaptive_strategy_modes_cold_start_evidence_and_stability(
    trained_mccfr_agent: AbstractHUNLAgent,
):
    obs = Observation(
        player="Hero",
        hole_cards=(Card(14, "hearts"), Card(13, "hearts")),
        community_cards=(
            Card(14, "clubs"),
            Card(8, "diamonds"),
            Card(3, "spades"),
        ),
        street="flop",
        in_position=True,
        stack=180,
        opponent_stack=180,
        pot=20,
        to_call=0,
        big_blind=2,
        min_raise=2,
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.BET, 10),
            Action(ActionType.BET, 20),
            Action(ActionType.ALL_IN),
        ),
        history=(
            ActionRecord("Hero", Action(ActionType.RAISE, 6), "preflop"),
            ActionRecord("Villain", Action(ActionType.CALL), "preflop"),
            ActionRecord("Villain", Action(ActionType.CHECK), "flop"),
        ),
        street_contribution=0,
        opponent_street_contribution=0,
        is_button=True,
    )

    # 1. Baseline-only mode: lambda == 0.0 and final probabilities match MCCFR exactly
    baseline_only = HarveySpecterAgent(
        trained_mccfr_agent,
        config=HarveySpecterConfig(mode="baseline_only"),
        opponent_id="Villain",
        seed=42,
    )
    dist_base, rep_base = baseline_only.evaluate_decision(obs, ingest=False)
    mccfr_direct = trained_mccfr_agent.action_probabilities(obs)
    assert rep_base.lambda_value == 0.0
    for act in obs.legal_actions:
        assert dist_base[act] == pytest.approx(mccfr_direct[act], abs=1e-6)

    # 2. Cold-start adaptive mode (0 prior hands): conservative lambda
    adaptive = HarveySpecterAgent(
        trained_mccfr_agent,
        config=HarveySpecterConfig(mode="adaptive", max_lambda=0.80),
        opponent_id="Station",
        seed=42,
    )
    dist_cold, rep_cold = adaptive.evaluate_decision(obs, ingest=False)
    assert rep_cold.lambda_value == 0.0
    assert sum(dist_cold.values()) == pytest.approx(1.0)
    assert set(dist_cold.keys()) == set(obs.legal_actions)

    # 3. After observing 35 hands against a Calling Station, lambda increases significantly
    _simulate_hands_into_profile(
        CallingStationAgent(seed=7),
        hands=35,
        seed=70,
        profile=adaptive.profile,
    )
    dist_warm, rep_warm = adaptive.evaluate_decision(obs, ingest=False)
    assert rep_warm.lambda_value > 0.35
    assert sum(dist_warm.values()) == pytest.approx(1.0)
    chosen = adaptive.act(obs)
    assert chosen in obs.legal_actions

    # 4. When opponent abruptly changes behavior (reducing stability_score), lambda scales down
    shifter_agent = HarveySpecterAgent(
        trained_mccfr_agent,
        config=HarveySpecterConfig(mode="adaptive", max_lambda=0.80),
        opponent_id="Shifter",
        seed=42,
    )
    _simulate_hands_into_profile(
        NitAgent(seed=8), hands=15, seed=80, profile=shifter_agent.profile
    )
    _, rep_stable = shifter_agent.evaluate_decision(obs, ingest=False)
    _simulate_hands_into_profile(
        ManiacAgent(seed=9), hands=10, seed=90, profile=shifter_agent.profile
    )
    _, rep_unstable = shifter_agent.evaluate_decision(obs, ingest=False)
    assert rep_unstable.stability_score < rep_stable.stability_score


def test_exploitative_engine_suppresses_bluffs_vs_station_and_increases_vs_nit():
    engine = ExploitativeStrategyEngine(equity_simulations=40)

    # Weak air hand on river when checked to (7-2 offsuit missed board)
    river_bluff_obs = Observation(
        player="Hero",
        hole_cards=(Card(7, "clubs"), Card(2, "diamonds")),
        community_cards=(
            Card(14, "spades"),
            Card(13, "hearts"),
            Card(11, "diamonds"),
            Card(9, "clubs"),
            Card(4, "spades"),
        ),
        street="river",
        in_position=True,
        stack=150,
        opponent_stack=150,
        pot=40,
        to_call=0,
        big_blind=2,
        min_raise=2,
        legal_actions=(
            Action(ActionType.CHECK),
            Action(ActionType.BET, 20),
            Action(ActionType.BET, 40),
        ),
        history=(),
        is_button=True,
    )

    station_prof = _simulate_hands_into_profile(
        CallingStationAgent(seed=11), hands=35, seed=110
    )
    nit_prof = _simulate_hands_into_profile(NitAgent(seed=12), hands=35, seed=120)

    eval_vs_station = engine.evaluate(
        river_bluff_obs,
        station_prof.stats_snapshot(),
        station_prof.archetype_distribution(),
        station_prof.range_model,
        station_prof.range_snapshot(),
    )
    eval_vs_nit = engine.evaluate(
        river_bluff_obs,
        nit_prof.stats_snapshot(),
        nit_prof.archetype_distribution(),
        nit_prof.range_model,
        nit_prof.range_snapshot(),
    )

    check_action = Action(ActionType.CHECK)
    # Hero checks weak air much more often against Calling Station than against Nit
    assert (
        eval_vs_station.probabilities[check_action]
        > eval_vs_nit.probabilities[check_action]
    )


# ============================================================================
# 6. Persistence Tests (Save, Load, Corrupt, Missing, Reset, Isolation)
# ============================================================================


def test_profile_persistence_save_load_corrupt_missing_reset_and_isolation(
    tmp_path: Path,
):
    store = OpponentProfileStore(storage_dir=tmp_path)

    # Multiple isolated opponent profiles
    prof_a = _simulate_hands_into_profile(
        CallingStationAgent(seed=21),
        hands=12,
        seed=210,
        profile=store.get_or_create("Player_A"),
    )
    prof_b = _simulate_hands_into_profile(
        NitAgent(seed=22),
        hands=8,
        seed=220,
        profile=store.get_or_create("Player_B"),
    )
    assert prof_a.stats_snapshot().hands_observed == 12
    assert prof_b.stats_snapshot().hands_observed == 8
    assert (
        prof_a.stats_snapshot().vpip.smoothed_value
        > prof_b.stats_snapshot().vpip.smoothed_value
    )

    # Save and deterministic load
    saved_path = store.save_profile("Player_A")
    assert saved_path.exists()

    store2 = OpponentProfileStore(storage_dir=tmp_path)
    loaded_a, status = store2.load_profile("Player_A", strict=True)
    assert status == "loaded"
    assert loaded_a.stats_snapshot().to_dict() == prof_a.stats_snapshot().to_dict()
    assert (
        loaded_a.archetype_distribution().to_dict()
        == prof_a.archetype_distribution().to_dict()
    )

    # Missing file handling
    missing_prof, missing_status = store2.load_profile(
        "NonExistentOpponent", strict=False
    )
    assert missing_status == "missing_created_new"
    assert missing_prof.stats_snapshot().hands_observed == 0
    with pytest.raises(FileNotFoundError):
        store2.load_profile("NonExistentStrict", strict=True)

    # Corrupt file handling
    corrupt_path = tmp_path / "profile_CorruptUser.json"
    corrupt_path.write_text("{not valid json!!", encoding="utf-8")
    recovered_prof, corrupt_status = store2.load_profile(
        "CorruptUser", path=corrupt_path, strict=False
    )
    assert corrupt_status == "corrupt_recovered_new"
    assert recovered_prof.stats_snapshot().hands_observed == 0
    with pytest.raises(ProfileStorageError):
        store2.load_profile("CorruptUser", path=corrupt_path, strict=True)

    # Reset profile
    store.reset_profile("Player_A")
    assert store.get_or_create("Player_A").stats_snapshot().hands_observed == 0
    # Player_B remains untouched
    assert store.get_or_create("Player_B").stats_snapshot().hands_observed == 8


# ============================================================================
# 7. Arena & Policy Inspector Integration Tests
# ============================================================================


def test_arena_harvey_specter_gameplay_inspector_profile_routes_and_privacy(
    tmp_path: Path,
):
    app = ArenaApp(
        ArenaSession(
            ArenaConfig(
                seat1_name="Hero",
                seat2_name="HarveySpecter",
                seat1_agent="human",
                seat2_agent="harvey",
                starting_stack=200,
                small_blind=1,
                big_blind=2,
                seed=77,
            )
        )
    )

    # 1. Agent catalog includes harvey and archetype agents
    code, _, agents_payload = app.handle_request("GET", "/api/agents")
    assert code == 200
    assert isinstance(agents_payload, dict)
    agent_ids = {a["id"] for a in agents_payload["agents"] if a["compatible"]}
    assert {
        "harvey",
        "calling_station",
        "tight_passive",
        "loose_aggressive",
        "maniac",
        "nit",
    }.issubset(agent_ids)

    # 2. Play 3 hands between Human and HarveySpecter
    for _ in range(3):
        code, _, state = app.handle_request("POST", "/api/start", {})
        assert code == 200
        assert isinstance(state, dict)
        while not state["hand_complete"]:
            hc = state["human_controls"]
            assert hc["is_human_turn"] is True
            act = "call" if hc["can_call"] else "check"
            code, _, state = app.handle_request(
                "POST", "/api/human/action", {"action_type": act}
            )
            assert code == 200

    # 3. Verify Policy Inspector captured HarveySpecter decision telemetry
    code, _, dec_payload = app.handle_request("GET", "/api/decisions")
    assert code == 200
    assert isinstance(dec_payload, dict)
    harvey_decisions = [
        d for d in dec_payload["decisions"] if d["agent_id"] == "harvey"
    ]
    assert len(harvey_decisions) > 0
    latest_t = harvey_decisions[-1]["telemetry"]
    assert latest_t["agent_type"] == "HarveySpecterAgent"
    assert "mccfr_probabilities" in latest_t
    assert "exploit_probabilities" in latest_t
    assert "final_probabilities" in latest_t
    assert "lambda_value" in latest_t
    assert "explanation" in latest_t
    assert "opponent_profile" in latest_t
    assert "archetype_classification" in latest_t
    assert "bayesian_range" in latest_t

    # 4. Play a hand where Hero folds preflop and verify hidden-card privacy
    code, _, state = app.handle_request("POST", "/api/start", {})
    assert code == 200
    if not state["hand_complete"] and state["human_controls"]["is_human_turn"]:
        code, _, state = app.handle_request(
            "POST", "/api/human/action", {"action_type": "fold"}
        )
        assert code == 200
    hero_prof = app.session.profile_store.get_or_create("Hero")
    folded_hand_rec = hero_prof.collector.get_hand("Hero", state["hand_number"])
    assert folded_hand_rec is not None
    assert folded_hand_rec.reached_showdown is False
    assert folded_hand_rec.revealed_hole_cards is None

    # 5. Test /api/opponent-profile save, reset, and load routes
    save_file = tmp_path / "hero_profile.json"
    code, _, _saved_resp = app.handle_request(
        "POST",
        "/api/opponent-profile/save",
        {"opponent_id": "Hero", "path": str(save_file)},
    )
    assert code == 200
    assert save_file.exists()

    code, _, reset_resp = app.handle_request(
        "POST", "/api/opponent-profile/reset", {"opponent_id": "Hero"}
    )
    assert code == 200
    assert (
        reset_resp["opponent_intelligence"]["opponent_profile"]["hands_observed"] == 0
    )

    code, _, load_resp = app.handle_request(
        "POST",
        "/api/opponent-profile/load",
        {"opponent_id": "Hero", "path": str(save_file)},
    )
    assert code == 200
    assert (
        load_resp["state"]["opponent_intelligence"]["opponent_profile"][
            "hands_observed"
        ]
        >= 3
    )


def test_benchmark_runner_smoke():
    cfg = HarveyBenchmarkConfig(
        seeds=(11, 22),
        hands_per_seed=10,
        mccfr_iterations=3,
        equity_simulations=16,
        opponent_equity_simulations=16,
    )
    res = run_harvey_specter_benchmark(
        cfg,
        opponents=(("CallingStation", "calling_station"),),
        variants=(
            "BaselineMCCFR",
            "HarveySpecter_Disabled",
            "HarveySpecter_Adaptive",
        ),
    )
    assert len(res["summary"]) == 3
    for row in res["summary"]:
        assert row["total_hands"] == 20
        assert (
            row["ci95_low_bb_per_100"]
            <= row["bb_per_100"]
            <= row["ci95_high_bb_per_100"]
        )
