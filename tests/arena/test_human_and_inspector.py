"""Comprehensive tests for PokerLab Phase 9: Human vs Agent & Policy Inspector."""

from __future__ import annotations

import copy
import math
import random

import pytest

from pokerlab.agents import (
    EquityAgent,
    EquityAgentConfig,
    HeuristicAgent,
    RandomAgent,
    observe,
)
from pokerlab.algorithms.monte_carlo import estimate_equity
from pokerlab.arena import (
    ActionBusyError,
    ArenaApp,
    ArenaConfig,
    ArenaSession,
    compute_shannon_entropy,
    create_agent,
    inspect_and_execute_agent,
)
from pokerlab.engine import ActionType, Deck, GameState


def test_human_start_turn_and_out_of_turn_protection() -> None:
    # Seat 1 is Human (Button on Hand #1 -> acts first preflop).
    session = ArenaSession(
        ArenaConfig(
            seat1_name="Hero",
            seat2_name="Bot",
            seat1_agent="human",
            seat2_agent="heuristic",
            starting_stack=200,
            seed=42,
        )
    )
    state = session.start_hand()
    assert state["mode"] == "human_vs_agent"
    assert state["has_human_player"] is True
    assert state["current_player"] == "Hero"
    assert state["human_controls"]["is_human_turn"] is True
    assert state["human_controls"]["acting_player"] == "Hero"
    assert state["human_controls"]["to_call"] == 1
    assert state["human_controls"]["min_raise_to"] == 4
    assert state["human_controls"]["max_raise_to"] == 200

    # Calling step() when it is the human's turn is rejected.
    with pytest.raises(ValueError, match="human-controlled"):
        session.step()

    # Submitting an action claiming to be "Bot" when it is "Hero"'s turn is rejected.
    with pytest.raises(ValueError, match="Cannot act out of turn"):
        session.submit_human_action("call", player="Bot")

    # Seat 2 is Human (Big Blind on Hand #1 -> Seat 1 Bot acts first preflop).
    # With auto_advance_ai=False we can verify that human cannot act on Bot's turn.
    s2_session = ArenaSession(
        ArenaConfig(
            seat1_name="Bot",
            seat2_name="Hero",
            seat1_agent="heuristic",
            seat2_agent="human",
            starting_stack=200,
            seed=42,
        )
    )
    paused_start = s2_session.start_hand(auto_advance_ai=False)
    assert paused_start["current_player"] == "Bot"
    assert paused_start["human_controls"]["is_human_turn"] is False
    with pytest.raises(ValueError, match="Cannot submit human action on AI turn"):
        s2_session.submit_human_action("call")


def test_human_all_action_types_and_illegal_action_rejection() -> None:
    session = ArenaSession(
        ArenaConfig(
            seat1_name="Hero",
            seat2_name="Bot",
            seat1_agent="human",
            seat2_agent="heuristic",
            starting_stack=100,
            small_blind=1,
            big_blind=2,
            seed=101,
        )
    )
    state = session.start_hand()
    assert state["current_player"] == "Hero"
    assert state["human_controls"]["to_call"] == 1

    # Illegal actions preflop when facing a bet (to_call == 1, current_bet == 2):
    with pytest.raises(ValueError, match="Cannot check while facing a bet"):
        session.submit_human_action("check")
    with pytest.raises(
        ValueError, match="Bet is only legal when no bet is outstanding"
    ):
        session.submit_human_action("bet", 6)
    with pytest.raises(ValueError, match="Raise target must be between 4 and 100"):
        session.submit_human_action("raise", 3)
    with pytest.raises(ValueError, match="Raise target must be between 4 and 100"):
        session.submit_human_action("raise", 105)
    with pytest.raises(ValueError, match="Unsupported action_type"):
        session.submit_human_action("invalid_action")
    with pytest.raises(ValueError, match="call action does not accept a custom amount"):
        session.submit_human_action("call", 10)

    # Legal CALL preflop (auto_advance=False so we can inspect step-by-step):
    after_call = session.submit_human_action("call", auto_advance=False)
    assert after_call["current_player"] == "Bot"

    # Let Bot act on BB; if flop is reached and Bot checks OOP, Hero is on flop with to_call == 0
    after_bot = session.step()
    if not after_bot["hand_complete"] and after_bot["current_player"] == "Hero":
        hc = after_bot["human_controls"]
        if hc["to_call"] == 0:
            # Facing no bet: calling and raising are illegal; checking and betting are legal!
            with pytest.raises(ValueError, match="Nothing to call"):
                session.submit_human_action("call")
            with pytest.raises(ValueError, match="Raise is only legal when facing"):
                session.submit_human_action("raise", 8)
            with pytest.raises(ValueError, match="Bet amount must be between"):
                session.submit_human_action("bet", 1)

            # Legal BET on flop
            after_bet = session.submit_human_action("bet", hc["min_bet"])
            assert after_bet["hand_number"] == 1

    # Start a new hand and test legal RAISE, then FOLD or ALL_IN
    state2 = session.start_hand()
    if not state2["hand_complete"]:
        hc2 = state2["human_controls"]
        assert hc2["is_human_turn"] is True
        if hc2["can_raise"]:
            state2 = session.submit_human_action("raise", hc2["min_raise_to"])
        if not state2["hand_complete"] and state2["human_controls"]["is_human_turn"]:
            if state2["human_controls"]["can_check"]:
                state2 = session.submit_human_action("check")
            else:
                state2 = session.submit_human_action("all_in")

    # Start Hand #3 and test legal FOLD
    state3 = session.start_hand()
    if not state3["hand_complete"] and state3["human_controls"]["is_human_turn"]:
        folded_state = session.submit_human_action("fold")
        assert folded_state["hand_complete"] is True
        assert folded_state["winners"].get("Bot", 0) > 0


def test_opponent_auto_response_and_control_return() -> None:
    # Human in Seat 2 (Big Blind on Hand #1):
    # Bot in Seat 1 (Button) acts first preflop automatically when hand starts!
    session = ArenaSession(
        ArenaConfig(
            seat1_name="Bot",
            seat2_name="Hero",
            seat1_agent="random",
            seat2_agent="human",
            starting_stack=100,
            seed=17,
        )
    )
    state = session.start_hand()
    # Either Bot folded preflop (hand_complete) or Bot called/raised and control is now on Hero
    assert state["hand_complete"] or state["current_player"] == "Hero"
    assert len(state["decisions"]) >= 1
    assert state["decisions"][0]["player"] == "Bot"
    assert state["decisions"][0]["is_human"] is False

    # Play hands until we see control return to Hero after an action
    for _ in range(5):
        if state["hand_complete"]:
            state = session.start_hand(auto_rebuy=True)
        while not state["hand_complete"]:
            assert state["current_player"] == "Hero"
            assert state["human_controls"]["is_human_turn"] is True
            hc = state["human_controls"]
            if hc["can_call"]:
                state = session.submit_human_action("call")
            elif hc["can_check"]:
                state = session.submit_human_action("check")
            else:
                state = session.submit_human_action("fold")
        assert state["hand_complete"] is True


def test_privacy_human_cards_visible_and_opponent_hidden_until_showdown() -> None:
    session = ArenaSession(
        ArenaConfig(
            seat1_name="Hero",
            seat2_name="Bot",
            seat1_agent="human",
            seat2_agent="heuristic",
            starting_stack=100,
            reveal_mode="showdown",
            seed=73,
        )
    )
    state = session.start_hand()
    hero_seat = state["seats"][0]
    bot_seat = state["seats"][1]

    # Human sees own cards during active play
    assert hero_seat["cards_revealed"] is True
    assert all(
        not c["hidden"] and c["rank"] is not None for c in hero_seat["hole_cards"]
    )

    # Opponent cards remain strictly hidden during active play
    assert bot_seat["cards_revealed"] is False
    assert all(
        c["hidden"] is True and c["rank"] is None and c["suit"] is None
        for c in bot_seat["hole_cards"]
    )

    # Equity estimation for human uses only public + human cards and never leaks opponent cards
    eq_panel = state["equity_panel"]
    assert eq_panel["human_equity"]["enabled"] is True
    assert eq_panel["human_equity"]["is_exact"] is False
    assert 0.0 <= eq_panel["human_equity"]["equity"] <= 1.0
    assert "hole_cards" not in str(eq_panel)

    # Disabling human equity estimation works
    disabled_eq = session.update_settings(human_equity_enabled=False)
    assert disabled_eq["equity_panel"]["human_equity_enabled"] is False
    assert disabled_eq["equity_panel"]["human_equity"]["enabled"] is False
    assert disabled_eq["equity_panel"]["human_equity"]["equity"] is None

    # Re-enable and fold pre-showdown: opponent cards must remain hidden on fold!
    session.update_settings(human_equity_enabled=True)
    fold_end = session.submit_human_action("fold")
    assert fold_end["hand_complete"] is True
    assert fold_end["seats"][1]["cards_revealed"] is False
    assert all(c["hidden"] is True for c in fold_end["seats"][1]["hole_cards"])

    # Play next hand all the way to showdown: opponent cards are revealed at showdown!
    state2 = session.start_hand()
    while not state2["hand_complete"]:
        hc = state2["human_controls"]
        if hc["can_call"]:
            state2 = session.submit_human_action("call")
        elif hc["can_check"]:
            state2 = session.submit_human_action("check")
        else:
            break
    if state2["hand_complete"] and any(
        e["type"] == "showdown" for e in state2["action_log"]
    ):
        assert state2["seats"][0]["cards_revealed"] is True
        assert state2["seats"][1]["cards_revealed"] is True
        assert all(not c["hidden"] for c in state2["seats"][1]["hole_cards"])


def test_policy_inspector_mccfr_equity_heuristic_and_random_accuracy() -> None:
    game = GameState(
        {"P1": 100, "P2": 100},
        small_blind=1,
        big_blind=2,
        button="P1",
        deck=Deck(seed=55),
    )
    obs = observe(game, "P1", ())

    # 1. MCCFR (AbstractHUNLAgent) telemetry verification
    mccfr_agent = create_agent("mccfr", seed=12, starting_stack=100, big_blind=2)
    expected_dist = mccfr_agent.action_probabilities(obs)
    chosen_mccfr, mccfr_tel = inspect_and_execute_agent(mccfr_agent, "mccfr", obs)
    assert chosen_mccfr in obs.legal_actions
    assert mccfr_tel["lookup_tier"] in {"exact", "generalized", "fallback"}
    assert "matched_abstraction_key" in mccfr_tel
    assert "information_set_key" in mccfr_tel
    recorded_probs = {
        (row["action_type"], row["amount"]): row["probability"]
        for row in mccfr_tel["action_probabilities"]
    }
    for act, prob in expected_dist.items():
        key = (ActionType(act.type).value, act.amount)
        assert math.isclose(recorded_probs[key], prob, abs_tol=1e-5)

    expected_entropy = compute_shannon_entropy(list(expected_dist.values()))
    assert mccfr_tel["strategy_entropy"] == expected_entropy
    assert any(
        row["is_chosen"]
        and row["action_type"] == ActionType(chosen_mccfr.type).value
        and row["amount"] == chosen_mccfr.amount
        for row in mccfr_tel["action_probabilities"]
    )
    assert len(mccfr_tel["unavailable_fields"]) >= 1

    # 2. EquityAgent telemetry verification (exact seed & rollout match)
    eq_agent_1 = EquityAgent(EquityAgentConfig(simulations=60), seed=999)
    eq_agent_2 = EquityAgent(EquityAgentConfig(simulations=60), seed=999)
    # Capture what eq_agent_2 would compute on its own
    shadow_rng = random.Random(999)
    expected_seed = shadow_rng.randrange(2**32)
    expected_est = estimate_equity(
        obs.hole_cards,
        obs.community_cards,
        simulations=60,
        seed=expected_seed,
        method="monte_carlo",
    )
    direct_action = eq_agent_2.act(obs)

    inspected_action, eq_tel = inspect_and_execute_agent(eq_agent_1, "equity", obs)
    assert inspected_action == direct_action
    assert eq_tel["equity_estimate"]["rollout_seed"] == expected_seed
    assert eq_tel["equity_estimate"]["simulations"] == 60
    assert math.isclose(
        eq_tel["equity_estimate"]["equity"], expected_est.equity, abs_tol=1e-6
    )
    assert eq_tel["equity_estimate"]["is_exact"] is False

    # 3. HeuristicAgent telemetry verification
    heur_agent = HeuristicAgent()
    raw_s, has_draw = heur_agent.strength(obs)
    chosen_heur, heur_tel = inspect_and_execute_agent(heur_agent, "heuristic", obs)
    assert chosen_heur == heur_agent.act(obs)
    assert math.isclose(
        heur_tel["heuristic_signals"]["raw_strength"], raw_s, abs_tol=1e-6
    )
    assert heur_tel["heuristic_signals"]["has_draw"] is has_draw
    assert len(heur_tel["unavailable_fields"]) >= 1

    # 4. RandomAgent telemetry verification
    rand_agent = RandomAgent(seed=321)
    chosen_rand, rand_tel = inspect_and_execute_agent(rand_agent, "random", obs)
    assert chosen_rand in obs.legal_actions
    assert rand_tel["sampling_info"]["legal_action_count"] == len(obs.legal_actions)
    assert math.isclose(
        rand_tel["sampling_info"]["uniform_probability"],
        1.0 / len(obs.legal_actions),
        abs_tol=1e-6,
    )


def test_decision_timeline_immutability_across_subsequent_decisions() -> None:
    session = ArenaSession(
        ArenaConfig(
            seat1_name="Hero",
            seat2_name="Bot",
            seat1_agent="human",
            seat2_agent="mccfr",
            starting_stack=100,
            seed=88,
        )
    )
    state = session.start_hand()
    state = session.submit_human_action("call")
    first_Batch = copy.deepcopy(state["decisions"])
    assert len(first_Batch) >= 2  # Hero's call + Bot's response

    # Advance further in the hand or into the next hand
    while not state["hand_complete"]:
        hc = state["human_controls"]
        if hc["can_check"]:
            state = session.submit_human_action("check")
        elif hc["can_call"]:
            state = session.submit_human_action("call")
        else:
            state = session.submit_human_action("fold")

    later_decisions = state["decisions"]
    assert len(later_decisions) >= len(first_Batch)
    # Verify historical snapshots remain 100% unchanged after subsequent decisions
    for idx, original_snapshot in enumerate(first_Batch):
        assert later_decisions[idx] == original_snapshot


def test_stale_duplicate_concurrent_and_match_over_robustness() -> None:
    app = ArenaApp(
        ArenaSession(
            ArenaConfig(
                seat1_name="Hero",
                seat2_name="Bot",
                seat1_agent="human",
                seat2_agent="heuristic",
                starting_stack=20,
                small_blind=1,
                big_blind=2,
                seed=50,
            )
        )
    )
    status, _, state = app.handle_request("POST", "/api/human/start", {})
    assert status == 200
    assert isinstance(state, dict)
    assert state["hand_number"] == 1
    step_before = state["step_count"]

    # Submit valid human action with expected_hand_number and expected_step_count
    status, _, after_act = app.handle_request(
        "POST",
        "/api/human/action",
        {
            "action_type": "call",
            "expected_hand_number": 1,
            "expected_step_count": step_before,
        },
    )
    assert status == 200
    assert isinstance(after_act, dict)

    # Duplicate submission with old expected_step_count -> 400 Bad Request
    status, _, dup_err = app.handle_request(
        "POST",
        "/api/human/action",
        {
            "action_type": "check",
            "expected_hand_number": 1,
            "expected_step_count": step_before,
        },
    )
    assert status == 400
    assert isinstance(dup_err, dict)
    assert "Stale or duplicate" in dup_err["error"]

    # Stale submission with wrong expected_hand_number -> 400 Bad Request
    status, _, stale_err = app.handle_request(
        "POST",
        "/api/human/action",
        {
            "action_type": "check",
            "expected_hand_number": 999,
        },
    )
    assert status == 400
    assert isinstance(stale_err, dict)
    assert "Stale hand_number" in stale_err["error"]

    # Concurrent request guard -> 409 Conflict
    assert app.session._lock.acquire(blocking=False) is True
    try:
        with pytest.raises(ActionBusyError):
            app.session.submit_human_action("fold")
        status, _, conflict = app.handle_request(
            "POST", "/api/human/action", {"action_type": "fold"}
        )
        assert status == 409
        assert isinstance(conflict, dict)
    finally:
        app.session._lock.release()

    # Query /api/human/decisions and /api/human/state
    status, _, dec_resp = app.handle_request(
        "GET", "/api/human/decisions?hand_number=1"
    )
    assert status == 200
    assert isinstance(dec_resp, dict)
    assert len(dec_resp["decisions"]) >= 1

    # Reset via /api/human/reset clears statistics and decisions
    status, _, reset_resp = app.handle_request(
        "POST",
        "/api/human/reset",
        {"human_seat": 2, "opponent_agent": "equity", "starting_stack": 10},
    )
    assert status == 200
    assert isinstance(reset_resp, dict)
    assert reset_resp["decisions"] == []
    assert reset_resp["match_statistics"]["hands_played"] == 0
    assert reset_resp["seats"][1]["is_human"] is True
    assert reset_resp["seats"][0]["agent_id"] == "equity"


def test_human_all_in_runout_side_pot_conservation_and_match_over() -> None:
    session = ArenaSession(
        ArenaConfig(
            seat1_name="Hero",
            seat2_name="Bot",
            seat1_agent="human",
            seat2_agent="heuristic",
            starting_stack=10,
            small_blind=1,
            big_blind=2,
            seed=12,
        )
    )
    # Play hands with all-in shoves until a player reaches 0 chips (match_over)
    for _ in range(15):
        state = session.start_hand(auto_rebuy=False)
        while not state["hand_complete"]:
            hc = state["human_controls"]
            assert hc["is_human_turn"] is True
            if hc["can_all_in"]:
                state = session.submit_human_action("all_in")
            elif hc["can_call"]:
                state = session.submit_human_action("call")
            else:
                state = session.submit_human_action("check")

        # Chip conservation must hold after every hand
        assert sum(s["stack"] for s in state["seats"]) == 20
        if state["status"] == "match_over":
            break

    final_state = session.serialize_state()
    assert final_state["status"] == "match_over"
    assert any(s["stack"] == 0 for s in final_state["seats"])
    assert final_state["match_statistics"]["hands_played"] >= 1
    assert sum(final_state["match_statistics"]["net_chips"].values()) == 0

    # Starting a new hand without auto_rebuy when match_over raises ValueError
    with pytest.raises(ValueError, match="One player has 0 chips"):
        session.start_hand(auto_rebuy=False)


def test_seat2_human_owes_25_with_1000_stack_regression() -> None:
    """Regression test for Seat 2 human owing 25 chips with 1000 chips available.

    Verifies:
    1. GET /api/human/state returns exact human_controls flags:
       can_fold=True, can_check=False, can_call=True (25),
       can_raise=True (100..1000), can_all_in=True (1000).
    2. Frontend index.html resets isRequestInFlight=false BEFORE renderState(state)
       in startHand/stepAction/resetSession/submitHumanAction so Fold, Call 25,
       Raise To 100, and All-In (1000) are ENABLED while Check is DISABLED.
    3. Backend accepts legal action submission from Seat 2 in this exact state.
    """
    app = ArenaApp(
        ArenaSession(
            ArenaConfig(
                seat1_name="Seat 1",
                seat2_name="Seat 2",
                seat1_agent="heuristic",
                seat2_agent="human",
                starting_stack=1000,
                small_blind=25,
                big_blind=50,
                seed=42,
            )
        )
    )
    # Put Seat 2 on the button for the upcoming hand so Seat 2 posts SB=25,
    # Seat 1 posts BB=50, and Seat 2 acts first owing 25 with 975 behind (1000 total).
    app.session.hand_number = 1
    app.session.button = "Seat 1"
    status, _, started_state = app.handle_request("POST", "/api/start", {})
    assert status == 200
    assert isinstance(started_state, dict)

    status, _, state = app.handle_request("GET", "/api/human/state")
    assert status == 200
    assert isinstance(state, dict)

    hc = state["human_controls"]
    assert hc["is_human_turn"] is True
    assert hc["acting_player"] == "Seat 2"
    assert hc["to_call"] == 25
    assert hc["call_amount"] == 25
    assert hc["min_amount"] == 100
    assert hc["max_amount"] == 1000
    assert hc["default_amount"] == 100
    assert hc["all_in_to"] == 1000

    # Verify exact backend capability flags
    assert hc["can_fold"] is True
    assert hc["can_check"] is False
    assert hc["can_call"] is True
    assert hc["can_raise"] is True
    assert hc["can_all_in"] is True

    # Verify frontend JS in index.html clears isRequestInFlight = false BEFORE renderState(state)
    status, _, html = app.handle_request("GET", "/")
    assert status == 200
    html_str = str(html)
    for fn_name in (
        "async function startHand()",
        "async function submitHumanAction(",
        "async function stepAction()",
        "async function pauseGame()",
        "async function resumeGame()",
        "async function resetSession()",
    ):
        fn_start = html_str.index(fn_name)
        fn_block = html_str[fn_start : fn_start + 950]
        reset_flag_pos = fn_block.index("isRequestInFlight = false;")
        render_state_pos = fn_block.index("renderState(state);")
        assert reset_flag_pos < render_state_pos, (
            f"{fn_name} must set isRequestInFlight = false before calling renderState(state)"
        )

    # Evaluate the exact frontend button disabled conditions after startHand completes
    is_request_in_flight = False
    button_disabled_state = {
        "fold": (not hc["can_fold"]) or is_request_in_flight,
        "check": (not hc["can_check"]) or is_request_in_flight,
        "call_25": (not hc["can_call"]) or is_request_in_flight,
        "raise_to_100": (not (hc["can_bet"] or hc["can_raise"]))
        or is_request_in_flight,
        "all_in_1000": (not hc["can_all_in"]) or is_request_in_flight,
    }
    assert button_disabled_state == {
        "fold": False,  # ENABLED
        "check": True,  # DISABLED
        "call_25": False,  # ENABLED
        "raise_to_100": False,  # ENABLED
        "all_in_1000": False,  # ENABLED
    }

    # Verify Seat 2 can submit Raise To 100 via POST /api/human/action
    status, _, after_raise = app.handle_request(
        "POST",
        "/api/human/action",
        {
            "action_type": "raise",
            "amount": 100,
            "expected_hand_number": state["hand_number"],
            "expected_step_count": state["step_count"],
        },
    )
    assert status == 200
    assert isinstance(after_raise, dict)
    assert any(
        d["player"] == "Seat 2"
        and d["chosen_action"]["action_type"] == "raise"
        and d["chosen_action"]["street_total"] == 100
        for d in after_raise["decisions"]
    )
