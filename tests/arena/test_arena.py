"""Tests for PokerLab Arena session management, agent integration, and HTTP API."""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

import pytest

from pokerlab.agents.base import Agent, Observation
from pokerlab.arena import (
    ActionBusyError,
    ArenaApp,
    ArenaConfig,
    ArenaSession,
    create_agent,
    create_http_server,
    list_agent_catalog,
    serialize_action,
    serialize_card,
)
from pokerlab.engine import Action, ActionType, Card


class _AlwaysCallOrCheckAgent(Agent):
    """Deterministic test helper that never folds so hands reach showdown."""

    def act(self, observation: Observation) -> Action:
        for preferred in (ActionType.CHECK, ActionType.CALL):
            for candidate in observation.legal_actions:
                if candidate.type == preferred:
                    return candidate
        return observation.legal_actions[0]


class _AlwaysShoveAgent(Agent):
    """Deterministic test helper that forces all-in or maximum raise immediately."""

    def act(self, observation: Observation) -> Action:
        for candidate in reversed(observation.legal_actions):
            if candidate.type in {
                ActionType.ALL_IN,
                ActionType.RAISE,
                ActionType.BET,
                ActionType.CALL,
            }:
                return candidate
        return observation.legal_actions[-1]


def test_session_creation_and_determinism() -> None:
    cfg = ArenaConfig(
        seat1_name="Alpha",
        seat2_name="Beta",
        seat1_agent="heuristic",
        seat2_agent="random",
        starting_stack=200,
        small_blind=1,
        big_blind=2,
        seed=12345,
    )
    s1 = ArenaSession(cfg)
    s2 = ArenaSession(cfg)

    state1 = s1.serialize_state()
    assert state1["status"] == "idle"
    assert state1["hand_number"] == 0
    assert state1["pot"] == 0
    assert [seat["name"] for seat in state1["seats"]] == ["Alpha", "Beta"]
    assert [seat["stack"] for seat in state1["seats"]] == [200, 200]

    s1.start_hand()
    s2.start_hand()
    while not s1.serialize_state()["hand_complete"]:
        s1.step()
        s2.step()

    final1 = s1.serialize_state()
    final2 = s2.serialize_state()
    assert final1["winners"] == final2["winners"]
    assert final1["community_cards"] == final2["community_cards"]
    assert [e["text"] for e in final1["action_log"]] == [
        e["text"] for e in final2["action_log"]
    ]


def test_start_and_advance_hand_to_completion() -> None:
    session = ArenaSession(
        ArenaConfig(
            seat1_agent="heuristic",
            seat2_agent="equity",
            starting_stack=100,
            equity_simulations=15,
            seed=77,
        )
    )
    started = session.start_hand()
    assert started["status"] == "running"
    assert started["hand_number"] == 1
    assert started["street"] == "preflop"
    assert started["pot"] == 3
    assert len(started["action_log"]) == 3
    assert started["action_log"][0]["type"] == "hand_start"
    assert started["action_log"][1]["type"] == "blind"
    assert started["action_log"][2]["type"] == "blind"

    steps = 0
    state = started
    while not state["hand_complete"]:
        state = session.step()
        steps += 1
        assert steps <= 30

    assert state["hand_complete"] is True
    assert state["status"] in {"hand_complete", "match_over"}
    assert state["hands_completed"] == 1
    assert sum(seat["stack"] for seat in state["seats"]) == 200
    assert state["action_log"][-1]["type"] == "winner"


def test_pause_and_resume_lifecycle() -> None:
    session = ArenaSession(ArenaConfig(seed=19))
    with pytest.raises(ValueError, match="Cannot pause before starting a hand"):
        session.pause()
    with pytest.raises(ValueError, match="Cannot resume before starting a hand"):
        session.resume()

    session.start_hand()
    paused = session.pause()
    assert paused["status"] == "paused"

    # Stepping one action while paused remains allowed for single-step inspection.
    after_step = session.step()
    assert after_step["step_count"] == 1

    if not after_step["hand_complete"]:
        resumed = session.resume()
        assert resumed["status"] == "running"
        while not session.serialize_state()["hand_complete"]:
            session.step()

    with pytest.raises(ValueError, match="Cannot resume a completed hand"):
        session.resume()


def test_reset_creates_fresh_session_and_reproduces_seed() -> None:
    cfg = ArenaConfig(
        seat1_agent="random",
        seat2_agent="heuristic",
        starting_stack=120,
        seed=909,
    )
    session = ArenaSession(cfg)
    first_session_id = session.session_id
    session.start_hand()
    while not session.serialize_state()["hand_complete"]:
        session.step()
    first_hand_log = [e["text"] for e in session.serialize_state()["action_log"]]

    reset_state = session.reset(cfg)
    assert reset_state["session_id"] != first_session_id
    assert reset_state["status"] == "idle"
    assert reset_state["hand_number"] == 0
    assert reset_state["hands_completed"] == 0
    assert reset_state["action_log"] == []
    assert [seat["stack"] for seat in reset_state["seats"]] == [120, 120]

    session.start_hand()
    while not session.serialize_state()["hand_complete"]:
        session.step()
    replayed_log = [e["text"] for e in session.serialize_state()["action_log"]]
    assert replayed_log == first_hand_log


def test_agent_catalog_and_selection_validation() -> None:
    catalog = list_agent_catalog()
    by_id = {entry["id"]: entry for entry in catalog}
    assert {"random", "heuristic", "equity", "mccfr"}.issubset(by_id.keys())
    assert {"kuhn_cfr", "leduc_cfr", "deep_cfr"}.issubset(by_id.keys())

    for agent_id in ("random", "heuristic", "equity", "mccfr"):
        assert by_id[agent_id]["compatible"] is True
        agent = create_agent(
            agent_id,
            seed=11,
            equity_simulations=12,
            starting_stack=100,
            big_blind=2,
        )
        assert isinstance(agent, Agent)

    # Verify MCCFR agent plays a full hand in ArenaSession cleanly.
    mccfr_session = ArenaSession(
        ArenaConfig(
            seat1_agent="mccfr",
            seat2_agent="heuristic",
            starting_stack=80,
            seed=33,
        )
    )
    mccfr_session.start_hand()
    while not mccfr_session.serialize_state()["hand_complete"]:
        mccfr_session.step()
    assert mccfr_session.serialize_state()["hands_completed"] == 1

    # Verify toy-game and neural solvers report explicit incompatibility reasons.
    for incompatible_id in ("kuhn_cfr", "leduc_cfr", "deep_cfr"):
        assert by_id[incompatible_id]["compatible"] is False
        assert by_id[incompatible_id]["incompatibility_reason"]
        with pytest.raises(ValueError, match="not compatible with HUNL Arena"):
            create_agent(incompatible_id)

    with pytest.raises(ValueError, match="Unknown agent"):
        create_agent("nonexistent_agent")


def test_card_and_action_serialization() -> None:
    ace_spades = Card(14, "spades")
    visible = serialize_card(ace_spades, hidden=False)
    assert visible == {
        "hidden": False,
        "rank": 14,
        "rank_label": "A",
        "suit": "spades",
        "suit_symbol": "♠",
        "color": "dark",
        "code": "A♠",
    }

    hidden = serialize_card(ace_spades, hidden=True)
    assert hidden["hidden"] is True
    assert hidden["rank"] is None
    assert hidden["suit"] is None
    assert hidden["code"] == "??"

    raise_evt = serialize_action(
        "Seat 1",
        Action(ActionType.RAISE, 12),
        "flop",
        paid=8,
        street_total=12,
        is_all_in=False,
    )
    assert raise_evt["type"] == "player_action"
    assert raise_evt["action_type"] == "raise"
    assert raise_evt["paid"] == 8
    assert raise_evt["street_total"] == 12
    assert raise_evt["text"] == "Seat 1 raises to 12"

    all_in_call = serialize_action(
        "Seat 2",
        Action(ActionType.CALL, 50),
        "turn",
        paid=50,
        street_total=80,
        is_all_in=True,
    )
    assert "all-in" in all_in_call["text"]


def test_hidden_hole_card_modes_and_post_hand_reveal() -> None:
    session = ArenaSession(
        ArenaConfig(
            seat1_agent="heuristic",
            seat2_agent="heuristic",
            reveal_mode="showdown",
            seed=55,
        )
    )
    state = session.start_hand()
    # During active hand in "showdown" mode, both seats have hidden cards.
    for seat in state["seats"]:
        assert seat["cards_revealed"] is False
        assert all(c["hidden"] for c in seat["hole_cards"])

    # Switch to "seat1" mode mid-hand: Seat 1 revealed, Seat 2 hidden.
    seat1_mode = session.update_settings(reveal_mode="seat1")
    assert seat1_mode["seats"][0]["cards_revealed"] is True
    assert all(not c["hidden"] for c in seat1_mode["seats"][0]["hole_cards"])
    assert seat1_mode["seats"][1]["cards_revealed"] is False
    assert all(c["hidden"] for c in seat1_mode["seats"][1]["hole_cards"])

    # Switch to "all" debug mode: both revealed.
    all_mode = session.update_settings(reveal_mode="all")
    assert all(seat["cards_revealed"] for seat in all_mode["seats"])
    for seat in all_mode["seats"]:
        assert all(not c["hidden"] for c in seat["hole_cards"])

    # Switch to "hidden" mode: both stay hidden even after hand completes.
    session.update_settings(reveal_mode="hidden")
    while not session.serialize_state()["hand_complete"]:
        session.step()
    hidden_end = session.serialize_state()
    assert all(not seat["cards_revealed"] for seat in hidden_end["seats"])

    # Switch back to "showdown" mode after hand completion: both revealed.
    showdown_end = session.update_settings(reveal_mode="showdown")
    assert all(seat["cards_revealed"] for seat in showdown_end["seats"])


def test_showdown_and_winner_display() -> None:
    session = ArenaSession(ArenaConfig(starting_stack=100, seed=101))
    session.agents = {
        session.players[0]: _AlwaysCallOrCheckAgent(),
        session.players[1]: _AlwaysCallOrCheckAgent(),
    }
    session.start_hand()
    while not session.serialize_state()["hand_complete"]:
        session.step()

    final = session.serialize_state()
    assert final["hand_complete"] is True
    assert len(final["community_cards"]) == 5
    event_types = [e["type"] for e in final["action_log"]]
    assert "showdown" in event_types
    assert event_types[-1] == "winner"
    for seat in final["seats"]:
        assert seat["cards_revealed"] is True
        assert seat["hand_category"] is not None
    assert sum(final["winners"].values()) == final["pot"]
    assert sum(seat["stack"] for seat in final["seats"]) == 200


def test_all_in_multi_street_runout() -> None:
    session = ArenaSession(ArenaConfig(starting_stack=40, seed=202))
    session.agents = {
        session.players[0]: _AlwaysShoveAgent(),
        session.players[1]: _AlwaysCallOrCheckAgent(),
    }
    session.start_hand()
    while not session.serialize_state()["hand_complete"]:
        session.step()

    final = session.serialize_state()
    assert final["hand_complete"] is True
    assert len(final["community_cards"]) == 5
    street_deals = [
        e["street"] for e in final["action_log"] if e["type"] == "street_deal"
    ]
    assert street_deals == ["flop", "turn", "river"]
    assert any(
        e.get("is_all_in") for e in final["action_log"] if e["type"] == "player_action"
    )
    assert sum(seat["stack"] for seat in final["seats"]) == 80


def test_api_routes_concurrency_and_http_server() -> None:
    app = ArenaApp(ArenaSession(ArenaConfig(seed=42)))

    status, ctype, html = app.handle_request("GET", "/")
    assert status == 200
    assert "text/html" in ctype
    assert "PokerLab Arena" in str(html)

    status, _, catalog_payload = app.handle_request("GET", "/api/agents")
    assert status == 200
    assert isinstance(catalog_payload, dict)
    assert len(catalog_payload["agents"]) >= 7

    # Illegal step before starting a hand -> 400 Bad Request
    status, _, err_payload = app.handle_request("POST", "/api/step", "{}")
    assert status == 400
    assert isinstance(err_payload, dict)
    assert "No active hand" in err_payload["error"]

    # Invalid JSON -> 400 Bad Request
    status, _, bad_json = app.handle_request("POST", "/api/reset", "{not-valid-json")
    assert status == 400
    assert isinstance(bad_json, dict)

    # Incompatible agent selection -> 400 Bad Request
    status, _, bad_agent = app.handle_request(
        "POST",
        "/api/reset",
        {"seat1_agent": "kuhn_cfr"},
    )
    assert status == 400
    assert isinstance(bad_agent, dict)
    assert "not compatible" in bad_agent["error"]

    # Unknown route -> 404 Not Found
    status, _, not_found = app.handle_request("GET", "/api/does-not-exist")
    assert status == 404
    assert isinstance(not_found, dict)

    # Concurrent action guard -> 409 Conflict when session lock is already held
    app.handle_request("POST", "/api/start", "{}")
    assert app.session._lock.acquire(blocking=False) is True
    try:
        with pytest.raises(ActionBusyError):
            app.session.step()
        status, _, conflict = app.handle_request("POST", "/api/step", "{}")
        assert status == 409
        assert isinstance(conflict, dict)
        assert "currently processing" in conflict["error"]
    finally:
        app.session._lock.release()

    # HTTP handler wire-protocol check over an in-memory socket pair
    server = create_http_server(
        host="127.0.0.1",
        port=8501,
        app=app,
        bind_and_activate=False,
    )

    def _raw_http_request(raw_request: bytes) -> tuple[int, dict[str, Any]]:
        client_sock, server_sock = socket.socketpair()
        client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
        try:
            client_sock.sendall(raw_request)
            client_sock.shutdown(socket.SHUT_WR)

            def _serve_once() -> None:
                try:
                    server.finish_request(server_sock, ("127.0.0.1", 0))
                finally:
                    server_sock.close()

            worker = threading.Thread(target=_serve_once, daemon=True)
            worker.start()
            response_bytes = b""
            while chunk := client_sock.recv(16384):
                response_bytes += chunk
            worker.join(timeout=2)
        finally:
            client_sock.close()
            server_sock.close()
        header_blob, body_blob = response_bytes.split(b"\r\n\r\n", 1)
        status_line = header_blob.splitlines()[0].decode("utf-8")
        status_code = int(status_line.split()[1])
        return status_code, json.loads(body_blob.decode("utf-8"))

    code, live_state = _raw_http_request(
        b"GET /api/state HTTP/1.1\r\nHost: localhost\r\n\r\n"
    )
    assert code == 200
    assert live_state["hand_number"] == 1

    code, stepped_state = _raw_http_request(
        b"POST /api/step HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n\r\n{}"
    )
    assert code == 200
    assert stepped_state["step_count"] == 1

    code, err_body = _raw_http_request(
        b"GET /api/unknown HTTP/1.1\r\nHost: localhost\r\n\r\n"
    )
    assert code == 404
    assert "Unknown route" in err_body["error"]
