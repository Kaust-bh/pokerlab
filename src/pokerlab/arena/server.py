"""Lightweight HTTP server and JSON API dispatcher for PokerLab Arena."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pokerlab.arena.agents import list_agent_catalog
from pokerlab.arena.session import ActionBusyError, ArenaConfig, ArenaSession

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _parse_json_payload(body: dict[str, Any] | bytes | str | None) -> dict[str, Any]:
    if body is None:
        return {}
    if isinstance(body, dict):
        return dict(body)
    if isinstance(body, bytes):
        text = body.decode("utf-8").strip()
    elif isinstance(body, str):
        text = body.strip()
    else:
        raise TypeError("Request body must be a JSON object")
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise TypeError("JSON request payload must be an object")
    return parsed


def _resolve_seats_from_payload(
    payload: dict[str, Any],
    fallback: ArenaConfig,
    *,
    default_human_mode: bool = False,
) -> tuple[str, str, str, str]:
    seat1_name = str(payload.get("seat1_name", fallback.seat1_name))
    seat2_name = str(payload.get("seat2_name", fallback.seat2_name))
    seat1_agent = str(payload.get("seat1_agent", fallback.seat1_agent))
    seat2_agent = str(payload.get("seat2_agent", fallback.seat2_agent))

    if "human_seat" in payload or "opponent_agent" in payload:
        raw_seat = payload.get("human_seat", 1)
        if str(raw_seat).strip().lower() in {"1", "seat1", "seat_1"}:
            human_idx = 1
        elif str(raw_seat).strip().lower() in {"2", "seat2", "seat_2"}:
            human_idx = 2
        else:
            raise ValueError("human_seat must be 1 or 2")
        opp_agent = str(
            payload.get(
                "opponent_agent",
                fallback.seat2_agent if human_idx == 1 else fallback.seat1_agent,
            )
        )
        if opp_agent.strip().lower() == "human":
            opp_agent = "heuristic"
        if human_idx == 1:
            seat1_agent = "human"
            seat2_agent = opp_agent
            if "human_name" in payload:
                seat1_name = str(payload["human_name"])
            if "opponent_name" in payload:
                seat2_name = str(payload["opponent_name"])
        else:
            seat2_agent = "human"
            seat1_agent = opp_agent
            if "human_name" in payload:
                seat2_name = str(payload["human_name"])
            if "opponent_name" in payload:
                seat1_name = str(payload["opponent_name"])
    elif (
        default_human_mode
        and seat1_agent.strip().lower() != "human"
        and seat2_agent.strip().lower() != "human"
    ):
        seat1_agent = "human"

    return seat1_name, seat2_name, seat1_agent, seat2_agent


def _build_arena_config(
    payload: dict[str, Any],
    fallback: ArenaConfig,
    *,
    default_human_mode: bool = False,
) -> ArenaConfig:
    s1_name, s2_name, s1_agent, s2_agent = _resolve_seats_from_payload(
        payload, fallback, default_human_mode=default_human_mode
    )
    return ArenaConfig(
        seat1_name=s1_name,
        seat2_name=s2_name,
        seat1_agent=s1_agent,
        seat2_agent=s2_agent,
        starting_stack=int(payload.get("starting_stack", fallback.starting_stack)),
        small_blind=int(payload.get("small_blind", fallback.small_blind)),
        big_blind=int(payload.get("big_blind", fallback.big_blind)),
        seed=int(payload.get("seed", fallback.seed)),
        equity_simulations=int(
            payload.get("equity_simulations", fallback.equity_simulations)
        ),
        action_profile=str(payload.get("action_profile", fallback.action_profile)),
        playback_speed_ms=int(
            payload.get("playback_speed_ms", fallback.playback_speed_ms)
        ),
        reveal_mode=str(payload.get("reveal_mode", fallback.reveal_mode)),
        human_equity_enabled=bool(
            payload.get("human_equity_enabled", fallback.human_equity_enabled)
        ),
        human_equity_simulations=int(
            payload.get("human_equity_simulations", fallback.human_equity_simulations)
        ),
    )


_CONFIG_KEYS = frozenset(
    {
        "seat1_name",
        "seat2_name",
        "seat1_agent",
        "seat2_agent",
        "human_seat",
        "opponent_agent",
        "human_name",
        "opponent_name",
        "starting_stack",
        "small_blind",
        "big_blind",
        "seed",
        "equity_simulations",
        "action_profile",
        "playback_speed_ms",
        "reveal_mode",
        "human_equity_enabled",
        "human_equity_simulations",
    }
)


class ArenaApp:
    """Transport-independent request dispatcher wrapping an `ArenaSession`."""

    def __init__(self, session: ArenaSession | None = None) -> None:
        self.session: ArenaSession = session or ArenaSession()
        self._index_html: str = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    def handle_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | bytes | str | None = None,
    ) -> tuple[int, str, dict[str, Any] | str]:
        """Dispatch an HTTP method + path and return `(status_code, content_type, payload)`."""
        parsed_url = urlparse(path)
        route = parsed_url.path.rstrip("/") or "/"
        query = parse_qs(parsed_url.query)
        verb = method.upper()

        try:
            if verb == "GET" and route in {"/", "/index.html"}:
                return (
                    int(HTTPStatus.OK),
                    "text/html; charset=utf-8",
                    self._index_html,
                )

            if verb == "GET" and route == "/api/agents":
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    {"agents": list_agent_catalog()},
                )

            if verb == "GET" and route in {"/api/state", "/api/human/state"}:
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.serialize_state(),
                )

            if verb == "GET" and route in {"/api/decisions", "/api/human/decisions"}:
                hand_num_raw = query.get("hand_number", [None])[0]
                dec_id_raw = query.get("decision_id", [None])[0]
                decisions = self.session.serialize_decisions(
                    hand_number=int(hand_num_raw) if hand_num_raw is not None else None,
                    decision_id=int(dec_id_raw) if dec_id_raw is not None else None,
                )
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    {
                        "session_id": self.session.session_id,
                        "hand_number": self.session.hand_number,
                        "decisions": decisions,
                    },
                )

            if verb == "GET" and route == "/api/opponent-profile":
                opp_id = query.get("opponent_id", [None])[0]
                export_raw = query.get("export", ["false"])[0]
                if str(export_raw).strip().lower() in {"1", "true", "yes"}:
                    return (
                        int(HTTPStatus.OK),
                        "application/json; charset=utf-8",
                        self.session.save_opponent_profile(opponent_id=opp_id),
                    )
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.serialize_opponent_intelligence(target_player=opp_id),
                )

            if verb == "POST" and route == "/api/opponent-profile/reset":
                payload = _parse_json_payload(body)
                opp_id = payload.get("opponent_id")
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.reset_opponent_profile(
                        opponent_id=str(opp_id) if opp_id is not None else None
                    ),
                )

            if verb == "POST" and route == "/api/opponent-profile/save":
                payload = _parse_json_payload(body)
                opp_id = payload.get("opponent_id")
                save_path = payload.get("path")
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.save_opponent_profile(
                        opponent_id=str(opp_id) if opp_id is not None else None,
                        path=str(save_path) if save_path is not None else None,
                    ),
                )

            if verb == "POST" and route == "/api/opponent-profile/load":
                payload = _parse_json_payload(body)
                opp_id = payload.get("opponent_id")
                load_path = payload.get("path")
                prof_payload = payload.get("profile")
                strict = bool(payload.get("strict", False))
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.load_opponent_profile(
                        opponent_id=str(opp_id) if opp_id is not None else None,
                        path=str(load_path) if load_path is not None else None,
                        payload=prof_payload,
                        strict=strict,
                    ),
                )

            if verb == "POST" and route in {
                "/api/reset",
                "/api/session",
                "/api/human/reset",
            }:
                payload = _parse_json_payload(body)
                is_human_route = route == "/api/human/reset"
                cfg = (
                    _build_arena_config(
                        payload,
                        self.session.config,
                        default_human_mode=is_human_route,
                    )
                    if (payload or is_human_route)
                    else None
                )
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.reset(cfg),
                )

            if verb == "POST" and route in {"/api/start", "/api/new-hand"}:
                payload = _parse_json_payload(body)
                auto_rebuy = bool(payload.get("auto_rebuy", True))
                auto_advance_ai = payload.get("auto_advance_ai")
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.start_hand(
                        auto_rebuy=auto_rebuy,
                        auto_advance_ai=(
                            bool(auto_advance_ai)
                            if auto_advance_ai is not None
                            else None
                        ),
                    ),
                )

            if verb == "POST" and route == "/api/human/start":
                payload = _parse_json_payload(body)
                has_setup_fields = bool(_CONFIG_KEYS & payload.keys())
                if (
                    has_setup_fields
                    or bool(payload.get("reset", False))
                    or not self.session.has_human_player
                ):
                    cfg = _build_arena_config(
                        payload,
                        self.session.config,
                        default_human_mode=True,
                    )
                    self.session.reset(cfg)
                auto_rebuy = bool(payload.get("auto_rebuy", True))
                auto_advance_ai = bool(payload.get("auto_advance_ai", True))
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.start_hand(
                        auto_rebuy=auto_rebuy,
                        auto_advance_ai=auto_advance_ai,
                    ),
                )

            if verb == "POST" and route == "/api/human/next-hand":
                payload = _parse_json_payload(body)
                auto_rebuy = bool(payload.get("auto_rebuy", False))
                auto_advance_ai = bool(payload.get("auto_advance_ai", True))
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.start_hand(
                        auto_rebuy=auto_rebuy,
                        auto_advance_ai=auto_advance_ai,
                    ),
                )

            if verb == "POST" and route in {"/api/human/action", "/api/action"}:
                payload = _parse_json_payload(body)
                raw_action_type = (
                    payload.get("action_type")
                    if "action_type" in payload
                    else payload.get("action", payload.get("type"))
                )
                if raw_action_type is None:
                    raise ValueError("Missing required field 'action_type'")
                raw_amount = payload.get("amount")
                parsed_amount: int | None = None
                if raw_amount is not None:
                    if isinstance(raw_amount, bool):
                        raise TypeError("amount must be an integer")
                    parsed_amount = int(raw_amount)
                expected_hand = payload.get(
                    "expected_hand_number", payload.get("hand_number")
                )
                expected_step = payload.get(
                    "expected_step_count", payload.get("step_count")
                )
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.submit_human_action(
                        action_type=str(raw_action_type),
                        amount=parsed_amount,
                        player=(
                            str(payload["player"])
                            if payload.get("player") is not None
                            else None
                        ),
                        expected_hand_number=(
                            int(expected_hand) if expected_hand is not None else None
                        ),
                        expected_step_count=(
                            int(expected_step) if expected_step is not None else None
                        ),
                        auto_advance=bool(payload.get("auto_advance", True)),
                    ),
                )

            if verb == "POST" and route == "/api/step":
                _parse_json_payload(body)
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.step(),
                )

            if verb == "POST" and route == "/api/pause":
                _parse_json_payload(body)
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.pause(),
                )

            if verb == "POST" and route == "/api/resume":
                _parse_json_payload(body)
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.resume(),
                )

            if verb == "POST" and route == "/api/settings":
                payload = _parse_json_payload(body)
                return (
                    int(HTTPStatus.OK),
                    "application/json; charset=utf-8",
                    self.session.update_settings(
                        playback_speed_ms=(
                            int(payload["playback_speed_ms"])
                            if payload.get("playback_speed_ms") is not None
                            else None
                        ),
                        reveal_mode=(
                            str(payload["reveal_mode"])
                            if payload.get("reveal_mode") is not None
                            else None
                        ),
                        seat1_agent=(
                            str(payload["seat1_agent"])
                            if payload.get("seat1_agent") is not None
                            else None
                        ),
                        seat2_agent=(
                            str(payload["seat2_agent"])
                            if payload.get("seat2_agent") is not None
                            else None
                        ),
                        human_equity_enabled=(
                            bool(payload["human_equity_enabled"])
                            if payload.get("human_equity_enabled") is not None
                            else None
                        ),
                        human_equity_simulations=(
                            int(payload["human_equity_simulations"])
                            if payload.get("human_equity_simulations") is not None
                            else None
                        ),
                    ),
                )

            return (
                int(HTTPStatus.NOT_FOUND),
                "application/json; charset=utf-8",
                {"error": f"Unknown route: {verb} {route}"},
            )
        except ActionBusyError as exc:
            return (
                int(HTTPStatus.CONFLICT),
                "application/json; charset=utf-8",
                {"error": str(exc)},
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return (
                int(HTTPStatus.BAD_REQUEST),
                "application/json; charset=utf-8",
                {"error": str(exc)},
            )


def create_http_server(
    host: str = "127.0.0.1",
    port: int = 8501,
    app: ArenaApp | None = None,
    *,
    bind_and_activate: bool = True,
) -> ThreadingHTTPServer:
    """Create a `ThreadingHTTPServer` bound to `(host, port)` serving `ArenaApp`."""
    arena_app = app or ArenaApp()

    class ArenaRequestHandler(BaseHTTPRequestHandler):
        def _dispatch(self, method: str) -> None:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            raw_body = self.rfile.read(content_length) if content_length > 0 else b""
            status, content_type, payload = arena_app.handle_request(
                method, self.path, raw_body
            )
            if isinstance(payload, str):
                encoded = payload.encode("utf-8")
            else:
                encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer(
        (host, port),
        ArenaRequestHandler,
        bind_and_activate=bind_and_activate,
    )


def run_server(
    host: str = "127.0.0.1",
    port: int = 8501,
    config: ArenaConfig | None = None,
) -> None:
    """Start the PokerLab Arena web server and block until interrupted."""
    app = ArenaApp(ArenaSession(config))
    server = create_http_server(host=host, port=port, app=app)
    bound_host, bound_port = server.server_address[:2]
    print(f"PokerLab Arena listening on http://{bound_host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
