"""Command-line entry point for launching PokerLab Arena."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pokerlab.arena.server import run_server
from pokerlab.arena.session import ArenaConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pokerlab-arena",
        description="Launch PokerLab Arena, the local interactive HUNL poker viewer.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="TCP port to listen on (default: 8501).",
    )
    parser.add_argument(
        "--seat1-agent",
        default="heuristic",
        help="Initial Seat 1 agent id: random, heuristic, equity, or mccfr.",
    )
    parser.add_argument(
        "--seat2-agent",
        default="equity",
        help="Initial Seat 2 agent id: random, heuristic, equity, or mccfr.",
    )
    parser.add_argument(
        "--stack",
        type=int,
        default=200,
        help="Starting chip stack per player (default: 200).",
    )
    parser.add_argument(
        "--small-blind",
        type=int,
        default=1,
        help="Small blind chip amount (default: 1).",
    )
    parser.add_argument(
        "--big-blind",
        type=int,
        default=2,
        help="Big blind chip amount (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed for deck deals and agents (default: 42).",
    )
    parser.add_argument(
        "--reveal-mode",
        default="showdown",
        choices=("showdown", "seat1", "hidden", "all"),
        help="Card visibility mode (default: showdown).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ArenaConfig(
        seat1_agent=args.seat1_agent,
        seat2_agent=args.seat2_agent,
        starting_stack=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        seed=args.seed,
        reveal_mode=args.reveal_mode,
    )
    run_server(host=args.host, port=args.port, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
