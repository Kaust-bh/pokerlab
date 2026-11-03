"""Stateful game session controller, human gameplay handler, and serializer for PokerLab Arena."""

from __future__ import annotations

import random
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from pokerlab.agents.base import ActionRecord, Agent, Observation, observe
from pokerlab.arena.agents import _CATALOG_BY_ID, create_agent
from pokerlab.arena.inspector import (
    DecisionRecord,
    current_utc_iso,
    estimate_human_public_equity,
    format_action_label,
    inspect_and_execute_agent,
    serialize_engine_action_choice,
)
from pokerlab.engine import Action, ActionType, Card, Deck, GameState, HandCategory
from pokerlab.opponent_model.adaptive_agent import HarveySpecterAgent
from pokerlab.opponent_model.profile_store import OpponentProfileStore

SUIT_SYMBOLS: dict[str, str] = {
    "spades": "♠",
    "hearts": "♥",
    "diamonds": "♦",
    "clubs": "♣",
}

SUIT_COLORS: dict[str, str] = {
    "spades": "dark",
    "hearts": "red",
    "diamonds": "blue",
    "clubs": "green",
}

RANK_LABELS: dict[int, str] = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}

HAND_CATEGORY_NAMES: dict[int, str] = {
    int(HandCategory.HIGH_CARD): "High Card",
    int(HandCategory.ONE_PAIR): "One Pair",
    int(HandCategory.TWO_PAIR): "Two Pair",
    int(HandCategory.THREE_OF_A_KIND): "Three of a Kind",
    int(HandCategory.STRAIGHT): "Straight",
    int(HandCategory.FLUSH): "Flush",
    int(HandCategory.FULL_HOUSE): "Full House",
    int(HandCategory.FOUR_OF_A_KIND): "Four of a Kind",
    int(HandCategory.STRAIGHT_FLUSH): "Straight Flush",
    int(HandCategory.ROYAL_FLUSH): "Royal Flush",
}

ACTION_KEYS: tuple[str, ...] = (
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "all_in",
)


class ActionBusyError(RuntimeError):
    """Raised when a concurrent step or mutation is attempted while busy."""


def serialize_card(card: Card | None, *, hidden: bool = False) -> dict[str, Any]:
    """Serialize a Card into a frontend-ready dictionary without exposing hidden cards."""
    if hidden or card is None:
        return {
            "hidden": True,
            "rank": None,
            "rank_label": "?",
            "suit": None,
            "suit_symbol": "?",
            "color": "hidden",
            "code": "??",
        }
    rank_label = RANK_LABELS[card.rank]
    symbol = SUIT_SYMBOLS[card.suit]
    return {
        "hidden": False,
        "rank": card.rank,
        "rank_label": rank_label,
        "suit": card.suit,
        "suit_symbol": symbol,
        "color": SUIT_COLORS[card.suit],
        "code": f"{rank_label}{symbol}",
    }


def serialize_action(
    player: str,
    action: Action,
    street: str,
    *,
    paid: int = 0,
    street_total: int = 0,
    is_all_in: bool = False,
    decision_id: int | None = None,
) -> dict[str, Any]:
    """Serialize an engine Action and its chip delta into a structured event."""
    kind = ActionType(action.type).value
    if kind == ActionType.FOLD.value:
        text = f"{player} folds"
    elif kind == ActionType.CHECK.value:
        text = f"{player} checks"
    elif kind == ActionType.CALL.value:
        if is_all_in:
            text = f"{player} calls {paid} (all-in for {street_total})"
        else:
            text = f"{player} calls {paid} (to {street_total})"
    elif kind == ActionType.BET.value:
        if is_all_in:
            text = f"{player} bets {street_total} (all-in)"
        else:
            text = f"{player} bets {street_total}"
    elif kind == ActionType.RAISE.value:
        if is_all_in:
            text = f"{player} raises to {street_total} (all-in)"
        else:
            text = f"{player} raises to {street_total}"
    elif kind == ActionType.ALL_IN.value:
        text = f"{player} goes all-in for {street_total} (+{paid})"
    else:
        text = f"{player} performs {kind}"

    payload: dict[str, Any] = {
        "type": "player_action",
        "player": player,
        "action_type": kind,
        "amount": action.amount,
        "paid": paid,
        "street_total": street_total,
        "is_all_in": is_all_in,
        "street": street,
        "text": text,
    }
    if decision_id is not None:
        payload["decision_id"] = decision_id
    return payload


@dataclass(slots=True)
class ArenaConfig:
    """Configuration for a two-player Heads-Up No-Limit Hold'em Arena session."""

    seat1_name: str = "Seat 1"
    seat2_name: str = "Seat 2"
    seat1_agent: str = "heuristic"
    seat2_agent: str = "equity"
    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2
    seed: int = 42
    equity_simulations: int = 100
    action_profile: str = "legacy"
    playback_speed_ms: int = 800
    reveal_mode: str = "showdown"  # "seat1", "hidden", "showdown", or "all"
    human_equity_enabled: bool = True
    human_equity_simulations: int = 250

    def __post_init__(self) -> None:
        self.seat1_name = self.seat1_name.strip() or "Seat 1"
        self.seat2_name = self.seat2_name.strip() or "Seat 2"
        if self.seat1_name == self.seat2_name:
            raise ValueError("seat1_name and seat2_name must be distinct")
        if self.starting_stack <= 0:
            raise ValueError("starting_stack must be positive")
        if self.small_blind <= 0 or self.big_blind < self.small_blind:
            raise ValueError("blinds must be positive and big_blind >= small_blind")
        if self.playback_speed_ms < 50 or self.playback_speed_ms > 10000:
            raise ValueError("playback_speed_ms must be between 50 and 10000")
        if self.reveal_mode not in {"seat1", "hidden", "showdown", "all"}:
            raise ValueError(
                "reveal_mode must be 'seat1', 'hidden', 'showdown', or 'all'"
            )
        if self.human_equity_simulations < 25 or self.human_equity_simulations > 2000:
            raise ValueError("human_equity_simulations must be between 25 and 2000")


class ArenaSession:
    """Manages an interactive Heads-Up No-Limit Hold'em match for PokerLab Arena.

    All poker rules, betting validation, side-pot construction, street runouts,
    and showdown evaluations are delegated strictly to `pokerlab.engine.GameState`.
    """

    def __init__(self, config: ArenaConfig | None = None) -> None:
        self._lock = threading.Lock()
        self.session_id: str = uuid.uuid4().hex[:12]
        self.config: ArenaConfig = config or ArenaConfig()
        self._rng = random.Random(self.config.seed)
        self.players: tuple[str, str] = (
            self.config.seat1_name,
            self.config.seat2_name,
        )
        self.agent_ids: dict[str, str] = {
            self.config.seat1_name: self.config.seat1_agent.strip().lower(),
            self.config.seat2_name: self.config.seat2_agent.strip().lower(),
        }
        self.profile_store = OpponentProfileStore()
        self.agents: dict[str, Agent] = {}
        self._init_agents()

        self.stacks: dict[str, int] = {
            p: self.config.starting_stack for p in self.players
        }
        self._hand_start_stacks: dict[str, int] = dict(self.stacks)
        self.hand_number: int = 0
        self.hands_completed: int = 0
        self.hand_wins: dict[str, int] = {p: 0 for p in self.players}
        self.hand_ties: int = 0
        self.net_chips: dict[str, int] = {p: 0 for p in self.players}
        self.total_pot_sum: int = 0
        self.showdown_hands: int = 0
        self.fold_hands: int = 0
        self.folds_by_player: dict[str, int] = {p: 0 for p in self.players}
        self.action_counts: dict[str, dict[str, int]] = {
            p: dict.fromkeys(ACTION_KEYS, 0) for p in self.players
        }
        self.decisions: list[DecisionRecord] = []
        self._equity_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

        self.button: str = self.players[0]
        self.game: GameState | None = None
        self.history: list[ActionRecord] = []
        self.action_log: list[dict[str, Any]] = []
        self.latest_action: dict[str, Any] | None = None
        self.status: str = (
            "idle"  # "idle", "running", "paused", "hand_complete", "match_over"
        )
        self.step_count: int = 0

    def _init_agents(self) -> None:
        for idx, player in enumerate(self.players):
            other = self.players[1 - idx]
            agent_seed = self._rng.randrange(2**32) + idx
            self.agents[player] = create_agent(
                self.agent_ids[player],
                seed=agent_seed,
                equity_simulations=self.config.equity_simulations,
                starting_stack=self.config.starting_stack,
                big_blind=self.config.big_blind,
                opponent_id=other,
                profile_store=self.profile_store,
            )

    def is_human_player(self, player: str | None) -> bool:
        if player is None:
            return False
        return self.agent_ids.get(player) == "human"

    @property
    def has_human_player(self) -> bool:
        return any(aid == "human" for aid in self.agent_ids.values())

    @property
    def human_players(self) -> tuple[str, ...]:
        return tuple(p for p in self.players if self.is_human_player(p))

    def _acquire(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise ActionBusyError(
                "Another action is currently processing in this session"
            )

    def reset(self, config: ArenaConfig | None = None) -> dict[str, Any]:
        """Reset the session to a fresh state with deterministic seeding."""
        self._acquire()
        try:
            target_cfg = config if config is not None else self.config
            new_rng = random.Random(target_cfg.seed)
            new_players = (target_cfg.seat1_name, target_cfg.seat2_name)
            new_agent_ids = {
                target_cfg.seat1_name: target_cfg.seat1_agent.strip().lower(),
                target_cfg.seat2_name: target_cfg.seat2_agent.strip().lower(),
            }
            new_profile_store = OpponentProfileStore()
            new_agents: dict[str, Agent] = {}
            for idx, player in enumerate(new_players):
                other = new_players[1 - idx]
                agent_seed = new_rng.randrange(2**32) + idx
                new_agents[player] = create_agent(
                    new_agent_ids[player],
                    seed=agent_seed,
                    equity_simulations=target_cfg.equity_simulations,
                    starting_stack=target_cfg.starting_stack,
                    big_blind=target_cfg.big_blind,
                    opponent_id=other,
                    profile_store=new_profile_store,
                )

            self.config = target_cfg
            self.session_id = uuid.uuid4().hex[:12]
            self._rng = new_rng
            self.players = new_players
            self.agent_ids = new_agent_ids
            self.profile_store = new_profile_store
            self.agents = new_agents
            self.stacks = {p: self.config.starting_stack for p in self.players}
            self._hand_start_stacks = dict(self.stacks)
            self.hand_number = 0
            self.hands_completed = 0
            self.hand_wins = {p: 0 for p in self.players}
            self.hand_ties = 0
            self.net_chips = {p: 0 for p in self.players}
            self.total_pot_sum = 0
            self.showdown_hands = 0
            self.fold_hands = 0
            self.folds_by_player = {p: 0 for p in self.players}
            self.action_counts = {
                p: dict.fromkeys(ACTION_KEYS, 0) for p in self.players
            }
            self.decisions = []
            self._equity_cache = {}
            self.button = self.players[0]
            self.game = None
            self.history = []
            self.action_log = []
            self.latest_action = None
            self.status = "idle"
            self.step_count = 0
            return self.serialize_state()
        finally:
            self._lock.release()

    def update_settings(
        self,
        *,
        playback_speed_ms: int | None = None,
        reveal_mode: str | None = None,
        seat1_agent: str | None = None,
        seat2_agent: str | None = None,
        human_equity_enabled: bool | None = None,
        human_equity_simulations: int | None = None,
    ) -> dict[str, Any]:
        """Update playback speed, card visibility, equity toggle, or seat agents."""
        self._acquire()
        try:
            new_speed = (
                self.config.playback_speed_ms
                if playback_speed_ms is None
                else int(playback_speed_ms)
            )
            new_reveal = (
                self.config.reveal_mode
                if reveal_mode is None
                else str(reveal_mode).strip().lower()
            )
            new_s1 = (
                self.config.seat1_agent
                if seat1_agent is None
                else str(seat1_agent).strip().lower()
            )
            new_s2 = (
                self.config.seat2_agent
                if seat2_agent is None
                else str(seat2_agent).strip().lower()
            )
            new_eq_enabled = (
                self.config.human_equity_enabled
                if human_equity_enabled is None
                else bool(human_equity_enabled)
            )
            new_eq_sims = (
                self.config.human_equity_simulations
                if human_equity_simulations is None
                else int(human_equity_simulations)
            )
            updated = ArenaConfig(
                seat1_name=self.config.seat1_name,
                seat2_name=self.config.seat2_name,
                seat1_agent=new_s1,
                seat2_agent=new_s2,
                starting_stack=self.config.starting_stack,
                small_blind=self.config.small_blind,
                big_blind=self.config.big_blind,
                seed=self.config.seed,
                equity_simulations=self.config.equity_simulations,
                action_profile=self.config.action_profile,
                playback_speed_ms=new_speed,
                reveal_mode=new_reveal,
                human_equity_enabled=new_eq_enabled,
                human_equity_simulations=new_eq_sims,
            )
            if new_s1 != self.config.seat1_agent:
                p1 = self.players[0]
                p2 = self.players[1]
                self.agents[p1] = create_agent(
                    new_s1,
                    seed=self._rng.randrange(2**32),
                    equity_simulations=updated.equity_simulations,
                    starting_stack=updated.starting_stack,
                    big_blind=updated.big_blind,
                    opponent_id=p2,
                    profile_store=self.profile_store,
                )
                self.agent_ids[p1] = new_s1
            if new_s2 != self.config.seat2_agent:
                p1 = self.players[0]
                p2 = self.players[1]
                self.agents[p2] = create_agent(
                    new_s2,
                    seed=self._rng.randrange(2**32),
                    equity_simulations=updated.equity_simulations,
                    starting_stack=updated.starting_stack,
                    big_blind=updated.big_blind,
                    opponent_id=p1,
                    profile_store=self.profile_store,
                )
                self.agent_ids[p2] = new_s2
            self.config = updated
            return self.serialize_state()
        finally:
            self._lock.release()

    def start_hand(
        self,
        *,
        auto_rebuy: bool = False,
        auto_advance_ai: bool | None = None,
    ) -> dict[str, Any]:
        """Start a new hand using the existing engine and log blind/deal events.

        When a human seat is present and `auto_advance_ai` is True (the default
        in human mode), any initial AI decision is automatically executed so
        control returns ready for the human's turn (or hand completion).
        """
        self._acquire()
        try:
            if any(s <= 0 for s in self.stacks.values()):
                if auto_rebuy:
                    self.stacks = {p: self.config.starting_stack for p in self.players}
                else:
                    self.status = "match_over"
                    raise ValueError(
                        "One player has 0 chips; reset session or enable auto_rebuy to start a new hand"
                    )

            self.hand_number += 1
            if self.hand_number == 1:
                self.button = self.players[0]
            else:
                self.button = (
                    self.players[1]
                    if self.button == self.players[0]
                    else self.players[0]
                )
            other = (
                self.players[1] if self.button == self.players[0] else self.players[0]
            )
            stacks_before = dict(self.stacks)
            self._hand_start_stacks = dict(self.stacks)
            deck = Deck(seed=self._rng.randrange(2**32))
            self.game = GameState(
                self.stacks,
                small_blind=self.config.small_blind,
                big_blind=self.config.big_blind,
                button=self.button,
                deck=deck,
            )
            self.history = []
            self.action_log = []
            self.step_count = 0

            # Initialize per-hand opponent profile state for each seat (strictly using
            # only the observer seat's own hole cards, never the opponent's hidden cards).
            eff_stack = min(stacks_before.values())
            for idx, observer_seat in enumerate(self.players):
                target_opp = self.players[1 - idx]
                prof = self.profile_store.get_or_create(target_opp)
                prof.start_hand(
                    hand_number=self.hand_number,
                    opponent_is_button=(self.button == target_opp),
                    starting_stack=stacks_before[target_opp],
                    effective_stack=eff_stack,
                    big_blind=self.config.big_blind,
                    hero_hole=self.game.hole_cards[observer_seat],
                    community_cards=tuple(self.game.community_cards),
                )

            sb_paid = min(stacks_before[self.button], self.config.small_blind)
            bb_paid = min(stacks_before[other], self.config.big_blind)

            self._append_log(
                {
                    "type": "hand_start",
                    "hand_number": self.hand_number,
                    "button": self.button,
                    "street": "preflop",
                    "text": f"Hand #{self.hand_number} started — {self.button} is on the button",
                }
            )
            self._append_log(
                {
                    "type": "blind",
                    "player": self.button,
                    "blind_type": "small_blind",
                    "amount": sb_paid,
                    "street": "preflop",
                    "text": f"{self.button} posts small blind ({sb_paid})",
                }
            )
            self._append_log(
                {
                    "type": "blind",
                    "player": other,
                    "blind_type": "big_blind",
                    "amount": bb_paid,
                    "street": "preflop",
                    "text": f"{other} posts big blind ({bb_paid})",
                }
            )

            # Handle rare case where posting blinds puts a short-stacked player all-in
            # and the engine immediately runs out all streets to showdown.
            self._log_board_transitions(0, len(self.game.community_cards))
            if self.game.complete:
                self._finalize_hand()
            else:
                self.status = "running"
                should_advance = (
                    self.has_human_player
                    if auto_advance_ai is None
                    else bool(auto_advance_ai)
                )
                if should_advance:
                    self._advance_ai_turns_locked()
            return self.serialize_state()
        finally:
            self._lock.release()

    def pause(self) -> dict[str, Any]:
        """Pause automatic playback while keeping manual stepping available."""
        self._acquire()
        try:
            if self.game is None:
                raise ValueError("Cannot pause before starting a hand")
            if self.status == "running":
                self.status = "paused"
            return self.serialize_state()
        finally:
            self._lock.release()

    def resume(self, *, auto_advance_ai: bool | None = None) -> dict[str, Any]:
        """Resume automatic playback on an active hand."""
        self._acquire()
        try:
            if self.game is None:
                raise ValueError("Cannot resume before starting a hand")
            if self.game.complete:
                raise ValueError("Cannot resume a completed hand")
            self.status = "running"
            should_advance = (
                self.has_human_player
                if auto_advance_ai is None
                else bool(auto_advance_ai)
            )
            if should_advance:
                self._advance_ai_turns_locked()
            return self.serialize_state()
        finally:
            self._lock.release()

    def step(self) -> dict[str, Any]:
        """Advance the current hand by exactly one AI agent decision."""
        self._acquire()
        try:
            self._execute_single_ai_step_locked()
            return self.serialize_state()
        finally:
            self._lock.release()

    def submit_human_action(
        self,
        action_type: str | ActionType,
        amount: int | None = None,
        *,
        player: str | None = None,
        expected_hand_number: int | None = None,
        expected_step_count: int | None = None,
        auto_advance: bool = True,
    ) -> dict[str, Any]:
        """Validate and apply a human player's action, then auto-advance AI responses."""
        self._acquire()
        try:
            if self.game is None:
                raise ValueError("No active hand; call start_hand first")
            if self.game.complete:
                raise ValueError("Hand is already complete")
            if (
                expected_hand_number is not None
                and int(expected_hand_number) != self.hand_number
            ):
                raise ValueError(
                    f"Stale hand_number {expected_hand_number}; "
                    f"current hand_number is {self.hand_number}"
                )
            if (
                expected_step_count is not None
                and int(expected_step_count) != self.step_count
            ):
                raise ValueError(
                    f"Stale or duplicate action submission for step_count "
                    f"{expected_step_count}; current step_count is {self.step_count}"
                )

            actor = self.game.current_player
            if actor is None:
                raise ValueError("No player can act in the current state")
            if player is not None and player != actor:
                raise ValueError(
                    f"Cannot act out of turn: it is {actor!r}'s turn, not {player!r}'s turn"
                )
            if not self.is_human_player(actor):
                raise ValueError(
                    f"Cannot submit human action on AI turn: current player {actor!r} "
                    f"is controlled by {self.agent_ids[actor]!r}"
                )

            engine_action = self._validate_and_normalize_human_action(
                actor, action_type, amount
            )
            observation = observe(
                self.game,
                actor,
                tuple(self.history),
                action_profile=self.config.action_profile,
            )
            human_telemetry = {
                "agent_type": "HumanPlayer",
                "agent_id": "human",
                "submitted_action": {
                    "action_type": ActionType(engine_action.type).value,
                    "amount": engine_action.amount,
                    "label": format_action_label(engine_action),
                },
                "direct_telemetry": ["submitted_action"],
                "derived_analysis": [],
                "unavailable_fields": [
                    "internal_policy_distribution (human player decision)"
                ],
            }
            self._apply_action_to_engine_locked(
                actor=actor,
                action=engine_action,
                observation=observation,
                telemetry=human_telemetry,
                is_human=True,
            )

            if auto_advance and not self.game.complete and self.status == "running":
                self._advance_ai_turns_locked()

            return self.serialize_state()
        finally:
            self._lock.release()

    def _validate_and_normalize_human_action(
        self,
        actor: str,
        action_type: str | ActionType,
        amount: int | None,
    ) -> Action:
        assert self.game is not None
        if not isinstance(action_type, (str, ActionType)):
            raise TypeError("action_type must be a string or ActionType")
        raw_kind = str(action_type).strip().lower()
        try:
            kind = ActionType(raw_kind)
        except ValueError as exc:
            raise ValueError(f"Unsupported action_type {action_type!r}") from exc

        round_ = self.game.betting
        contrib = round_.contributions[actor]
        stack = round_.stacks[actor]
        current_bet = round_.current_bet
        owed = current_bet - contrib
        maximum = contrib + stack
        can_raise = actor not in round_.acted_since_full_raise

        if kind == ActionType.FOLD:
            if amount is not None:
                raise ValueError("fold action does not accept an amount")
            return Action(ActionType.FOLD)

        if kind == ActionType.CHECK:
            if amount is not None:
                raise ValueError("check action does not accept an amount")
            if owed > 0:
                raise ValueError(
                    f"Cannot check while facing a bet ({min(stack, owed)} to call)"
                )
            return Action(ActionType.CHECK)

        if kind == ActionType.CALL:
            if amount is not None:
                raise ValueError("call action does not accept a custom amount")
            if owed <= 0:
                raise ValueError("Nothing to call; check instead")
            return Action(ActionType.CALL)

        if kind == ActionType.ALL_IN:
            if stack <= 0:
                raise ValueError("Player has no remaining chips to go all-in")
            if maximum > current_bet and not can_raise:
                raise ValueError("Action has not been reopened for an all-in raise")
            return Action(ActionType.ALL_IN)

        if kind == ActionType.BET:
            if current_bet != 0:
                raise ValueError(
                    "Bet is only legal when no bet is outstanding on this street; use raise instead"
                )
            if not can_raise or stack <= 0:
                raise ValueError("Player cannot open a bet in this state")
            if amount is None or isinstance(amount, bool):
                raise ValueError("Bet action requires an integer target amount")
            target = int(amount)
            min_bet = contrib + min(round_.min_raise, stack)
            if target < min_bet or target > maximum:
                raise ValueError(
                    f"Bet amount must be between {min_bet} and {maximum} (got {target})"
                )
            return Action(ActionType.BET, target)

        if kind == ActionType.RAISE:
            if current_bet == 0:
                raise ValueError(
                    "Raise is only legal when facing an existing bet or blind; use bet instead"
                )
            if not can_raise:
                raise ValueError("Raise action has not been reopened for this player")
            if maximum <= current_bet:
                raise ValueError(
                    "Player stack does not exceed the current bet; call instead"
                )
            if amount is None or isinstance(amount, bool):
                raise ValueError("Raise action requires an integer target street total")
            target = int(amount)
            min_raise_to = min(maximum, current_bet + round_.min_raise)
            if target < min_raise_to or target > maximum:
                raise ValueError(
                    f"Raise target must be between {min_raise_to} and {maximum} (got {target})"
                )
            return Action(ActionType.RAISE, target)

        raise ValueError(f"Unsupported action {kind!r}")

    def _advance_ai_turns_locked(self, max_steps: int = 64) -> None:
        """Advance consecutive AI turns until human turn, hand completion, or pause."""
        steps = 0
        while (
            self.game is not None
            and not self.game.complete
            and self.status == "running"
            and self.game.current_player is not None
            and not self.is_human_player(self.game.current_player)
        ):
            if steps >= max_steps:
                raise RuntimeError("Exceeded maximum consecutive AI steps")
            self._execute_single_ai_step_locked()
            steps += 1

    def _execute_single_ai_step_locked(self) -> None:
        if self.game is None:
            raise ValueError("No active hand; call start_hand first")
        if self.game.complete:
            raise ValueError("Hand is already complete")
        actor = self.game.current_player
        if actor is None:
            raise RuntimeError("Nonterminal hand has no current_player")
        if self.is_human_player(actor):
            raise ValueError(
                f"Current player {actor!r} is human-controlled; submit a human action"
            )

        observation = observe(
            self.game,
            actor,
            tuple(self.history),
            action_profile=self.config.action_profile,
        )
        agent = self.agents[actor]
        agent_id = self.agent_ids[actor]
        action, telemetry = inspect_and_execute_agent(agent, agent_id, observation)
        if action not in observation.legal_actions:
            raise ValueError(f"Agent {actor!r} returned illegal action {action!r}")

        self._apply_action_to_engine_locked(
            actor=actor,
            action=action,
            observation=observation,
            telemetry=telemetry,
            is_human=False,
        )

    def _apply_action_to_engine_locked(
        self,
        *,
        actor: str,
        action: Action,
        observation: Observation,
        telemetry: dict[str, Any],
        is_human: bool,
    ) -> None:
        assert self.game is not None
        street_before = self.game.street.value
        cards_before = len(self.game.community_cards)
        contrib_before = self.game.betting.contributions[actor]
        pot_contrib_before = self.game.pot.contributions.get(actor, 0)
        stack_before = self.game.stacks[actor]

        self.game.act(action, player=actor)
        self.history.append(ActionRecord(actor, action, street_before))
        self.step_count += 1

        pot_contrib_after = self.game.pot.contributions.get(actor, 0)
        paid = pot_contrib_after - pot_contrib_before
        street_total = contrib_before + paid
        went_all_in = action.type == ActionType.ALL_IN or (
            paid > 0 and paid == stack_before
        )
        kind_str = ActionType(action.type).value
        if kind_str in self.action_counts[actor]:
            self.action_counts[actor][kind_str] += 1
        if kind_str == ActionType.FOLD.value:
            self.folds_by_player[actor] += 1

        decision_id = len(self.decisions) + 1
        action_event = serialize_action(
            actor,
            action,
            street_before,
            paid=paid,
            street_total=street_total,
            is_all_in=went_all_in,
            decision_id=decision_id,
        )

        public_context = {
            "pot": observation.pot,
            "to_call": observation.to_call,
            "pot_odds": round(observation.pot_odds, 6),
            "stack": observation.stack,
            "opponent_stack": observation.opponent_stack,
            "effective_stack": observation.effective_stack,
            "street_contribution": observation.street_contribution,
            "opponent_street_contribution": observation.opponent_street_contribution,
            "in_position": observation.in_position,
            "is_button": observation.is_button,
            "community_cards": [
                serialize_card(c, hidden=False) for c in observation.community_cards
            ],
            "legal_actions": [
                serialize_engine_action_choice(a) for a in observation.legal_actions
            ],
        }
        agent_id = self.agent_ids[actor]
        catalog_entry = _CATALOG_BY_ID[agent_id]
        record = DecisionRecord(
            decision_id=decision_id,
            hand_number=self.hand_number,
            step_index=self.step_count,
            timestamp_iso=current_utc_iso(),
            street=street_before,
            player=actor,
            agent_id=agent_id,
            agent_name=catalog_entry.name,
            is_human=is_human,
            chosen_action=action_event,
            public_context=public_context,
            telemetry=telemetry,
        )
        self.decisions.append(record)

        self.latest_action = action_event
        self._append_log(action_event)

        cards_after = len(self.game.community_cards)
        if cards_after > cards_before:
            self._log_board_transitions(cards_before, cards_after)

        # Update the other seat's OpponentProfile with the public action just taken
        observer_seat = self.players[1] if actor == self.players[0] else self.players[0]
        observer_obs = observe(
            self.game,
            observer_seat,
            tuple(self.history),
            action_profile=self.config.action_profile,
        )
        self.profile_store.get_or_create(actor).ingest_observation(
            observer_obs,
            hand_number=self.hand_number,
        )

        if self.game.complete:
            self._finalize_hand()

    def _log_board_transitions(self, before_count: int, after_count: int) -> None:
        assert self.game is not None
        cards = self.game.community_cards
        if before_count < 3 <= after_count:
            flop_str = " ".join(
                f"{RANK_LABELS[c.rank]}{SUIT_SYMBOLS[c.suit]}" for c in cards[:3]
            )
            self._append_log(
                {
                    "type": "street_deal",
                    "street": "flop",
                    "cards": [serialize_card(c) for c in cards[:3]],
                    "text": f"Flop dealt: {flop_str}",
                }
            )
        if before_count < 4 <= after_count:
            turn_c = cards[3]
            turn_str = f"{RANK_LABELS[turn_c.rank]}{SUIT_SYMBOLS[turn_c.suit]}"
            self._append_log(
                {
                    "type": "street_deal",
                    "street": "turn",
                    "cards": [serialize_card(turn_c)],
                    "text": f"Turn dealt: {turn_str}",
                }
            )
        if before_count < 5 <= after_count:
            river_c = cards[4]
            river_str = f"{RANK_LABELS[river_c.rank]}{SUIT_SYMBOLS[river_c.suit]}"
            self._append_log(
                {
                    "type": "street_deal",
                    "street": "river",
                    "cards": [serialize_card(river_c)],
                    "text": f"River dealt: {river_str}",
                }
            )

    def _finalize_hand(self) -> None:
        assert self.game is not None
        self.stacks = dict(self.game.stacks)
        self.hands_completed += 1
        self.total_pot_sum += self.game.current_pot
        for p in self.players:
            self.net_chips[p] += self.stacks[p] - self._hand_start_stacks[p]

        # Finalize opponent profiles with legitimate terminal outcomes and showdown cards
        for idx, target_opp in enumerate(self.players):
            observer_seat = self.players[1 - idx]
            reached_sd = bool(
                self.game.hand_ranks and target_opp in self.game.hand_ranks
            )
            revealed_cards = self.game.hole_cards[target_opp] if reached_sd else None
            opp_net = self.stacks[target_opp] - self._hand_start_stacks[target_opp]
            opp_won = self.game.winners.get(target_opp, 0) > 0
            self.profile_store.get_or_create(target_opp).finalize_hand(
                hand_number=self.hand_number,
                history=tuple(self.history),
                hero_player=observer_seat,
                community_cards=tuple(self.game.community_cards),
                reached_showdown=reached_sd,
                revealed_opponent_cards=revealed_cards,
                opponent_net_chips=opp_net,
                opponent_won=opp_won,
                final_pot=self.game.current_pot,
            )

        if self.game.hand_ranks:
            self.showdown_hands += 1
            contender_parts = []
            for p in self.players:
                if p in self.game.hand_ranks:
                    rank = self.game.hand_ranks[p]
                    cat_name = HAND_CATEGORY_NAMES.get(
                        int(rank.category), str(rank.category)
                    )
                    hole_str = " ".join(
                        f"{RANK_LABELS[c.rank]}{SUIT_SYMBOLS[c.suit]}"
                        for c in self.game.hole_cards[p]
                    )
                    contender_parts.append(f"{p} shows {hole_str} ({cat_name})")
            self._append_log(
                {
                    "type": "showdown",
                    "street": "showdown",
                    "text": "Showdown — " + "; ".join(contender_parts),
                }
            )
        else:
            self.fold_hands += 1

        winners = [(p, amt) for p, amt in self.game.winners.items() if amt > 0]
        if len(winners) == 1:
            winner_name, amount = winners[0]
            self.hand_wins[winner_name] += 1
            self._append_log(
                {
                    "type": "winner",
                    "winners": {winner_name: amount},
                    "street": "complete",
                    "text": f"{winner_name} wins {amount} chips",
                }
            )
        elif len(winners) > 1:
            self.hand_ties += 1
            parts = [f"{p} receives {amt} chips" for p, amt in winners]
            self._append_log(
                {
                    "type": "winner",
                    "winners": dict(winners),
                    "street": "complete",
                    "text": "Split pot — " + ", ".join(parts),
                }
            )

        if any(s <= 0 for s in self.stacks.values()):
            self.status = "match_over"
        else:
            self.status = "hand_complete"

    def _append_log(self, event: dict[str, Any]) -> None:
        entry = {"index": len(self.action_log) + 1, **event}
        self.action_log.append(entry)

    def _should_reveal_player_cards(self, player: str) -> bool:
        mode = self.config.reveal_mode
        if mode == "all":
            return True
        if mode == "hidden":
            return False
        if self.is_human_player(player):
            return True
        if self.has_human_player:
            # Opponent cards remain hidden during play and are revealed at showdown
            return bool(
                self.game is not None
                and self.game.complete
                and self.game.hand_ranks
                and player in self.game.hand_ranks
                and mode == "showdown"
            )
        if self.game is not None and self.game.complete and mode == "showdown":
            return True
        return bool(mode == "seat1" and player == self.players[0])

    def _serialize_human_controls(self) -> dict[str, Any]:
        """Compute legal actions and bet/raise bounds for the active human turn."""
        if self.game is None or self.game.complete:
            return {
                "is_human_turn": False,
                "acting_player": None,
                "hand_number": self.hand_number,
                "step_count": self.step_count,
                "to_call": 0,
                "pot": self.game.current_pot if self.game is not None else 0,
                "stack": 0,
                "current_bet": 0,
                "street_contribution": 0,
                "min_raise_increment": 0,
                "can_fold": False,
                "can_check": False,
                "can_call": False,
                "call_amount": 0,
                "can_bet": False,
                "min_bet": None,
                "max_bet": None,
                "default_bet": None,
                "can_raise": False,
                "min_raise_to": None,
                "max_raise_to": None,
                "default_raise_to": None,
                "can_all_in": False,
                "all_in_to": None,
                "all_in_paid": None,
                "sizing_mode": None,
                "min_amount": None,
                "max_amount": None,
                "default_amount": None,
                "legal_action_types": [],
                "preset_actions": [],
            }

        actor = self.game.current_player
        is_human_turn = bool(actor is not None and self.is_human_player(actor))
        if actor is None:
            return {
                "is_human_turn": False,
                "acting_player": None,
                "hand_number": self.hand_number,
                "step_count": self.step_count,
                "to_call": 0,
                "pot": self.game.current_pot,
                "stack": 0,
                "current_bet": 0,
                "street_contribution": 0,
                "min_raise_increment": 0,
                "can_fold": False,
                "can_check": False,
                "can_call": False,
                "call_amount": 0,
                "can_bet": False,
                "min_bet": None,
                "max_bet": None,
                "default_bet": None,
                "can_raise": False,
                "min_raise_to": None,
                "max_raise_to": None,
                "default_raise_to": None,
                "can_all_in": False,
                "all_in_to": None,
                "all_in_paid": None,
                "sizing_mode": None,
                "min_amount": None,
                "max_amount": None,
                "default_amount": None,
                "legal_action_types": [],
                "preset_actions": [],
            }

        round_ = self.game.betting
        obs = observe(
            self.game,
            actor,
            tuple(self.history),
            action_profile=self.config.action_profile,
        )
        contrib = round_.contributions[actor]
        stack = round_.stacks[actor]
        current_bet = round_.current_bet
        owed = max(0, current_bet - contrib)
        call_amount = min(stack, owed)
        maximum = contrib + stack
        can_raise_reopened = actor not in round_.acted_since_full_raise

        can_fold = True
        can_check = owed == 0
        can_call = owed > 0 and stack > 0
        can_bet = current_bet == 0 and stack > 0 and can_raise_reopened
        can_raise = current_bet > 0 and maximum > current_bet and can_raise_reopened
        can_all_in = stack > 0 and (maximum <= current_bet or can_raise_reopened)

        min_bet: int | None = None
        max_bet: int | None = None
        default_bet: int | None = None
        if can_bet:
            min_bet = contrib + min(round_.min_raise, stack)
            max_bet = maximum
            half_pot = contrib + max(
                min(round_.min_raise, stack), int(self.game.current_pot * 0.5)
            )
            default_bet = min(max_bet, max(min_bet, half_pot))

        min_raise_to: int | None = None
        max_raise_to: int | None = None
        default_raise_to: int | None = None
        if can_raise:
            min_raise_to = min(maximum, current_bet + round_.min_raise)
            max_raise_to = maximum
            after_call_pot = self.game.current_pot + call_amount
            half_pot_raise = current_bet + max(
                round_.min_raise, int(after_call_pot * 0.5)
            )
            default_raise_to = min(max_raise_to, max(min_raise_to, half_pot_raise))

        sizing_mode = "bet" if can_bet else ("raise" if can_raise else None)
        min_amount = min_bet if can_bet else min_raise_to
        max_amount = max_bet if can_bet else max_raise_to
        default_amount = default_bet if can_bet else default_raise_to

        legal_types: list[str] = []
        if can_fold:
            legal_types.append("fold")
        if can_check:
            legal_types.append("check")
        if can_call:
            legal_types.append("call")
        if can_bet:
            legal_types.append("bet")
        if can_raise:
            legal_types.append("raise")
        if can_all_in:
            legal_types.append("all_in")

        presets = [serialize_engine_action_choice(act) for act in obs.legal_actions]

        return {
            "is_human_turn": is_human_turn,
            "acting_player": actor,
            "hand_number": self.hand_number,
            "step_count": self.step_count,
            "to_call": call_amount,
            "pot": self.game.current_pot,
            "stack": stack,
            "current_bet": current_bet,
            "street_contribution": contrib,
            "min_raise_increment": round_.min_raise,
            "can_fold": can_fold,
            "can_check": can_check,
            "can_call": can_call,
            "call_amount": call_amount,
            "can_bet": can_bet,
            "min_bet": min_bet,
            "max_bet": max_bet,
            "default_bet": default_bet,
            "can_raise": can_raise,
            "min_raise_to": min_raise_to,
            "max_raise_to": max_raise_to,
            "default_raise_to": default_raise_to,
            "can_all_in": can_all_in,
            "all_in_to": maximum if can_all_in else None,
            "all_in_paid": stack if can_all_in else None,
            "sizing_mode": sizing_mode,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "default_amount": default_amount,
            "legal_action_types": legal_types,
            "preset_actions": presets,
        }

    def _serialize_equity_panel(self) -> dict[str, Any]:
        """Build the equity panel payload without leaking hidden opponent hole cards."""
        human_equity: dict[str, Any] | None = None
        if (
            self.config.human_equity_enabled
            and self.game is not None
            and self.human_players
        ):
            human_seat = self.human_players[0]
            board_tuple = tuple(self.game.community_cards)
            cache_key = (
                self.session_id,
                self.hand_number,
                len(board_tuple),
                human_seat,
                self.config.human_equity_simulations,
            )
            if cache_key not in self._equity_cache:
                derived_seed = (
                    (self.config.seed * 1_000_003)
                    ^ (self.hand_number * 97_531)
                    ^ (len(board_tuple) * 7_919)
                ) & 0xFFFFFFFF
                self._equity_cache[cache_key] = estimate_human_public_equity(
                    self.game.hole_cards[human_seat],
                    board_tuple,
                    simulations=self.config.human_equity_simulations,
                    seed=derived_seed,
                )
            human_equity = dict(self._equity_cache[cache_key])
            human_equity["player"] = human_seat
            human_equity["street"] = self.game.street.value
        elif not self.config.human_equity_enabled:
            human_equity = {
                "enabled": False,
                "is_exact": False,
                "label": "Human public Monte Carlo equity estimation is disabled.",
                "equity": None,
            }

        ai_equity_telemetry: dict[str, Any] | None = None
        ai_note: str | None = None
        ai_Decisions = [
            d
            for d in self.decisions
            if not d.is_human and d.hand_number == self.hand_number
        ]
        if ai_Decisions:
            latest_ai = ai_Decisions[-1]
            if latest_ai.agent_id == "equity":
                ai_equity_telemetry = {
                    "decision_id": latest_ai.decision_id,
                    "player": latest_ai.player,
                    "agent_name": latest_ai.agent_name,
                    "street": latest_ai.street,
                    **latest_ai.telemetry.get("equity_estimate", {}),
                }
            elif latest_ai.agent_id == "harvey":
                ai_equity_telemetry = {
                    "decision_id": latest_ai.decision_id,
                    "player": latest_ai.player,
                    "agent_name": latest_ai.agent_name,
                    "street": latest_ai.street,
                    "equity": latest_ai.telemetry.get("hero_range_equity"),
                    "uniform_equity": latest_ai.telemetry.get("hero_uniform_equity"),
                    "lambda_value": latest_ai.telemetry.get("lambda_value"),
                    "estimated_fold_prob": latest_ai.telemetry.get(
                        "estimated_fold_prob"
                    ),
                    "method": "bayesian_range_monte_carlo",
                }
                ai_note = (
                    "HarveySpecter estimates equity against the opponent's Bayesian posterior "
                    "hand range and blends MCCFR with exploitative EV adjustments."
                )
            elif latest_ai.agent_id == "mccfr":
                ai_note = (
                    "AbstractHUNLAgent (MCCFR) acts from counterfactual regret strategy "
                    "probabilities over abstract buckets rather than computing a hand equity estimate."
                )
            elif latest_ai.agent_id == "heuristic":
                ai_note = (
                    "HeuristicAgent uses deterministic preflop/hand-category strength bands "
                    "rather than a calibrated win-probability equity estimate."
                )
            elif latest_ai.agent_id == "random":
                ai_note = "RandomAgent samples uniformly over legal actions without estimating hand equity."

        return {
            "human_equity_enabled": self.config.human_equity_enabled,
            "human_equity_simulations": self.config.human_equity_simulations,
            "human_equity": human_equity,
            "ai_equity_telemetry": ai_equity_telemetry,
            "ai_note": ai_note,
        }

    def serialize_match_statistics(self) -> dict[str, Any]:
        """Return match-scoped statistics cleared on session reset."""
        played = self.hands_completed
        win_rates = {
            p: round(self.hand_wins[p] / played, 4) if played > 0 else 0.0
            for p in self.players
        }
        hands_lost = {
            p: max(0, played - self.hand_wins[p] - self.hand_ties) for p in self.players
        }
        action_dist: dict[str, dict[str, float]] = {}
        for p in self.players:
            total_acts = sum(self.action_counts[p].values())
            action_dist[p] = {
                k: round(v / total_acts, 4) if total_acts > 0 else 0.0
                for k, v in self.action_counts[p].items()
            }

        opponent_player = (
            next(
                (p for p in self.players if not self.is_human_player(p)),
                self.players[1],
            )
            if self.has_human_player
            else self.players[1]
        )

        return {
            "hands_played": played,
            "hands_won": dict(self.hand_wins),
            "hands_lost": hands_lost,
            "hand_ties": self.hand_ties,
            "net_chips": dict(self.net_chips),
            "win_rate": win_rates,
            "average_pot_size": round(self.total_pot_sum / played, 2)
            if played > 0
            else 0.0,
            "showdown_hands": self.showdown_hands,
            "showdown_frequency": round(self.showdown_hands / played, 4)
            if played > 0
            else 0.0,
            "fold_hands": self.fold_hands,
            "fold_frequency": round(self.fold_hands / played, 4) if played > 0 else 0.0,
            "folds_by_player": dict(self.folds_by_player),
            "action_counts": {p: dict(self.action_counts[p]) for p in self.players},
            "action_distribution": action_dist,
            "opponent_player": opponent_player,
            "opponent_action_counts": dict(self.action_counts[opponent_player]),
            "opponent_action_distribution": dict(action_dist[opponent_player]),
        }

    def _default_profiled_player(self) -> str:
        """Select which player's profile is highlighted by default in the UI.

        - If HarveySpecter is seated in one seat, highlight the opponent that
          HarveySpecter is modeling (e.g., the human player when playing against Harvey).
        - Otherwise, if a human is seated, highlight the AI opponent.
        - Otherwise, highlight Seat 2.
        """
        for idx, p in enumerate(self.players):
            if self.agent_ids.get(p) == "harvey":
                return self.players[1 - idx]
        if self.has_human_player:
            return next(
                (p for p in self.players if not self.is_human_player(p)),
                self.players[1],
            )
        return self.players[1]

    def serialize_opponent_intelligence(
        self, target_player: str | None = None
    ) -> dict[str, Any]:
        """Serialize live Opponent Profile, Archetype Classification, and Bayesian Range."""
        target = (
            target_player.strip()
            if target_player and target_player.strip() in self.players
            else self._default_profiled_player()
        )
        observer = self.players[1] if target == self.players[0] else self.players[0]
        prof = self.profile_store.get_or_create(target)
        stats = prof.stats_snapshot()
        archetype = prof.archetype_distribution()
        range_snap = prof.range_snapshot()

        # Also provide brief profiles for both seats so the UI can switch between seats
        profiles_by_seat: dict[str, dict[str, Any]] = {}
        for p in self.players:
            p_prof = self.profile_store.get_or_create(p)
            profiles_by_seat[p] = {
                "opponent_id": p,
                "stats": p_prof.stats_snapshot().to_dict(),
                "archetype": p_prof.archetype_distribution().to_dict(),
                "bayesian_range": p_prof.range_snapshot().to_dict(),
            }

        latest_harvey_decision = next(
            (d.serialize() for d in reversed(self.decisions) if d.agent_id == "harvey"),
            None,
        )

        return {
            "target_player": target,
            "observer_player": observer,
            "observer_agent_id": self.agent_ids.get(observer, ""),
            "model_version": prof.model_version,
            "last_hand_number": prof.last_hand_number,
            "opponent_profile": stats.to_dict(),
            "archetype_classification": archetype.to_dict(),
            "bayesian_range": range_snap.to_dict(),
            "latest_harvey_decision": latest_harvey_decision,
            "profiles_by_player": profiles_by_seat,
        }

    def reset_opponent_profile(self, opponent_id: str | None = None) -> dict[str, Any]:
        """Reset one opponent's profile (or all profiles if `opponent_id` is None)."""
        self._acquire()
        try:
            if opponent_id is None or opponent_id.strip().lower() == "all":
                for p in self.players:
                    self.profile_store.reset_profile(p)
            else:
                target = opponent_id.strip()
                self.profile_store.reset_profile(target)
            for agent in self.agents.values():
                if isinstance(agent, HarveySpecterAgent):
                    agent.lambda_history.clear()
                    agent.last_decision_report = None
            return self.serialize_state()
        finally:
            self._lock.release()

    def save_opponent_profile(
        self,
        opponent_id: str | None = None,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Export an opponent profile dictionary and optionally write it to `path`."""
        self._acquire()
        try:
            target = (
                opponent_id.strip()
                if opponent_id and opponent_id.strip()
                else self._default_profiled_player()
            )
            exported = self.profile_store.export_profile(target)
            saved_path: str | None = None
            if path:
                out = self.profile_store.save_profile(target, path=path)
                saved_path = str(out)
            return {
                "status": "saved",
                "opponent_id": target,
                "saved_path": saved_path,
                "profile": exported,
                "opponent_intelligence": self.serialize_opponent_intelligence(target),
            }
        finally:
            self._lock.release()

    def load_opponent_profile(
        self,
        *,
        opponent_id: str | None = None,
        path: str | None = None,
        payload: dict[str, Any] | str | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Import an opponent profile from a JSON payload or file path."""
        self._acquire()
        try:
            target = (
                opponent_id.strip()
                if opponent_id and opponent_id.strip()
                else self._default_profiled_player()
            )
            if payload is not None:
                prof = self.profile_store.import_profile(payload)
                load_status = "imported_from_payload"
            elif path is not None:
                prof, load_status = self.profile_store.load_profile(
                    target, path=path, strict=strict
                )
            else:
                raise ValueError(
                    "Either 'payload' or 'path' must be provided to load_opponent_profile"
                )
            return {
                "status": load_status,
                "opponent_id": prof.opponent_id,
                "profile": prof.export_dict(),
                "state": self.serialize_state(),
            }
        finally:
            self._lock.release()

    def serialize_decisions(
        self,
        *,
        hand_number: int | None = None,
        decision_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return immutable serialized decision timeline snapshots."""
        records = self.decisions
        if hand_number is not None:
            records = [d for d in records if d.hand_number == int(hand_number)]
        if decision_id is not None:
            records = [d for d in records if d.decision_id == int(decision_id)]
        return [d.serialize() for d in records]

    def serialize_state(self) -> dict[str, Any]:
        """Return a complete, frontend-safe snapshot of the current Arena state."""
        seats: list[dict[str, Any]] = []
        for idx, player in enumerate(self.players):
            agent_id = self.agent_ids[player]
            catalog_entry = _CATALOG_BY_ID[agent_id]
            reveal = self._should_reveal_player_cards(player)
            is_human = self.is_human_player(player)
            if self.game is not None:
                raw_hole = self.game.hole_cards[player]
                hole_cards = [serialize_card(c, hidden=not reveal) for c in raw_hole]
                current_bet = (
                    0
                    if self.game.complete
                    else self.game.betting.contributions.get(player, 0)
                )
                total_committed = self.game.pot.contributions.get(player, 0)
                is_folded = player in self.game.folded
                is_all_in = player in self.game.all_in
                is_turn = not self.game.complete and self.game.current_player == player
                hand_category = None
                if reveal and self.game.hand_ranks and player in self.game.hand_ranks:
                    cat = int(self.game.hand_ranks[player].category)
                    hand_category = HAND_CATEGORY_NAMES.get(cat, str(cat))
                winnings = self.game.winners.get(player, 0)
            else:
                hole_cards = [serialize_card(None, hidden=True) for _ in range(2)]
                current_bet = 0
                total_committed = 0
                is_folded = False
                is_all_in = False
                is_turn = False
                hand_category = None
                winnings = 0

            seats.append(
                {
                    "seat_index": idx + 1,
                    "name": player,
                    "agent_id": agent_id,
                    "agent_name": catalog_entry.name,
                    "is_human": is_human,
                    "stack": self.stacks[player],
                    "net_chips": self.net_chips[player],
                    "current_bet": current_bet,
                    "total_committed": total_committed,
                    "is_button": self.button == player,
                    "is_current_player": is_turn,
                    "is_folded": is_folded,
                    "is_all_in": is_all_in,
                    "hole_cards": hole_cards,
                    "cards_revealed": reveal,
                    "hand_category": hand_category,
                    "winnings": winnings,
                    "hands_won": self.hand_wins[player],
                }
            )

        if self.game is not None:
            community = [serialize_card(c) for c in self.game.community_cards]
            pot = self.game.current_pot
            street = self.game.street.value
            current_player = None if self.game.complete else self.game.current_player
            winners = dict(self.game.winners)
            hand_complete = self.game.complete
        else:
            community = []
            pot = 0
            street = "idle"
            current_player = None
            winners = {}
            hand_complete = False

        serialized_decisions = self.serialize_decisions()
        latest_ai_decision = next(
            (d for d in reversed(serialized_decisions) if not d["is_human"]),
            None,
        )

        return {
            "session_id": self.session_id,
            "mode": "human_vs_agent" if self.has_human_player else "bot_vs_bot",
            "has_human_player": self.has_human_player,
            "status": self.status,
            "hand_number": self.hand_number,
            "hands_completed": self.hands_completed,
            "hand_ties": self.hand_ties,
            "step_count": self.step_count,
            "street": street,
            "pot": pot,
            "button": self.button,
            "current_player": current_player,
            "hand_complete": hand_complete,
            "winners": winners,
            "community_cards": community,
            "seats": seats,
            "human_controls": self._serialize_human_controls(),
            "equity_panel": self._serialize_equity_panel(),
            "match_statistics": self.serialize_match_statistics(),
            "opponent_intelligence": self.serialize_opponent_intelligence(),
            "decisions": serialized_decisions,
            "latest_ai_decision": latest_ai_decision,
            "latest_action": self.latest_action,
            "action_log": list(self.action_log),
            "config": {
                "seat1_name": self.config.seat1_name,
                "seat2_name": self.config.seat2_name,
                "seat1_agent": self.config.seat1_agent,
                "seat2_agent": self.config.seat2_agent,
                "starting_stack": self.config.starting_stack,
                "small_blind": self.config.small_blind,
                "big_blind": self.config.big_blind,
                "seed": self.config.seed,
                "equity_simulations": self.config.equity_simulations,
                "action_profile": self.config.action_profile,
                "playback_speed_ms": self.config.playback_speed_ms,
                "reveal_mode": self.config.reveal_mode,
                "human_equity_enabled": self.config.human_equity_enabled,
                "human_equity_simulations": self.config.human_equity_simulations,
            },
        }
