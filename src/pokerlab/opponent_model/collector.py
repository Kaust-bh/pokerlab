"""Opponent Data Collector for recording strictly public gameplay observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pokerlab.agents.base import ActionRecord, Observation
from pokerlab.engine import Action, ActionType, Card
from pokerlab.opponent_model.models import HandObservationRecord, ObservedActionEvent


class OpponentDataCollector:
    """Collects and validates public per-hand and per-action opponent observations.

    Privacy invariant:
    - Never reads or stores hidden opponent hole cards during a hand.
    - Rejects any attempt to record opponent hole cards unless `reached_showdown=True`.
    - Tracks `processed_history_len` per hand so repeated observations within the same
      hand only append newly observed actions (preventing duplicate updates).
    """

    def __init__(self, default_opponent_id: str = "opponent") -> None:
        self.default_opponent_id = default_opponent_id
        self._hands_by_opponent: dict[str, dict[int, HandObservationRecord]] = {}
        self._active_hand_by_opponent: dict[str, int] = {}
        self._hand_counter_by_opponent: dict[str, int] = {}

    def reset(self, opponent_id: str | None = None) -> None:
        """Clear recorded observations for one opponent or all opponents."""
        if opponent_id is None:
            self._hands_by_opponent.clear()
            self._active_hand_by_opponent.clear()
            self._hand_counter_by_opponent.clear()
        else:
            self._hands_by_opponent.pop(opponent_id, None)
            self._active_hand_by_opponent.pop(opponent_id, None)
            self._hand_counter_by_opponent.pop(opponent_id, None)

    def start_hand(
        self,
        *,
        opponent_id: str | None = None,
        hand_number: int | None = None,
        opponent_is_button: bool,
        starting_stack: int = 200,
        effective_stack: int | None = None,
        big_blind: int = 2,
    ) -> HandObservationRecord:
        """Start or retrieve a hand record for the given opponent."""
        opp_id = (opponent_id or self.default_opponent_id).strip() or "opponent"
        opp_hands = self._hands_by_opponent.setdefault(opp_id, {})

        if hand_number is None:
            next_num = self._hand_counter_by_opponent.get(opp_id, 0) + 1
            hand_number = next_num
        self._hand_counter_by_opponent[opp_id] = max(
            self._hand_counter_by_opponent.get(opp_id, 0),
            int(hand_number),
        )

        if hand_number in opp_hands:
            self._active_hand_by_opponent[opp_id] = hand_number
            return opp_hands[hand_number]

        position = "button" if opponent_is_button else "big_blind"
        eff_stack = (
            int(effective_stack) if effective_stack is not None else int(starting_stack)
        )
        record = HandObservationRecord(
            opponent_id=opp_id,
            hand_number=int(hand_number),
            opponent_position=position,
            opponent_is_button=bool(opponent_is_button),
            starting_stack=max(1, int(starting_stack)),
            effective_stack=max(1, eff_stack),
            big_blind=max(1, int(big_blind)),
        )
        opp_hands[int(hand_number)] = record
        self._active_hand_by_opponent[opp_id] = int(hand_number)
        return record

    def get_hand(
        self, opponent_id: str, hand_number: int
    ) -> HandObservationRecord | None:
        return self._hands_by_opponent.get(opponent_id, {}).get(hand_number)

    def get_active_hand(
        self, opponent_id: str | None = None
    ) -> HandObservationRecord | None:
        opp_id = opponent_id or self.default_opponent_id
        active_num = self._active_hand_by_opponent.get(opp_id)
        if active_num is None:
            return None
        return self.get_hand(opp_id, active_num)

    def list_hands(
        self,
        opponent_id: str | None = None,
        *,
        completed_only: bool = False,
    ) -> list[HandObservationRecord]:
        """Return hands in chronological order for `opponent_id`."""
        opp_id = opponent_id or self.default_opponent_id
        opp_hands = self._hands_by_opponent.get(opp_id, {})
        records = [opp_hands[k] for k in sorted(opp_hands)]
        if completed_only:
            return [r for r in records if r.completed]
        return records

    @staticmethod
    def _infer_opponent_name(
        observation: Observation, fallback: str = "opponent"
    ) -> str:
        for rec in observation.history:
            if rec.player != observation.player:
                return rec.player
        return fallback

    def _detect_new_hand_from_observation(
        self,
        record: HandObservationRecord | None,
        observation: Observation,
    ) -> bool:
        """Detect when an observation comes from a new hand in automatic tournament play."""
        if record is None or record.completed:
            return True
        if len(observation.history) < record.processed_history_len:
            return True
        if (
            observation.street == "preflop"
            and len(observation.history) <= 1
            and record.processed_history_len > 1
        ):
            return True
        # Check if history prefix matches recorded actions
        for idx in range(min(len(observation.history), len(record.actions))):
            hist_item = observation.history[idx]
            logged = record.actions[idx]
            if (
                hist_item.street != logged.street
                or ActionType(hist_item.action.type).value != logged.action_type
                or hist_item.action.amount != logged.amount
            ):
                return True
        return False

    def ingest_observation(
        self,
        observation: Observation,
        *,
        opponent_id: str | None = None,
        hand_number: int | None = None,
    ) -> tuple[HandObservationRecord, list[ObservedActionEvent]]:
        """Extract newly observed public actions from an agent's `Observation`.

        Returns `(hand_record, newly_added_opponent_actions)`.
        Never accesses hidden opponent cards (`observation` only contains hero's own
        `hole_cards` and public `community_cards`).
        """
        opp_id = opponent_id or self._infer_opponent_name(
            observation, self.default_opponent_id
        )
        active = (
            self.get_hand(opp_id, hand_number)
            if hand_number is not None
            else self.get_active_hand(opp_id)
        )

        opponent_is_button = not observation.is_button
        total_chips_estimate = (
            observation.stack + observation.opponent_stack + observation.pot
        )
        starting_stack_est = max(observation.big_blind, total_chips_estimate // 2)

        if hand_number is None and self._detect_new_hand_from_observation(
            active, observation
        ):
            # Finalize previous unfinalized hand as incomplete/folded if it had actions
            if active is not None and not active.completed and active.actions:
                self._auto_close_previous_hand(active)
            active = self.start_hand(
                opponent_id=opp_id,
                hand_number=None,
                opponent_is_button=opponent_is_button,
                starting_stack=starting_stack_est,
                effective_stack=starting_stack_est,
                big_blind=observation.big_blind,
            )
        elif active is None:
            active = self.start_hand(
                opponent_id=opp_id,
                hand_number=hand_number,
                opponent_is_button=opponent_is_button,
                starting_stack=starting_stack_est,
                effective_stack=starting_stack_est,
                big_blind=observation.big_blind,
            )

        if len(observation.community_cards) >= 3:
            active.saw_flop = True
        if len(observation.community_cards) > len(active.community_cards):
            active.community_cards = tuple(observation.community_cards)

        new_opp_events = self._replay_unprocessed_history(
            active,
            observation.history,
            hero_player=observation.player,
        )
        return active, new_opp_events

    def _replay_unprocessed_history(
        self,
        record: HandObservationRecord,
        history: Sequence[ActionRecord],
        *,
        hero_player: str,
    ) -> list[ObservedActionEvent]:
        """Replay public action history from `record.processed_history_len` onward."""
        if len(history) <= record.processed_history_len:
            return []

        bb = max(1, record.big_blind)
        sb = max(1, bb // 2)
        start_stack = max(bb, record.starting_stack)

        # Reconstruct exact street-by-street pot, stacks, and contributions
        opp_button = record.opponent_is_button
        stacks = {
            "opp": start_stack - (sb if opp_button else bb),
            "hero": start_stack - (bb if opp_button else sb),
        }
        contribs = {
            "opp": sb if opp_button else bb,
            "hero": bb if opp_button else sb,
        }
        pot = contribs["opp"] + contribs["hero"]
        current_street = "preflop"
        street_raises = 0

        new_opponent_events: list[ObservedActionEvent] = []

        for idx, item in enumerate(history):
            if item.street != current_street:
                current_street = item.street
                contribs = {"opp": 0, "hero": 0}
                street_raises = 0
                if current_street in ("flop", "turn", "river"):
                    record.saw_flop = True

            is_opp = item.player != hero_player
            role_key = "opp" if is_opp else "hero"

            current_bet = max(contribs["opp"], contribs["hero"])
            to_call_before = max(0, current_bet - contribs[role_key])
            pot_before = pot
            stack_before = max(0, stacks[role_key])
            eff_stack_before = max(0, min(stacks["opp"], stacks["hero"]))
            facing_bet = to_call_before > 0
            facing_raises = street_raises

            kind = ActionType(item.action.type).value
            amt = item.action.amount
            paid = 0

            if kind == ActionType.FOLD.value:
                paid = 0
                if is_opp:
                    record.opponent_folded = True
                else:
                    record.hero_folded = True
            elif kind == ActionType.CHECK.value:
                paid = 0
            elif kind == ActionType.CALL.value:
                paid = min(stack_before, to_call_before)
            elif kind in (ActionType.BET.value, ActionType.RAISE.value):
                target = int(amt) if amt is not None else (current_bet + bb)
                paid = min(stack_before, max(0, target - contribs[role_key]))
                street_raises += 1
            elif kind == ActionType.ALL_IN.value:
                paid = stack_before
                if contribs[role_key] + paid > current_bet:
                    street_raises += 1

            stacks[role_key] = max(0, stacks[role_key] - paid)
            contribs[role_key] += paid
            pot += paid
            record.final_pot = pot

            if idx >= record.processed_history_len:
                if is_opp:
                    position = "button" if opp_button else "big_blind"
                    in_pos = not opp_button if item.street == "preflop" else opp_button
                else:
                    position = "big_blind" if opp_button else "button"
                    in_pos = opp_button if item.street == "preflop" else not opp_button

                denom = max(1, pot_before + to_call_before)
                wager_above_call = max(0, paid - to_call_before)
                pot_ratio = (
                    (wager_above_call / denom)
                    if kind
                    in (
                        ActionType.BET.value,
                        ActionType.RAISE.value,
                        ActionType.ALL_IN.value,
                    )
                    and wager_above_call > 0
                    else (paid / max(1, pot_before))
                )

                event = ObservedActionEvent(
                    hand_number=record.hand_number,
                    event_index=idx,
                    opponent_id=record.opponent_id,
                    actor=item.player,
                    is_opponent=is_opp,
                    street=item.street,
                    action_type=kind,
                    amount=amt,
                    paid_amount=paid,
                    pot_before=pot_before,
                    to_call_before=to_call_before,
                    stack_before=stack_before,
                    effective_stack_before=eff_stack_before,
                    position=position,
                    in_position=in_pos,
                    pot_ratio=pot_ratio,
                    facing_bet=facing_bet,
                    facing_raise_count=facing_raises,
                )
                record.actions.append(event)
                if is_opp:
                    new_opponent_events.append(event)

        record.processed_history_len = len(history)
        return new_opponent_events

    def record_action(
        self,
        *,
        opponent_id: str | None = None,
        hand_number: int | None = None,
        actor: str,
        is_opponent: bool,
        action: Action,
        street: str,
        pot_before: int,
        to_call_before: int,
        stack_before: int,
        effective_stack_before: int,
        opponent_is_button: bool,
        paid_amount: int | None = None,
    ) -> ObservedActionEvent:
        """Explicitly record a single observed action on the active or specified hand."""
        opp_id = opponent_id or self.default_opponent_id
        record = (
            self.get_hand(opp_id, hand_number)
            if hand_number is not None
            else self.get_active_hand(opp_id)
        )
        if record is None:
            record = self.start_hand(
                opponent_id=opp_id,
                hand_number=hand_number,
                opponent_is_button=opponent_is_button,
                starting_stack=max(stack_before, effective_stack_before, 200),
                effective_stack=max(effective_stack_before, 1),
            )

        if street in ("flop", "turn", "river"):
            record.saw_flop = True

        kind = ActionType(action.type).value
        if kind == ActionType.FOLD.value:
            if is_opponent:
                record.opponent_folded = True
            else:
                record.hero_folded = True

        prior_street_raises = sum(
            1
            for a in record.actions
            if a.street == street
            and a.action_type
            in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
        )
        paid = (
            int(paid_amount)
            if paid_amount is not None
            else (
                min(stack_before, to_call_before)
                if kind == ActionType.CALL.value
                else (
                    stack_before
                    if kind == ActionType.ALL_IN.value
                    else max(0, int(action.amount or 0))
                )
            )
        )
        denom = max(1, pot_before + to_call_before)
        wager_above_call = max(0, paid - to_call_before)
        pot_ratio = (
            wager_above_call / denom
            if kind
            in (
                ActionType.BET.value,
                ActionType.RAISE.value,
                ActionType.ALL_IN.value,
            )
            and wager_above_call > 0
            else paid / max(1, pot_before)
        )

        if is_opponent:
            position = "button" if record.opponent_is_button else "big_blind"
            in_pos = (
                not record.opponent_is_button
                if street == "preflop"
                else record.opponent_is_button
            )
        else:
            position = "big_blind" if record.opponent_is_button else "button"
            in_pos = (
                record.opponent_is_button
                if street == "preflop"
                else not record.opponent_is_button
            )

        event = ObservedActionEvent(
            hand_number=record.hand_number,
            event_index=len(record.actions),
            opponent_id=record.opponent_id,
            actor=actor,
            is_opponent=is_opponent,
            street=street,
            action_type=kind,
            amount=action.amount,
            paid_amount=paid,
            pot_before=pot_before,
            to_call_before=to_call_before,
            stack_before=stack_before,
            effective_stack_before=effective_stack_before,
            position=position,
            in_position=in_pos,
            pot_ratio=pot_ratio,
            facing_bet=to_call_before > 0,
            facing_raise_count=prior_street_raises,
        )
        record.actions.append(event)
        record.processed_history_len = len(record.actions)
        record.final_pot = max(record.final_pot, pot_before + paid)
        return event

    def _auto_close_previous_hand(self, record: HandObservationRecord) -> None:
        """Close an unfinalized hand conservatively without assuming showdown cards."""
        record.completed = True
        if not record.opponent_folded and not record.hero_folded:
            # Hand ended between observations without explicit showdown hook;
            # never fabricate showdown hole cards.
            record.reached_showdown = False
            record.revealed_hole_cards = None

    def finalize_hand(
        self,
        *,
        opponent_id: str | None = None,
        hand_number: int | None = None,
        history: Sequence[ActionRecord] | None = None,
        hero_player: str | None = None,
        community_cards: Sequence[Card] | None = None,
        reached_showdown: bool = False,
        revealed_opponent_cards: tuple[Card, Card] | None = None,
        opponent_net_chips: int = 0,
        opponent_won: bool | None = None,
        final_pot: int | None = None,
    ) -> HandObservationRecord:
        """Finalize a hand record with legitimate terminal outcome and showdown cards.

        Raises `ValueError` if `revealed_opponent_cards` is provided when
        `reached_showdown` is False.
        """
        if revealed_opponent_cards is not None and not reached_showdown:
            raise ValueError(
                "Privacy violation: cannot record opponent hole cards when hand "
                "did not reach showdown"
            )

        opp_id = opponent_id or self.default_opponent_id
        record = (
            self.get_hand(opp_id, hand_number)
            if hand_number is not None
            else self.get_active_hand(opp_id)
        )
        if record is None:
            record = self.start_hand(
                opponent_id=opp_id,
                hand_number=hand_number,
                opponent_is_button=False,
            )

        if history is not None and hero_player is not None:
            self._replay_unprocessed_history(
                record,
                history,
                hero_player=hero_player,
            )

        if community_cards is not None:
            record.community_cards = tuple(community_cards)
            if len(record.community_cards) >= 3:
                record.saw_flop = True

        if any(
            a.action_type == ActionType.FOLD.value and a.is_opponent
            for a in record.actions
        ):
            record.opponent_folded = True
        if any(
            a.action_type == ActionType.FOLD.value and not a.is_opponent
            for a in record.actions
        ):
            record.hero_folded = True

        if record.opponent_folded or record.hero_folded:
            if reached_showdown:
                raise ValueError(
                    "Hand with a fold cannot be marked as reached_showdown=True"
                )
            record.reached_showdown = False
            record.revealed_hole_cards = None
        else:
            record.reached_showdown = bool(reached_showdown)
            if record.reached_showdown and revealed_opponent_cards is not None:
                if len(revealed_opponent_cards) != 2:
                    raise ValueError("revealed_opponent_cards must contain 2 cards")
                record.revealed_hole_cards = (
                    revealed_opponent_cards[0],
                    revealed_opponent_cards[1],
                )
            else:
                record.revealed_hole_cards = None

        record.opponent_net_chips = int(opponent_net_chips)
        record.opponent_won = (
            bool(opponent_won)
            if opponent_won is not None
            else (record.opponent_net_chips > 0)
        )
        if final_pot is not None:
            record.final_pot = int(final_pot)
        record.completed = True
        return record

    def to_dict(self, opponent_id: str | None = None) -> dict[str, Any]:
        opp_id = opponent_id or self.default_opponent_id
        hands = self.list_hands(opp_id)
        return {
            "opponent_id": opp_id,
            "hand_counter": self._hand_counter_by_opponent.get(opp_id, 0),
            "active_hand": self._active_hand_by_opponent.get(opp_id),
            "hands": [h.to_dict() for h in hands],
        }

    def load_dict(self, payload: dict[str, Any]) -> str:
        opp_id = str(payload.get("opponent_id", self.default_opponent_id))
        hands_map: dict[int, HandObservationRecord] = {}
        max_num = int(payload.get("hand_counter", 0))
        for raw_hand in payload.get("hands", []):
            rec = HandObservationRecord.from_dict(raw_hand)
            hands_map[rec.hand_number] = rec
            max_num = max(max_num, rec.hand_number)
        self._hands_by_opponent[opp_id] = hands_map
        self._hand_counter_by_opponent[opp_id] = max_num
        active_hand = payload.get("active_hand")
        if active_hand is not None and int(active_hand) in hands_map:
            self._active_hand_by_opponent[opp_id] = int(active_hand)
        elif max_num in hands_map:
            self._active_hand_by_opponent[opp_id] = max_num
        return opp_id
