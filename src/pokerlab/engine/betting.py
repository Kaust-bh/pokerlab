"""No-limit betting round with short all-in raise reopening semantics."""

from dataclasses import dataclass
from enum import StrEnum


class ActionType(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True, slots=True)
class Action:
    type: ActionType | str
    amount: int | None = None


class BettingRound:
    """Tracks one street. Bet/raise amounts are target street contributions."""

    def __init__(
        self,
        stacks: dict[str, int],
        *,
        big_blind: int,
        start_player: str,
        contributions: dict[str, int] | None = None,
    ) -> None:
        if len(stacks) < 2 or any(stack < 0 for stack in stacks.values()):
            raise ValueError(
                "a betting round needs at least two players with nonnegative stacks"
            )
        if start_player not in stacks or big_blind <= 0:
            raise ValueError("invalid start player or big blind")
        self.players = list(stacks)
        self.stacks = dict(stacks)
        self.contributions = {p: (contributions or {}).get(p, 0) for p in stacks}
        self.current_bet = max(self.contributions.values(), default=0)
        self.min_raise = big_blind
        self.folded: set[str] = set()
        self.all_in: set[str] = {p for p, stack in self.stacks.items() if stack == 0}
        self.acted_since_full_raise: set[str] = set()
        self.current_player: str | None = start_player
        self.pending: set[str] = set(self.players) - self.folded - self.all_in
        self._normalize_turn(start_player)

    @property
    def complete(self) -> bool:
        return not self.pending or len(set(self.players) - self.folded) <= 1

    def _normalize_turn(self, preferred: str | None = None) -> None:
        live = [p for p in self.players if p in self.pending]
        if not live:
            self.current_player = None
            return
        start = self.players.index(preferred) if preferred in self.players else 0
        self.current_player = next(
            self.players[(start + i) % len(self.players)]
            for i in range(len(self.players))
            if self.players[(start + i) % len(self.players)] in self.pending
        )

    def _advance(self, after: str) -> None:
        self._normalize_turn(
            self.players[(self.players.index(after) + 1) % len(self.players)]
        )

    def act(
        self, player: str, action: Action | ActionType | str, amount: int | None = None
    ) -> None:
        """Apply a legal action or raise ValueError without mutating state."""
        if isinstance(action, Action):
            kind, amount = ActionType(action.type), action.amount
        else:
            kind = ActionType(action)
        if self.complete or player != self.current_player:
            raise ValueError("it is not this player's turn")
        if player not in self.pending or player in self.folded or player in self.all_in:
            raise ValueError("player cannot act in this betting round")
        owed = self.current_bet - self.contributions[player]
        stack = self.stacks[player]
        target: int | None = amount
        if kind == ActionType.FOLD:
            self.folded.add(player)
        elif kind == ActionType.CHECK:
            if owed:
                raise ValueError("cannot check while facing a bet")
        elif kind == ActionType.CALL:
            if not owed:
                raise ValueError("nothing to call; check instead")
        elif kind == ActionType.BET:
            target = stack + self.contributions[player] if amount is None else amount
            if (
                self.current_bet != 0
                or target <= 0
                or target <= self.contributions[player]
            ):
                raise ValueError("bet is only legal when no bet is outstanding")
            if (
                target - self.contributions[player] < self.min_raise
                and target - self.contributions[player] < stack
            ):
                raise ValueError("bet is below the minimum opening bet")
            if target - self.contributions[player] > stack:
                raise ValueError("bet exceeds player's stack")
        elif kind == ActionType.RAISE:
            target = amount
            if target is None or target <= self.current_bet:
                raise ValueError("raise target must exceed the current bet")
            if player in self.acted_since_full_raise:
                raise ValueError("action has not been reopened for this player")
            if target - self.contributions[player] > stack:
                raise ValueError("raise exceeds player's stack")
            increase = target - self.current_bet
            if (
                increase < self.min_raise
                and target - self.contributions[player] != stack
            ):
                raise ValueError("raise is below the minimum raise")
        elif kind == ActionType.ALL_IN:
            target = self.contributions[player] + stack
            if target > self.current_bet and player in self.acted_since_full_raise:
                raise ValueError("action has not been reopened for this player")
        if kind in (ActionType.BET, ActionType.RAISE) and target is not None:
            paid = target - self.contributions[player]
        elif kind in (ActionType.CALL, ActionType.ALL_IN):
            paid = min(stack, owed) if kind == ActionType.CALL else stack
        else:
            paid = 0
        if paid < 0 or paid > stack:
            raise ValueError("action exceeds player's stack")

        old_bet = self.current_bet
        self.stacks[player] -= paid
        self.contributions[player] += paid
        if self.stacks[player] == 0:
            self.all_in.add(player)
        if kind == ActionType.FOLD:
            self.pending.discard(player)
        elif (
            kind in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
            and self.contributions[player] > old_bet
        ):
            new_bet = self.contributions[player]
            increase = new_bet - old_bet
            self.current_bet = new_bet
            self.pending.discard(player)
            if increase >= self.min_raise:
                self.min_raise = increase
                self.acted_since_full_raise = {player}
                self.pending = set(self.players) - self.folded - self.all_in - {player}
            else:
                self.acted_since_full_raise.add(player)
                self.pending.update(
                    p
                    for p in self.players
                    if p != player
                    and p not in self.folded
                    and p not in self.all_in
                    and self.contributions[p] < self.current_bet
                )
        else:
            self.pending.discard(player)
            self.acted_since_full_raise.add(player)
        self._advance(player)
