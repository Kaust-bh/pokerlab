"""Heads-up no-limit Hold'em hand state and transitions."""

from enum import StrEnum

from pokerlab.engine.betting import Action, ActionType, BettingRound
from pokerlab.engine.cards import Card
from pokerlab.engine.deck import Deck
from pokerlab.engine.hand_evaluator import HandRank, evaluate_hand
from pokerlab.engine.pot import Pot


class Street(StrEnum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    COMPLETE = "complete"


class GameState:
    """A complete heads-up hand. Bets and raises specify target street totals."""

    def __init__(
        self,
        stacks: dict[str, int],
        *,
        small_blind: int = 1,
        big_blind: int = 2,
        button: str | None = None,
        deck: Deck | None = None,
    ) -> None:
        if len(stacks) != 2 or any(amount <= 0 for amount in stacks.values()):
            raise ValueError(
                "heads-up game requires exactly two players with positive stacks"
            )
        if small_blind <= 0 or big_blind < small_blind:
            raise ValueError(
                "blinds must be positive and big blind at least small blind"
            )
        self.players = list(stacks)
        if button is not None and button not in stacks:
            raise ValueError("button must identify a player")
        self.button = button or self.players[0]
        self.small_blind, self.big_blind = small_blind, big_blind
        self.stacks = dict(stacks)
        self.deck = deck or Deck()
        self.hole_cards: dict[str, tuple[Card, Card]] = {}
        dealt: dict[str, list[Card]] = {player: [] for player in self.players}
        for _ in range(2):
            for player in self.players:
                card = self.deck.draw()
                assert isinstance(card, Card)
                dealt[player].append(card)
        self.hole_cards = {
            player: (cards[0], cards[1]) for player, cards in dealt.items()
        }
        self.community_cards: list[Card] = []
        self.pot = Pot()
        self.street = Street.PREFLOP
        self.folded: set[str] = set()
        self.all_in: set[str] = set()
        self.winners: dict[str, int] = {}
        self.hand_ranks: dict[str, HandRank] = {}
        other = self.players[1 - self.players.index(self.button)]
        self._post_blind(self.button, small_blind)
        self._post_blind(other, big_blind)
        self.betting = BettingRound(
            self.stacks,
            big_blind=big_blind,
            start_player=self.button,
            contributions=self.pot.contributions,
        )
        self.current_player = self.betting.current_player
        if self.betting.complete:
            self._advance_street()

    def _post_blind(self, player: str, amount: int) -> None:
        paid = min(self.stacks[player], amount)
        self.stacks[player] -= paid
        self.pot.contribute(player, paid)
        if self.stacks[player] == 0:
            self.all_in.add(player)

    @property
    def complete(self) -> bool:
        return self.street == Street.COMPLETE

    @property
    def current_pot(self) -> int:
        return self.pot.total

    def act(
        self,
        action: Action | ActionType | str,
        amount: int | None = None,
        *,
        player: str | None = None,
    ) -> None:
        """Apply the current player's action and advance when the round is complete."""
        if self.complete or self.street in (Street.SHOWDOWN, Street.COMPLETE):
            raise ValueError("hand is already complete")
        actor = self.current_player if player is None else player
        if actor is None:
            raise ValueError("no player can act")
        if actor not in self.players:
            raise ValueError("unknown player")
        before = self.betting.contributions[actor]
        self.betting.act(actor, action, amount)
        after = self.betting.contributions[actor]
        if after > before:
            self.pot.contribute(actor, after - before)
        self.stacks = dict(self.betting.stacks)
        self.folded = set(self.betting.folded)
        self.all_in = set(self.betting.all_in)
        self.current_player = self.betting.current_player
        if len(self.players) - len(self.folded) == 1:
            winner = next(p for p in self.players if p not in self.folded)
            self.winners = {winner: self.pot.total}
            self.stacks[winner] += self.pot.total
            self.street = Street.COMPLETE
        elif self.betting.complete:
            self._advance_street()

    def _advance_street(self) -> None:
        order = [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]
        while True:
            if self.street == Street.RIVER:
                self._showdown()
                return
            self.street = order[order.index(self.street) + 1]
            count = {Street.FLOP: 3, Street.TURN: 1, Street.RIVER: 1}[self.street]
            drawn = self.deck.draw(count)
            self.community_cards.extend(drawn if isinstance(drawn, list) else [drawn])
            if self.street == Street.RIVER or self.all_in:
                if self.street == Street.RIVER:
                    self._showdown()
                    return
                continue
            start = self.players[1 - self.players.index(self.button)]
            self.betting = BettingRound(
                self.stacks, big_blind=self.big_blind, start_player=start
            )
            self.current_player = self.betting.current_player
            if self.betting.complete:
                continue
            return

    def _showdown(self) -> None:
        self.street = Street.SHOWDOWN
        self.hand_ranks = {
            p: evaluate_hand((*self.hole_cards[p], *self.community_cards))
            for p in self.players
            if p not in self.folded
        }
        self.winners = self.pot.distribute(self.hand_ranks, self.players)
        for player, amount in self.winners.items():
            self.stacks[player] += amount
        self.street = Street.COMPLETE

    def showdown(self) -> dict[str, int]:
        """Run out any remaining community cards and settle a contested hand."""
        if len(self.players) - len(self.folded) <= 1:
            return dict(self.winners)
        if (
            self.street == Street.PREFLOP
            or self.street == Street.FLOP
            or self.street == Street.TURN
            or self.street == Street.RIVER
        ):
            while len(self.community_cards) < 5:
                count = 3 if len(self.community_cards) == 0 else 1
                drawn = self.deck.draw(count)
                self.community_cards.extend(
                    drawn if isinstance(drawn, list) else [drawn]
                )
            self._showdown()
        return dict(self.winners)
