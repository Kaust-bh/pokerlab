"""Reproducible heads-up tournament matches with isolated player observations."""

import random
import time
from dataclasses import dataclass

from pokerlab.agents.base import ActionRecord, Agent, AgentFactory, observe
from pokerlab.engine import Deck, GameState


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    starting_stack: int = 500
    small_blind: int = 1
    big_blind: int = 2
    hand_limit: int = 500

    def __post_init__(self) -> None:
        if self.starting_stack <= 0 or self.hand_limit <= 0:
            raise ValueError("starting stack and hand limit must be positive")
        if self.small_blind <= 0 or self.big_blind < self.small_blind:
            raise ValueError("invalid blind sizes")


@dataclass(frozen=True, slots=True)
class MatchResult:
    player1: str
    player2: str
    starting_stacks: dict[str, int]
    ending_stacks: dict[str, int]
    net_chips: dict[str, int]
    hands_played: int
    hand_wins: dict[str, int]
    hand_ties: int
    match_winner: str | None
    positions: dict[str, int]
    runtime_seconds: float
    seed: int


class TournamentSimulator:
    """Registers agent factories and runs matches using independent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentFactory] = {}

    def register(self, name: str, factory: AgentFactory) -> None:
        if not name or name in self._agents:
            raise ValueError("agent name must be nonempty and unique")
        self._agents[name] = factory

    def run_match(
        self,
        player1: str,
        player2: str,
        *,
        config: TournamentConfig | None = None,
        seed: int = 0,
        first_button: str | None = None,
    ) -> MatchResult:
        if player1 not in self._agents or player2 not in self._agents:
            raise ValueError("both agents must be registered")
        if player1 == player2:
            raise ValueError("match participants need distinct registered names")
        if first_button is not None and first_button not in (player1, player2):
            raise ValueError("first_button must be one of the match participants")
        settings = config or TournamentConfig()
        names = (player1, player2)
        agents: dict[str, Agent] = {
            player1: self._agents[player1](),
            player2: self._agents[player2](),
        }
        seed_rng = random.Random(seed)
        for index, name in enumerate(names):
            agents[name].reset(seed_rng.randrange(2**32) + index)
        initial = {name: settings.starting_stack for name in names}
        stacks = dict(initial)
        hand_wins = {name: 0 for name in names}
        positions = {name: 0 for name in names}
        hand_ties = 0
        started = time.perf_counter()

        for hand_index in range(settings.hand_limit):
            if min(stacks.values()) <= 0:
                break
            button = first_button or player1
            if hand_index % 2:
                button = player2 if button == player1 else player1
            positions[button] += 1
            game = GameState(
                stacks,
                small_blind=settings.small_blind,
                big_blind=settings.big_blind,
                button=button,
                deck=Deck(seed=seed_rng.randrange(2**32)),
            )
            history: list[ActionRecord] = []
            actions_taken = 0
            while not game.complete:
                actor = game.current_player
                if actor is None:
                    raise RuntimeError("nonterminal hand has no player to act")
                observation = observe(game, actor, tuple(history))
                action = agents[actor].act(observation)
                if action not in observation.legal_actions:
                    raise ValueError(
                        f"agent {actor!r} returned an action outside its legal set"
                    )
                action_street = game.street.value
                game.act(action, player=actor)
                history.append(ActionRecord(actor, action, action_street))
                actions_taken += 1
                if actions_taken > 200:
                    raise RuntimeError("betting action limit exceeded")
            stacks = dict(game.stacks)
            if sum(stacks.values()) != sum(initial.values()):
                raise RuntimeError("chip conservation failed during the match")
            winners = [name for name, amount in game.winners.items() if amount > 0]
            if len(winners) == 1:
                hand_wins[winners[0]] += 1
            elif len(winners) > 1:
                hand_ties += 1

        if stacks[names[0]] == stacks[names[1]]:
            match_winner = None
        else:
            match_winner = max(names, key=lambda name: stacks[name])
        elapsed = time.perf_counter() - started
        return MatchResult(
            player1,
            player2,
            initial,
            stacks,
            {name: stacks[name] - initial[name] for name in names},
            sum(hand_wins.values()) + hand_ties,
            hand_wins,
            hand_ties,
            match_winner,
            positions,
            elapsed,
            seed,
        )

    def run_matches(
        self,
        player1: str,
        player2: str,
        *,
        matches: int = 10,
        config: TournamentConfig | None = None,
        seed: int = 0,
    ) -> list[MatchResult]:
        if matches < 1:
            raise ValueError("matches must be positive")
        seeds = random.Random(seed)
        return [
            self.run_match(
                player1,
                player2,
                config=config,
                seed=seeds.randrange(2**32),
                first_button=player1 if index % 2 == 0 else player2,
            )
            for index in range(matches)
        ]
