"""Reproducible heads-up tournament matches with isolated player observations."""

import random
import time
from dataclasses import dataclass

from pokerlab.agents.action_profiles import ACTION_PROFILES
from pokerlab.agents.base import ActionRecord, Agent, AgentFactory, observe
from pokerlab.engine import Deck, GameState


@dataclass(frozen=True, slots=True)
class TournamentConfig:
    starting_stack: int = 500
    small_blind: int = 1
    big_blind: int = 2
    hand_limit: int = 500
    action_profile: str = "legacy"

    def __post_init__(self) -> None:
        if self.starting_stack <= 0 or self.hand_limit <= 0:
            raise ValueError("starting stack and hand limit must be positive")
        if self.small_blind <= 0 or self.big_blind < self.small_blind:
            raise ValueError("invalid blind sizes")
        if self.action_profile not in ACTION_PROFILES:
            raise ValueError(f"unknown action profile {self.action_profile!r}")


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


@dataclass(frozen=True, slots=True)
class FixedHandBatchResult:
    """Independent fresh-stack hands used for paired policy evaluation."""

    player1: str
    player2: str
    net_chips: dict[str, int]
    hands_played: int
    hand_wins: dict[str, int]
    hand_ties: int
    positions: dict[str, int]
    hand_net_chips: tuple[int, ...]
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
                observation = observe(
                    game,
                    actor,
                    tuple(history),
                    action_profile=settings.action_profile,
                )
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

    def run_fixed_hands(
        self,
        player1: str,
        player2: str,
        *,
        hands: int,
        config: TournamentConfig | None = None,
        seed: int = 0,
        button: str | None = None,
    ) -> FixedHandBatchResult:
        """Evaluate independent hands with fresh equal stacks and a fixed role.

        This avoids tournament elimination truncating the hand sample. The
        selected button seat is held constant within the batch so the caller
        can pair separate button-position batches.
        """
        if hands < 1:
            raise ValueError("hands must be positive")
        if button is not None and button not in (player1, player2):
            raise ValueError("button must be one of the match participants")
        if player1 not in self._agents or player2 not in self._agents:
            raise ValueError("both agents must be registered")
        if player1 == player2:
            raise ValueError("hand participants need distinct registered names")
        settings = config or TournamentConfig()
        names = (player1, player2)
        agents: dict[str, Agent] = {
            player1: self._agents[player1](),
            player2: self._agents[player2](),
        }
        seed_rng = random.Random(seed)
        for index, name in enumerate(names):
            agents[name].reset(seed_rng.randrange(2**32) + index)
        button_name = button or player1
        net_chips = dict.fromkeys(names, 0)
        hand_wins = dict.fromkeys(names, 0)
        hand_ties = 0
        positions = {player1: 0, player2: 0}
        positions[button_name] = hands
        hand_net = []
        started = time.perf_counter()
        for _ in range(hands):
            initial = {name: settings.starting_stack for name in names}
            game = GameState(
                initial,
                small_blind=settings.small_blind,
                big_blind=settings.big_blind,
                button=button_name,
                deck=Deck(seed=seed_rng.randrange(2**32)),
            )
            history: list[ActionRecord] = []
            actions_taken = 0
            while not game.complete:
                actor = game.current_player
                if actor is None:
                    raise RuntimeError("nonterminal hand has no player to act")
                observation = observe(
                    game,
                    actor,
                    tuple(history),
                    action_profile=settings.action_profile,
                )
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
            if sum(game.stacks.values()) != sum(initial.values()):
                raise RuntimeError(
                    "chip conservation failed during fixed-hand evaluation"
                )
            delta = {name: game.stacks[name] - initial[name] for name in names}
            for name in names:
                net_chips[name] += delta[name]
            hand_net.append(delta[player1])
            if delta[player1] > 0:
                hand_wins[player1] += 1
            elif delta[player2] > 0:
                hand_wins[player2] += 1
            else:
                hand_ties += 1
        return FixedHandBatchResult(
            player1=player1,
            player2=player2,
            net_chips=net_chips,
            hands_played=hands,
            hand_wins=hand_wins,
            hand_ties=hand_ties,
            positions=positions,
            hand_net_chips=tuple(hand_net),
            runtime_seconds=time.perf_counter() - started,
            seed=seed,
        )
