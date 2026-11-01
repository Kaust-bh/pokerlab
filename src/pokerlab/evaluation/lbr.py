"""Local Best Response (LBR) evaluation framework.

Implements:
1. Exact unabstracted Bayesian range tracking over all 1,225 Hold'em starting hands
   using non-mutating action-probability queries (`Agent.action_probabilities`).
2. Independent forward-tree Local Best Response evaluators for Kuhn Poker and Leduc
   Poker (`evaluate_lbr_kuhn`, `evaluate_lbr_leduc`) that compute Bayesian posterior
   beliefs over hidden opponent cards and evaluate candidate actions without calling
   the exact tabular best-response solvers.
3. A Monte Carlo 1-step Local Best Response probe for Heads-Up No-Limit Hold'em
   (`LocalBestResponseAgent`, `evaluate_lbr_hunl`) inspired by Lisý & Bowling
   (IJCAI 2017). Because HUNL evaluation uses sampled posterior hands, sampled
   board runouts on preflop/flop/turn, and check/call-to-showdown continuation
   beyond the immediate opponent response, `evaluate_lbr_hunl` is a practical
   local exploitability probe rather than a formal exact exploitability bound.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

from pokerlab.agents.base import (
    Action,
    ActionRecord,
    ActionType,
    Agent,
    Observation,
    observe,
)
from pokerlab.engine import Card, Deck, GameState, evaluate_hand
from pokerlab.evaluation.tournament import TournamentConfig


@dataclass(frozen=True, slots=True)
class LBRResult:
    """Statistical summary of Local Best Response evaluation."""

    game: str
    target_agent: str
    hands_evaluated: int
    lbr_net_chips_mean: float
    lbr_bb_per_100: float
    lbr_mbb_per_hand: float
    standard_error_mbb: float
    ci95_low_mbb: float
    ci95_high_mbb: float
    runtime_seconds: float


class BayesianRangeTracker:
    """Maintains an exact unabstracted posterior distribution over opponent hole cards."""

    def __init__(
        self, hero_hole: tuple[Card, Card], dead_cards: Sequence[Card] = ()
    ) -> None:
        self.hero_hole = set(hero_hole)
        self.dead_cards = set(dead_cards)
        self.deck = tuple(
            Card(r, s)
            for s in ("clubs", "diamonds", "hearts", "spades")
            for r in range(2, 15)
        )
        available = [
            c for c in self.deck if c not in self.hero_hole and c not in self.dead_cards
        ]
        self.weights: dict[tuple[Card, Card], float] = {
            tuple(sorted(comb)): 1.0 for comb in combinations(available, 2)
        }

    def reveal_board_cards(self, cards: Sequence[Card]) -> None:
        """Zero out any holding that clashes with newly dealt public cards."""
        new_dead = set(cards) - self.dead_cards
        if not new_dead:
            return
        self.dead_cards.update(new_dead)
        for hand in list(self.weights):
            if hand[0] in new_dead or hand[1] in new_dead:
                self.weights[hand] = 0.0

    def update_beliefs(
        self,
        observation: Observation,
        action: Action,
        policy_fn: Callable[[Observation], Mapping[Action, float] | Action],
    ) -> None:
        """Bayesian update: weight(h) <- weight(h) * Pr(action | observation with hole=h)."""
        self.reveal_board_cards(observation.community_cards)
        total = sum(self.weights.values())
        if total <= 0:
            return

        updated: dict[tuple[Card, Card], float] = {}
        for hand, prior_weight in self.weights.items():
            if prior_weight <= 0:
                updated[hand] = 0.0
                continue
            hypothetical_obs = replace(observation, hole_cards=hand)
            resp = policy_fn(hypothetical_obs)
            if isinstance(resp, Mapping):
                prob = max(0.0, float(resp.get(action, 0.0)))
            elif isinstance(resp, Action):
                prob = 1.0 if resp == action else 0.0
            else:
                prob = 0.0
            updated[hand] = prior_weight * prob

        new_total = sum(updated.values())
        if new_total > 0:
            for hand, value in updated.items():
                self.weights[hand] = value / new_total
        else:
            # Zero-probability observation off-tree: retain prior over non-dead hands
            for hand in self.weights:
                self.weights[hand] /= total


class LocalBestResponseAgent(Agent):
    """Monte Carlo 1-step Local Best Response agent for Heads-Up No-Limit Hold'em.

    At each decision point, evaluates candidate actions (Fold, Check/Call, and
    representative Bet/Raise/All-In sizes) relative to folding right now
    (`EV(FOLD) = 0`), accounting for:
    - Sunk vs. incremental pot contributions (`observation.street_contribution`),
    - Effective callable stack risk (including `ActionType.ALL_IN`),
    - Opponent fold/continue probabilities via `target_agent.action_probabilities`,
    - Sampled 5-card board runouts on preflop, flop, and turn.
    """

    def __init__(
        self,
        target_agent: Agent,
        seed: int = 0,
        *,
        opponent_samples: int = 16,
        board_rollouts: int = 6,
    ) -> None:
        self.target_agent = target_agent
        self.rng = random.Random(seed)
        self.opponent_samples = opponent_samples
        self.board_rollouts = board_rollouts
        self.tracker: BayesianRangeTracker | None = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self.tracker = None

    def initialize_hand(self, hero_hole: tuple[Card, Card]) -> None:
        self.tracker = BayesianRangeTracker(hero_hole)

    def notify_action(self, obs: Observation, action: Action) -> None:
        if self.tracker is not None:
            self.tracker.reveal_board_cards(obs.community_cards)
            self.tracker.update_beliefs(
                obs, action, self.target_agent.action_probabilities
            )

    def _rollout_equity(
        self,
        hero_hole: tuple[Card, Card],
        opp_hole: tuple[Card, Card],
        board: tuple[Card, ...],
        deck: tuple[Card, ...],
    ) -> float:
        needed = 5 - len(board)
        if needed <= 0:
            h_rank = evaluate_hand((*hero_hole, *board))
            o_rank = evaluate_hand((*opp_hole, *board))
            return 1.0 if h_rank > o_rank else (0.5 if h_rank == o_rank else 0.0)

        dead = {*hero_hole, *opp_hole, *board}
        pool = [c for c in deck if c not in dead]
        wins = 0.0
        for _ in range(self.board_rollouts):
            runout = self.rng.sample(pool, needed)
            full_board = (*board, *runout)
            h_rank = evaluate_hand((*hero_hole, *full_board))
            o_rank = evaluate_hand((*opp_hole, *full_board))
            if h_rank > o_rank:
                wins += 1.0
            elif h_rank == o_rank:
                wins += 0.5
        return wins / self.board_rollouts

    @staticmethod
    def _opponent_response_observation(
        observation: Observation,
        opp_hole: tuple[Card, Card],
        hero_action: Action,
        hero_target_contribution: int,
    ) -> Observation:
        """Construct the hypothetical observation faced by the opponent after hero_action."""
        c_hero = observation.street_contribution
        c_opp = observation.opponent_street_contribution
        hero_paid = max(0, hero_target_contribution - c_hero)
        new_pot = observation.pot + hero_paid
        opp_to_call = max(0, hero_target_contribution - c_opp)
        opp_stack = observation.opponent_stack
        hero_new_stack = max(0, observation.stack - hero_paid)
        increase = max(observation.min_raise, hero_target_contribution - c_opp)

        opp_legal: list[Action] = [
            Action(ActionType.FOLD),
            Action(ActionType.CALL),
        ]
        if opp_stack > opp_to_call:
            max_opp_target = c_opp + opp_stack
            min_raise_target = hero_target_contribution + increase
            if max_opp_target > min_raise_target:
                opp_legal.append(Action(ActionType.RAISE, min_raise_target))
            opp_legal.append(Action(ActionType.ALL_IN))

        opp_name = "Target" if observation.player == "LBR" else "p1"
        new_history = (
            *observation.history,
            ActionRecord(observation.player, hero_action, observation.street),
        )
        return Observation(
            player=opp_name,
            hole_cards=opp_hole,
            community_cards=observation.community_cards,
            street=observation.street,
            in_position=not observation.in_position,
            stack=opp_stack,
            opponent_stack=hero_new_stack,
            pot=new_pot,
            to_call=opp_to_call,
            big_blind=observation.big_blind,
            min_raise=increase,
            legal_actions=tuple(opp_legal),
            history=new_history,
            street_contribution=c_opp,
            opponent_street_contribution=hero_target_contribution,
            is_button=not observation.is_button,
            action_profile=observation.action_profile,
        )

    def candidate_action_evs(self, observation: Observation) -> dict[Action, float]:
        """Compute local 1-step lookahead incremental EVs relative to folding (EV(FOLD)=0)."""
        if self.tracker is None:
            self.initialize_hand(observation.hole_cards)
        assert self.tracker is not None
        self.tracker.reveal_board_cards(observation.community_cards)

        legal = observation.legal_actions
        check_act = next((a for a in legal if a.type == ActionType.CHECK), None)
        call_act = next((a for a in legal if a.type == ActionType.CALL), None)
        fold_act = next((a for a in legal if a.type == ActionType.FOLD), None)
        bets = [
            a
            for a in legal
            if a.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
        ]

        candidates: list[Action] = []
        if fold_act is not None:
            candidates.append(fold_act)
        passive = check_act or call_act
        if passive is not None:
            candidates.append(passive)
        if bets:
            candidates.append(bets[0])
            if len(bets) > 2:
                candidates.append(bets[len(bets) // 2])
            if len(bets) > 1:
                candidates.append(bets[-1])

        active = [(h, w) for h, w in self.tracker.weights.items() if w > 0]
        if not active:
            return {c: 0.0 for c in candidates}

        if len(active) <= self.opponent_samples:
            sampled = active
        else:
            hands = [h for h, _ in active]
            weights = [w for _, w in active]
            chosen = self.rng.choices(hands, weights=weights, k=self.opponent_samples)
            counts: dict[tuple[Card, Card], int] = defaultdict(int)
            for h in chosen:
                counts[h] += 1
            sampled = [(h, counts[h] / self.opponent_samples) for h in counts]

        weight_sum = sum(w for _, w in sampled)
        sampled = [(h, w / weight_sum) for h, w in sampled]

        equities = {
            h: self._rollout_equity(
                observation.hole_cards,
                h,
                observation.community_cards,
                self.tracker.deck,
            )
            for h, _ in sampled
        }

        pot0 = float(observation.pot)
        c_hero = observation.street_contribution
        c_opp = observation.opponent_street_contribution
        s_hero = observation.stack
        s_opp = observation.opponent_stack
        uncallable_opp_excess = max(0, c_opp - (c_hero + s_hero))
        pot_eff = float(pot0 - uncallable_opp_excess)

        evs: dict[Action, float] = {}
        for cand in candidates:
            if cand.type == ActionType.FOLD:
                evs[cand] = 0.0
                continue
            if cand.type == ActionType.CHECK:
                evs[cand] = sum(w * equities[h] * pot0 for h, w in sampled)
                continue
            if cand.type == ActionType.CALL:
                risk_call = float(min(observation.to_call, s_hero))
                evs[cand] = sum(
                    w * (equities[h] * pot_eff - (1.0 - equities[h]) * risk_call)
                    for h, w in sampled
                )
                continue

            # BET, RAISE, or ALL_IN
            raw_target = (
                c_hero + s_hero
                if cand.type == ActionType.ALL_IN or cand.amount is None
                else int(cand.amount)
            )
            eff_target = min(raw_target, c_opp + s_opp)
            if eff_target <= c_opp:
                risk = float(min(s_hero, max(0, c_opp - c_hero)))
                evs[cand] = sum(
                    w * (equities[h] * pot_eff - (1.0 - equities[h]) * risk)
                    for h, w in sampled
                )
                continue

            risk_bet = float(eff_target - c_hero)
            delta_b = float(eff_target - c_opp)
            cand_ev = 0.0
            for h, w in sampled:
                opp_obs = self._opponent_response_observation(
                    observation, h, cand, raw_target
                )
                opp_dist = self.target_agent.action_probabilities(opp_obs)
                p_fold = max(
                    0.0,
                    min(1.0, float(opp_dist.get(Action(ActionType.FOLD), 0.0))),
                )
                eq = equities[h]
                continue_ev = eq * (pot0 + delta_b) - (1.0 - eq) * risk_bet
                cand_ev += w * (p_fold * pot0 + (1.0 - p_fold) * continue_ev)
            evs[cand] = cand_ev

        return evs

    def act(self, observation: Observation) -> Action:
        legal = observation.legal_actions
        if len(legal) == 1:
            return legal[0]
        evs = self.candidate_action_evs(observation)
        return max(evs, key=lambda a: evs[a])


def evaluate_lbr_kuhn(
    profile: Mapping[Any, Mapping[Any, float]], player: int = 0
) -> float:
    """Independent forward Bayesian range-tracking LBR evaluation on Kuhn Poker.

    At each decision node for ``player``, reconstructs the exact Bayesian
    posterior over the opponent's private card given observed opponent actions
    and selects the action maximizing 1-step local lookahead expected value.
    Does not call `pokerlab.research.kuhn.evaluation.best_response_value`.
    """
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    from pokerlab.research.kuhn.game import (
        KuhnAction,
        KuhnCard,
        KuhnGame,
        KuhnState,
    )
    from pokerlab.research.kuhn.information_sets import InformationSet

    def opp_dist(
        info: InformationSet, legal: tuple[KuhnAction, ...]
    ) -> dict[KuhnAction, float]:
        supplied = profile.get(info)
        if supplied is None:
            return {a: 1.0 / len(legal) for a in legal}
        return {a: float(supplied[a]) for a in legal}

    def posterior_weights(
        hero_card, history: tuple[KuhnAction, ...]
    ) -> dict[Any, float]:
        candidates = [c for c in KuhnCard if c != hero_card]
        weights = {c: 0.5 for c in candidates}
        for step_idx, act in enumerate(history):
            actor = step_idx % 2
            if actor != player:
                opp_card_player = 1 - player
                for c in candidates:
                    opp_info = InformationSet(opp_card_player, c, history[:step_idx])
                    prefix_deal = (hero_card, c) if player == 0 else (c, hero_card)
                    prefix_state = KuhnGame.initial_state(prefix_deal)
                    for prev_a in history[:step_idx]:
                        prefix_state = prefix_state.act(prev_a)
                    probs = opp_dist(opp_info, prefix_state.legal_actions)
                    weights[c] *= probs.get(act, 0.0)
        return weights

    def local_action_ev(
        hero_card, history: tuple[KuhnAction, ...], action: KuhnAction
    ) -> float:
        weights = posterior_weights(hero_card, history)
        total_w = sum(weights.values())
        if total_w <= 0:
            weights = {c: 0.5 for c in KuhnCard if c != hero_card}
            total_w = 1.0
        ev = 0.0
        for opp_card, w in weights.items():
            if w <= 0:
                continue
            deal = (hero_card, opp_card) if player == 0 else (opp_card, hero_card)
            state = KuhnGame.initial_state(deal)
            for prev_a in history:
                state = state.act(prev_a)
            next_state = state.act(action)
            if next_state.is_terminal:
                ev += (w / total_w) * float(next_state.utility(player))
            else:
                # 1-step opponent response lookahead
                opp_info = InformationSet.from_state(next_state)
                probs = opp_dist(opp_info, next_state.legal_actions)
                branch_val = 0.0
                for opp_a, p in probs.items():
                    after_opp = next_state.act(opp_a)
                    if after_opp.is_terminal:
                        branch_val += p * float(after_opp.utility(player))
                    else:
                        # Player 0 checked, Player 1 bet -> Player 0 faces CALL/FOLD
                        sub_evs = [
                            float(after_opp.act(sub_a).utility(player))
                            for sub_a in after_opp.legal_actions
                        ]
                        branch_val += p * max(sub_evs)
                ev += (w / total_w) * branch_val
        return ev

    def traverse(state: KuhnState) -> float:
        if state.is_terminal:
            return float(state.utility(player))
        actor = state.current_player
        if actor == player:
            hero_card = state.cards[player]
            best_a = max(
                state.legal_actions,
                key=lambda a: local_action_ev(hero_card, state.history, a),
            )
            return traverse(state.act(best_a))
        info = InformationSet.from_state(state)
        probs = opp_dist(info, state.legal_actions)
        return sum(p * traverse(state.act(a)) for a, p in probs.items())

    deals = tuple(KuhnGame.deals())
    return sum(traverse(KuhnGame.initial_state(d)) for d in deals) / len(deals)


def evaluate_lbr_leduc(
    profile: Mapping[Any, Mapping[Any, float]],
    player: int = 0,
    *,
    mode: str = "local_br",
) -> float:
    """Independent forward Bayesian range-tracking LBR evaluation on Leduc Poker.

    Maintains the exact Bayesian posterior distribution over the remaining 5
    deck cards (preflop) or 4 deck cards (flop) for the opponent's private card
    given all observed opponent actions, without calling `leduc.evaluation.best_response_value`.

    Parameters
    ----------
    profile:
        Target strategy profile mapping Leduc `InformationSet` to action probabilities.
    player:
        Responder index (0 or 1).
    mode:
        - ``"local_br"`` (default): evaluates candidate actions using posterior
          beliefs with recursive local best-response continuation on future hero decisions.
        - ``"check_call_rollout"``: evaluates candidate actions using 1-step
          opponent response on the current street and passive check/call continuation
          on subsequent streets (Lisý & Bowling local rollout lower bound).
    """
    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    if mode not in {"local_br", "check_call_rollout"}:
        raise ValueError("mode must be 'local_br' or 'check_call_rollout'")

    from pokerlab.research.leduc.betting import Action as LeducAction
    from pokerlab.research.leduc.cards import DECK
    from pokerlab.research.leduc.game import LeducGame, LeducState
    from pokerlab.research.leduc.information_sets import InformationSet

    def opp_dist(
        info: InformationSet, legal: tuple[LeducAction, ...]
    ) -> dict[LeducAction, float]:
        supplied = profile.get(info)
        if supplied is None:
            return {a: 1.0 / len(legal) for a in legal}
        return {a: float(supplied[a]) for a in legal}

    def replay_actions(
        deal, public_card, histories: tuple[tuple[LeducAction, ...], ...]
    ) -> LeducState | None:
        state = LeducGame.initial_state(deal)
        for act in histories[0]:
            if state.is_terminal or state.is_chance_node:
                return None
            if act not in state.legal_actions:
                return None
            state = state.act(act)
        if len(histories) > 1 and public_card is not None:
            if not state.is_chance_node:
                return None
            state = state.deal_public(public_card)
            for act in histories[1]:
                if state.is_terminal or state.is_chance_node:
                    return None
                if act not in state.legal_actions:
                    return None
                state = state.act(act)
        return state

    def posterior_states(info: InformationSet) -> list[tuple[LeducState, float]]:
        hero_card = info.private_card
        public_card = info.public_card
        opp_player = 1 - player

        # Enumerate remaining physical cards in the 6-card Leduc deck
        remaining_cards = [
            c
            for c in DECK
            if c != hero_card and (public_card is None or c != public_card)
        ]
        weighted_states: list[tuple[LeducState, float]] = []
        for opp_card in remaining_cards:
            deal = (hero_card, opp_card) if player == 0 else (opp_card, hero_card)
            state = LeducGame.initial_state(deal)
            weight = 1.0
            valid = True
            for act in info.histories[0]:
                if state.is_terminal or state.is_chance_node:
                    valid = False
                    break
                if state.current_player == opp_player:
                    opp_info = InformationSet.from_state(state)
                    probs = opp_dist(opp_info, state.legal_actions)
                    weight *= probs.get(act, 0.0)
                state = state.act(act)
            if not valid:
                continue
            if info.round_index == 1:
                if not state.is_chance_node or public_card is None:
                    continue
                chance_dict = dict(state.chance_outcomes)
                if public_card not in chance_dict:
                    continue
                weight *= chance_dict[public_card]
                state = state.deal_public(public_card)
                for act in info.histories[1]:
                    if state.is_terminal or state.is_chance_node:
                        valid = False
                        break
                    if state.current_player == opp_player:
                        opp_info = InformationSet.from_state(state)
                        probs = opp_dist(opp_info, state.legal_actions)
                        weight *= probs.get(act, 0.0)
                    state = state.act(act)
            if valid:
                weighted_states.append((state, weight))
        total_w = sum(w for _, w in weighted_states)
        if total_w > 0:
            return [(s, w / total_w) for s, w in weighted_states if w > 0]
        # Uniform fallback over structurally valid states if off-tree (zero reach)
        n = len(weighted_states)
        return [(s, 1.0 / n) for s, _ in weighted_states] if n else []

    memo_action: dict[InformationSet, LeducAction] = {}

    def rollout_value(state: LeducState, active_street: int) -> float:
        if state.is_terminal:
            val0 = float(state.utility(0))
            return val0 if player == 0 else -val0
        if state.is_chance_node:
            return sum(
                p * rollout_value(state.deal_public(card), active_street)
                for card, p in state.chance_outcomes
            )
        actor = state.current_player
        info = InformationSet.from_state(state)
        if actor == player:
            if mode == "check_call_rollout" and info.round_index > active_street:
                passive = (
                    LeducAction.CHECK
                    if LeducAction.CHECK in state.legal_actions
                    else LeducAction.CALL
                )
                return rollout_value(state.act(passive), active_street)
            chosen = choose_local_action(info)
            return rollout_value(state.act(chosen), active_street)
        probs = opp_dist(info, state.legal_actions)
        return sum(
            p * rollout_value(state.act(a), active_street) for a, p in probs.items()
        )

    def choose_local_action(info: InformationSet) -> LeducAction:
        if info in memo_action:
            return memo_action[info]
        belief_states = posterior_states(info)
        legal = belief_states[0][0].legal_actions
        best_act = legal[0]
        best_ev = float("-inf")
        for act in legal:
            ev = sum(
                w * rollout_value(s.act(act), info.round_index)
                for s, w in belief_states
            )
            if ev > best_ev:
                best_ev = ev
                best_act = act
        memo_action[info] = best_act
        return best_act

    def evaluate_state(state: LeducState) -> float:
        if state.is_terminal:
            val0 = float(state.utility(0))
            return val0 if player == 0 else -val0
        if state.is_chance_node:
            return sum(
                p * evaluate_state(state.deal_public(c))
                for c, p in state.chance_outcomes
            )
        info = InformationSet.from_state(state)
        if state.current_player == player:
            act = choose_local_action(info)
            return evaluate_state(state.act(act))
        probs = opp_dist(info, state.legal_actions)
        return sum(p * evaluate_state(state.act(a)) for a, p in probs.items())

    deals = LeducGame.deals()
    return sum(evaluate_state(LeducGame.initial_state(d)) for d in deals) / len(deals)


def evaluate_lbr_hunl(
    target_agent: Agent,
    *,
    hands: int = 200,
    config: TournamentConfig | None = None,
    seed: int = 42,
    opponent_samples: int = 16,
    board_rollouts: int = 6,
) -> LBRResult:
    """Run paired Monte Carlo 1-step Local Best Response evaluation against a target HUNL agent."""
    settings = config or TournamentConfig(
        starting_stack=200, small_blind=1, big_blind=2
    )
    lbr = LocalBestResponseAgent(
        target_agent,
        seed=seed,
        opponent_samples=opponent_samples,
        board_rollouts=board_rollouts,
    )
    rng = random.Random(seed)

    hand_net_chips: list[int] = []
    started = time.perf_counter()

    for hand_idx in range(hands):
        is_lbr_button = hand_idx % 2 == 0
        p0_name, _ = ("LBR", "Target") if is_lbr_button else ("Target", "LBR")
        agents = {"LBR": lbr, "Target": target_agent}

        deck = Deck(seed=rng.randrange(2**32))
        initial_stacks = {
            "LBR": settings.starting_stack,
            "Target": settings.starting_stack,
        }
        game = GameState(
            initial_stacks,
            small_blind=settings.small_blind,
            big_blind=settings.big_blind,
            button=p0_name,
            deck=deck,
        )
        lbr.initialize_hand(game.hole_cards["LBR"])
        history: list[ActionRecord] = []

        while not game.complete:
            actor = game.current_player
            if actor is None:
                break
            obs = observe(
                game, actor, tuple(history), action_profile=settings.action_profile
            )
            action = agents[actor].act(obs)
            if actor == "Target":
                lbr.notify_action(obs, action)
            street = game.street.value
            game.act(action, player=actor)
            history.append(ActionRecord(actor, action, street))

        net = game.stacks["LBR"] - initial_stacks["LBR"]
        hand_net_chips.append(net)

    elapsed = time.perf_counter() - started
    count = len(hand_net_chips)
    mean_net = statistics.mean(hand_net_chips)
    sd = statistics.stdev(hand_net_chips) if count > 1 else 0.0
    se = sd / math.sqrt(count) if count > 1 else 0.0

    bb = settings.big_blind
    mbb_per_hand = (mean_net / bb) * 1000.0
    bb_per_100 = (mean_net / bb) * 100.0
    se_mbb = (se / bb) * 1000.0

    t_crit = 1.96 if count > 30 else 2.0
    ci_low = mbb_per_hand - t_crit * se_mbb
    ci_high = mbb_per_hand + t_crit * se_mbb

    return LBRResult(
        game="HUNL",
        target_agent=target_agent.__class__.__name__,
        hands_evaluated=count,
        lbr_net_chips_mean=mean_net,
        lbr_bb_per_100=bb_per_100,
        lbr_mbb_per_hand=mbb_per_hand,
        standard_error_mbb=se_mbb,
        ci95_low_mbb=ci_low,
        ci95_high_mbb=ci_high,
        runtime_seconds=elapsed,
    )
