"""Legal-action average-policy inference for Kuhn and Leduc Deep CFR."""

from __future__ import annotations

import random

from pokerlab.algorithms.deep_cfr.encoding import GameCodec, KuhnCodec, LeducCodec
from pokerlab.algorithms.deep_cfr.networks import StrategyNetwork, require_torch
from pokerlab.research.kuhn.game import KuhnGame
from pokerlab.research.kuhn.information_sets import InformationSet as KuhnInfo
from pokerlab.research.leduc.game import LeducGame
from pokerlab.research.leduc.information_sets import InformationSet as LeducInfo


class DeepCFRPolicy:
    """Average-strategy network wrapper with legal-action masking."""

    def __init__(
        self,
        codec: GameCodec,
        strategy_network: StrategyNetwork,
        *,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        torch = require_torch()
        self.codec = codec
        self.strategy_network = strategy_network
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.strategy_network.eval()

    def action_probabilities(self, info, legal_actions) -> dict:
        torch = require_torch()
        indices = tuple(self.codec.action_index(action) for action in legal_actions)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("legal actions must map to distinct network outputs")
        mask = [index in indices for index in range(len(self.codec.action_vocabulary))]
        with torch.inference_mode():
            features = torch.tensor(
                self.codec.encode(info), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            legal_mask = torch.tensor(
                mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)
            probabilities = self.strategy_network.masked_probabilities(
                features, legal_mask
            )[0]
        result = {
            action: float(probabilities[index].item())
            for action, index in zip(legal_actions, indices, strict=True)
        }
        total = sum(result.values())
        return {action: value / total for action, value in result.items()}

    def choose_action(self, info, legal_actions, *, deterministic: bool = False):
        probabilities = self.action_probabilities(info, legal_actions)
        if deterministic:
            return max(legal_actions, key=lambda action: probabilities[action])
        return self.rng.choices(
            legal_actions,
            weights=[probabilities[action] for action in legal_actions],
            k=1,
        )[0]

    def action_for_state(self, state, *, deterministic: bool = False):
        if isinstance(self.codec, KuhnCodec):
            info = KuhnInfo.from_state(state)
        elif isinstance(self.codec, LeducCodec):
            info = LeducInfo.from_state(state)
        else:
            raise TypeError("action_for_state needs a KuhnCodec or LeducCodec")
        return self.choose_action(
            info, self.codec.legal_actions(state), deterministic=deterministic
        )

    def exact_strategy_profile(self) -> dict:
        """Enumerate each reachable information set without exposing hidden cards."""
        information_sets = set()

        if isinstance(self.codec, KuhnCodec):

            def visit_kuhn(state):
                if state.is_terminal:
                    return
                information_sets.add(KuhnInfo.from_state(state))
                for action in state.legal_actions:
                    visit_kuhn(state.act(action))

            for deal in KuhnGame.deals():
                visit_kuhn(KuhnGame.initial_state(deal))
        elif isinstance(self.codec, LeducCodec):

            def visit_leduc(state):
                if state.is_terminal:
                    return
                if state.is_chance_node:
                    for card, _probability in state.chance_outcomes:
                        visit_leduc(state.deal_public(card))
                    return
                information_sets.add(LeducInfo.from_state(state))
                for action in state.legal_actions:
                    visit_leduc(state.act(action))

            for deal in LeducGame.deals():
                visit_leduc(LeducGame.initial_state(deal))
        else:
            raise TypeError("exact profile enumeration is unavailable for this codec")

        return {
            info: self.action_probabilities(
                info,
                tuple(
                    action
                    for action in self.codec.action_vocabulary
                    if self.codec.action_index(action)
                    in _legal_indices(self.codec, info)
                ),
            )
            for info in sorted(information_sets, key=lambda item: item.identifier)
        }


def _legal_indices(codec, info) -> set[int]:
    from pokerlab.research.kuhn.game import legal_actions_for_history
    from pokerlab.research.leduc.betting import legal_actions

    if isinstance(codec, KuhnCodec):
        legal = legal_actions_for_history(info.history)
    else:
        legal = legal_actions(info.histories[info.round_index])
    return {codec.action_index(action) for action in legal}
