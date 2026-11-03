"""Agent catalog and instantiation adapters for PokerLab Arena."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerlab.agents import (
    Agent,
    EquityAgent,
    EquityAgentConfig,
    HeuristicAgent,
    HeuristicConfig,
    Observation,
    RandomAgent,
)
from pokerlab.engine import Action
from pokerlab.opponent_model.adaptive_agent import (
    CallingStationAgent,
    HarveySpecterAgent,
    HarveySpecterConfig,
    LooseAggressiveAgent,
    ManiacAgent,
    NitAgent,
    TightPassiveAgent,
)
from pokerlab.opponent_model.profile_store import OpponentProfileStore
from pokerlab.research.nlhe.agent import AbstractHUNLAgent, deserialize_strategy
from pokerlab.research.nlhe.config import HUNLGameConfig
from pokerlab.research.nlhe.mccfr import (
    ExternalSamplingMCCFR,
    HUNLMCCFRConfig,
)


@dataclass(frozen=True, slots=True)
class AgentCatalogEntry:
    """Metadata describing an agent or research algorithm in PokerLab."""

    id: str
    name: str
    category: str
    compatible: bool
    description: str
    incompatibility_reason: str | None = None
    is_human: bool = False


class HumanAgent(Agent):
    """Marker Agent representing an interactive human-controlled seat in Arena."""

    def act(self, observation: Observation) -> Action:
        raise RuntimeError(
            f"Player {observation.player!r} is human-controlled; "
            "submit an action via submit_human_action() instead of step()."
        )


AGENT_CATALOG: tuple[AgentCatalogEntry, ...] = (
    AgentCatalogEntry(
        id="human",
        name="Human Player",
        category="Interactive",
        compatible=True,
        description="Interactive human player submitting legal check, call, fold, bet, raise, and all-in actions.",
        is_human=True,
    ),
    AgentCatalogEntry(
        id="random",
        name="RandomAgent",
        category="Baseline HUNL",
        compatible=True,
        description="Samples uniformly from the legal No-Limit Hold'em action abstraction.",
    ),
    AgentCatalogEntry(
        id="heuristic",
        name="HeuristicAgent",
        category="Baseline HUNL",
        compatible=True,
        description="Rule-based policy using preflop hand classes, postflop hand categories, draw detection, and pot odds.",
    ),
    AgentCatalogEntry(
        id="equity",
        name="EquityAgent",
        category="Baseline HUNL",
        compatible=True,
        description="Monte Carlo showdown equity estimator (against uniform opponent hands) combined with pot-odds margins.",
    ),
    AgentCatalogEntry(
        id="mccfr",
        name="AbstractHUNLAgent (MCCFR)",
        category="Research HUNL",
        compatible=True,
        description=(
            "External-Sampling MCCFR policy for abstracted HUNL with 3-tier lookup "
            "(exact -> generalized -> card-aware tactical fallback)."
        ),
    ),
    AgentCatalogEntry(
        id="harvey",
        name="HarveySpecter (Adaptive MCCFR)",
        category="Adaptive Intelligence",
        compatible=True,
        description=(
            "Adaptive opponent modeling & exploitative strategy controller ('Play the man, "
            "not the odds'): mixes MCCFR with Bayesian range and behavioral exploitation."
        ),
    ),
    AgentCatalogEntry(
        id="calling_station",
        name="CallingStationAgent",
        category="Archetype HUNL",
        compatible=True,
        description="Loose-passive calling station: high VPIP, low PFR, rarely folds to bets.",
    ),
    AgentCatalogEntry(
        id="tight_passive",
        name="TightPassiveAgent",
        category="Archetype HUNL",
        compatible=True,
        description="Tight-passive (Rock) baseline: selective preflop, rarely bets or raises without a monster.",
    ),
    AgentCatalogEntry(
        id="loose_aggressive",
        name="LooseAggressiveAgent",
        category="Archetype HUNL",
        compatible=True,
        description="Loose-aggressive (LAG) baseline: wide opening range, frequent c-bets and barrels.",
    ),
    AgentCatalogEntry(
        id="maniac",
        name="ManiacAgent",
        category="Archetype HUNL",
        compatible=True,
        description="Hyper-aggressive maniac: raises and bets relentlessly across all streets.",
    ),
    AgentCatalogEntry(
        id="nit",
        name="NitAgent",
        category="Archetype HUNL",
        compatible=True,
        description="Ultra-tight nit: plays only premium holdings and folds marginal hands to pressure.",
    ),
    AgentCatalogEntry(
        id="kuhn_cfr",
        name="Kuhn Vanilla CFR / CFR+ / DCFR",
        category="Toy-Game Research",
        compatible=False,
        description="Exact full-tree tabular CFR family for 3-card Kuhn Poker.",
        incompatibility_reason=(
            "Operates on 3-card KuhnState/KuhnInformationSet with 1-chip fixed bets; "
            "cannot consume 52-card No-Limit Hold'em Observations."
        ),
    ),
    AgentCatalogEntry(
        id="leduc_cfr",
        name="Leduc Vanilla CFR / CFR+ / DCFR / MCCFR",
        category="Toy-Game Research",
        compatible=False,
        description="Tabular and sampled CFR solvers for 6-card fixed-limit Leduc Hold'em.",
        incompatibility_reason=(
            "Operates on 6-card LeducState/LeducInformationSet with two-bet fixed-limit "
            "rounds; incompatible with 52-card No-Limit Hold'em states and actions."
        ),
    ),
    AgentCatalogEntry(
        id="deep_cfr",
        name="DeepCFRPolicy (Kuhn / Leduc)",
        category="Neural Research",
        compatible=False,
        description="Neural advantage and strategy network solver (Brown et al., 2019) for Kuhn and Leduc.",
        incompatibility_reason=(
            "Requires a fixed-dimensional GameCodec (KuhnCodec dim=17 or LeducCodec dim=90). "
            "No HUNL neural feature codec or trained HUNL Deep CFR checkpoint exists in PokerLab."
        ),
    ),
)

_CATALOG_BY_ID: dict[str, AgentCatalogEntry] = {
    entry.id: entry for entry in AGENT_CATALOG
}

# Cache lightweight on-demand MCCFR strategies keyed by (stack_bb, seed) so
# resetting or switching seats in Arena stays fast and deterministic.
_MCCFR_CACHE: dict[tuple[float, int], tuple[dict, dict, HUNLGameConfig]] = {}


def list_agent_catalog() -> list[dict[str, Any]]:
    """Return JSON-serializable metadata for all supported and unsupported agents."""
    return [
        {
            "id": entry.id,
            "name": entry.name,
            "category": entry.category,
            "compatible": entry.compatible,
            "is_human": entry.is_human,
            "description": entry.description,
            "incompatibility_reason": entry.incompatibility_reason,
        }
        for entry in AGENT_CATALOG
    ]


def _load_or_train_mccfr(
    starting_stack: int,
    big_blind: int,
    seed: int,
    checkpoint_dir: Path | None = None,
) -> tuple[dict, dict, HUNLGameConfig]:
    if checkpoint_dir is not None:
        strategy_path = checkpoint_dir / "strategy.json"
        ckpt_path = checkpoint_dir / "checkpoint.json"
        meta_path = checkpoint_dir / "metadata.json"
        if strategy_path.exists():
            strategy = deserialize_strategy(json.loads(strategy_path.read_text()))
            strategy_sums = {}
            if ckpt_path.exists():
                ckpt_payload = json.loads(ckpt_path.read_text())
                strategy_sums = deserialize_strategy(
                    {
                        "information_sets": [
                            {"info": row["info"], "strategy": row["values"]}
                            for row in ckpt_payload.get("strategy_sums", [])
                        ]
                    }
                )
            cfg = HUNLGameConfig()
            if meta_path.exists():
                meta = json.loads(meta_path.read_text()).get("game_and_abstraction", {})
                cfg = HUNLGameConfig(**meta)
            return strategy, strategy_sums, cfg

    stack_bb = max(2.0, round(starting_stack / max(big_blind, 1), 1))
    cache_key = (stack_bb, seed)
    if cache_key not in _MCCFR_CACHE:
        cfg = HUNLGameConfig(
            starting_stack_bb=stack_bb,
            small_blind_bb=0.5,
            big_blind_bb=1.0,
            chips_per_bb=max(big_blind, 2),
            postflop_method="sampled_equity",
            postflop_buckets=8,
            equity_samples=8,
            board_abstraction="isomorphic",
        )
        solver = ExternalSamplingMCCFR(
            HUNLMCCFRConfig(iterations=4, seed=seed, game=cfg)
        )
        solver.train(4)
        strategy = solver.average_strategy_profile()
        strategy_sums = {k: dict(v) for k, v in solver.strategy_sums.items()}
        _MCCFR_CACHE[cache_key] = (strategy, strategy_sums, cfg)
    return _MCCFR_CACHE[cache_key]


def create_agent(
    agent_id: str,
    *,
    seed: int = 0,
    equity_simulations: int = 100,
    starting_stack: int = 200,
    big_blind: int = 2,
    checkpoint_dir: Path | None = None,
    opponent_id: str = "opponent",
    profile_store: OpponentProfileStore | None = None,
    harvey_config: HarveySpecterConfig | None = None,
) -> Agent:
    """Instantiate a compatible HUNL agent or raise a descriptive ValueError."""
    normalized = agent_id.strip().lower()
    if normalized == "harvey_specter":
        normalized = "harvey"
    entry = _CATALOG_BY_ID.get(normalized)
    if entry is None:
        valid = ", ".join(e.id for e in AGENT_CATALOG if e.compatible)
        raise ValueError(
            f"Unknown agent {agent_id!r}. Compatible HUNL agents: {valid}."
        )
    if not entry.compatible:
        raise ValueError(
            f"Agent {entry.name!r} ({entry.id}) is not compatible with HUNL Arena: "
            f"{entry.incompatibility_reason}"
        )
    if normalized == "human":
        return HumanAgent()
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized == "heuristic":
        return HeuristicAgent(HeuristicConfig())
    if normalized == "equity":
        return EquityAgent(
            EquityAgentConfig(simulations=max(10, int(equity_simulations))),
            seed=seed,
        )
    if normalized == "mccfr":
        strategy, strategy_sums, cfg = _load_or_train_mccfr(
            starting_stack=starting_stack,
            big_blind=big_blind,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
        )
        return AbstractHUNLAgent(
            strategy=strategy,
            config=cfg,
            seed=seed,
            strategy_sums=strategy_sums,
        )
    if normalized == "harvey":
        strategy, strategy_sums, cfg = _load_or_train_mccfr(
            starting_stack=starting_stack,
            big_blind=big_blind,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
        )
        base_mccfr = AbstractHUNLAgent(
            strategy=strategy,
            config=cfg,
            seed=seed,
            strategy_sums=strategy_sums,
        )
        return HarveySpecterAgent(
            baseline_agent=base_mccfr,
            config=harvey_config or HarveySpecterConfig(),
            opponent_id=opponent_id,
            profile_store=profile_store,
            seed=seed,
        )
    if normalized == "calling_station":
        return CallingStationAgent(seed=seed)
    if normalized == "tight_passive":
        return TightPassiveAgent(seed=seed)
    if normalized == "loose_aggressive":
        return LooseAggressiveAgent(seed=seed)
    if normalized == "maniac":
        return ManiacAgent(seed=seed)
    if normalized == "nit":
        return NitAgent(seed=seed)
    raise ValueError(f"Unhandled agent id {agent_id!r}")
