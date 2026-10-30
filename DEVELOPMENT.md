# Development notes

## Phase 2 design decisions

- The engine remains agent-independent. `agents.base.observe` creates a player-specific view and a small legal action abstraction.
- Agent factories create new instances for each match. The tournament runner resets each instance with a local seeded stream.
- Equity deals unknown cards from one precomputed unseen-card set per estimate. Exact enumeration groups by final board so the hero hand is evaluated only once for each board.
- Exact enumeration is automatic up to 100,000 outcomes and capped at 2,000,000 when explicitly requested. Monte Carlo intervals use Hoeffding's inequality for equity outcomes of 0, 0.5, or 1.
- Button position alternates by hand and initial button alternates between independent matches. Tournament stack totals are checked after each hand.
- Match confidence intervals use a normal approximation over independent match-level profit-per-hand observations; report them as approximate, especially for small samples.

## Run checks and experiments

```sh
pytest
ruff check .
ruff format --check .
pokerlab-benchmark --matches 20 --hands 100 --equity-simulations 100 --seed 2026
```

Benchmark output is explicitly preliminary. The framework does not infer agent strength from a small sample.
