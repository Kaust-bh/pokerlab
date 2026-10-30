# PokerLab

**An AI research platform for No-Limit Texas Hold'em.**

PokerLab explores decision-making under imperfect information through probability, game theory, and strategic learning.

## Objectives

- Build a modular, testable Texas Hold'em engine.
- Implement Monte Carlo-based poker agents.
- Explore Counterfactual Regret Minimization (CFR).
- Investigate opponent modelling and exploitative strategies.
- Compare AI agents through reproducible tournaments.
- Build an interactive Human vs AI poker arena.

## Planned AI Agents

| Agent | Approach |
|---|---|
| Random Agent | Random legal actions |
| Heuristic Agent | Rule-based poker strategy |
| Equity Agent | Monte Carlo simulation |
| CFR Agent | Counterfactual Regret Minimization |
| Neural Agent | Neural network-based strategy learning |

## Technology Stack

- Python 3.12+
- NumPy
- PyTorch (planned)
- Streamlit
- Plotly
- pytest

## Research Roadmap

1. Poker engine and hand evaluation.
2. Baseline poker agents.
3. Monte Carlo equity estimation.
4. CFR implementation using Kuhn Poker.
5. Extension to Leduc Poker.
6. Abstracted CFR for No-Limit Texas Hold'em.
7. Poker AI Arena and tournament evaluation.

## Project Structure

```text
src/pokerlab/
├── engine/
├── agents/
├── algorithms/
├── evaluation/
└── utils/

tests/
experiments/
models/
results/
```

## Development Status

Early development. Currently focused on building a correct, modular heads-up poker engine.

## Scope

PokerLab is an educational research simulator using play-money chips. It is not designed for real-money gambling.
