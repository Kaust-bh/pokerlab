# PokerLab

PokerLab is a research-oriented, play-money No-Limit Texas Hold'em platform. Phase 1 supplies a heads-up engine; Phase 2 adds baseline policies, equity estimation, and tournament benchmarking.

## Engine

`src/pokerlab/engine/` contains immutable cards, reproducible decks, five-to-seven-card hand evaluation, layered side pots, no-limit betting, and a complete heads-up hand state. Bets and raises are target street contributions. Full raises reopen action; short all-in raises do not. Odd chips in tied pots go to the first eligible seat in the supplied order.

The engine currently supports two players, integer chips, and no rake or antes. The button posts the small blind and acts first pre-flop; the other player acts first on later streets.

## Agents and information boundary

`src/pokerlab/agents/` provides `RandomAgent`, `HeuristicAgent`, and `EquityAgent`. Policies receive an `Observation` containing only their own hole cards, public board, public stacks, pot/call information, legal actions, position, and public action history. They never receive the `GameState` or opponent hole cards. Each agent uses an instance-local RNG and can be reset with a match seed.

The random baseline samples from the shared legal-action abstraction. The heuristic policy uses configurable thresholds and transparent hand-strength bands. Its post-flop strength score is a rough made-hand/draw heuristic, not a probability of winning.

## Equity estimation

`estimate_equity` returns win, tie, and loss probabilities, equity, outcome count, confidence interval, runtime, and method. Equity is defined as `P(win) + 0.5 * P(tie)`. Unknown opponent cards and board runouts are sampled uniformly without replacement. Known cards are excluded.

The estimator uses exact enumeration automatically when the opponent-and-runout outcome space is at most 100,000; otherwise it uses reproducible Monte Carlo samples. Exact turn and river calculations are supported, with an explicit two-million-outcome safety limit. Monte Carlo confidence intervals use Hoeffding's distribution-free bound for bounded [0, 1] outcomes. `EquityAgent` uses its configured Monte Carlo sample count at every street to keep its decision cost predictable.

The estimate assumes a uniformly random opponent hand. It does not model betting ranges, future actions, fold equity, implied odds, reverse implied odds, or multiway pots. Equity should therefore inform a decision, not be treated as a complete expected-value model.

## Tournament and benchmark methodology

`TournamentSimulator` registers agent factories, creates fresh instances per match, alternates the button hand by hand, and alternates the first button across repeated matches. A match ends when a stack reaches zero or the configured hand limit is reached. Results distinguish individual hand wins/ties from match winners and include stacks, net chips, position counts, and runtime. Chip conservation is checked after every hand.

`BenchmarkRunner` includes the requested Random/Random, Heuristic/Random, Equity/Random, Equity/Heuristic, and Heuristic/Heuristic matchups. Metrics report match win rate, net chips, average profit per hand, BB/100, sample standard deviation, approximate 95% confidence intervals, and runtime per hand. Profit intervals use a normal approximation over independent match-level profit-per-hand samples. Results are preliminary; use many independent matches and long hand limits before drawing strategy conclusions. CSV exports contain one row per agent-versus-opponent summary.

Run a quick experiment:

```sh
python -m pip install -e '.[dev]'
pokerlab-benchmark --matches 20 --hands 100 --equity-simulations 100 --seed 2026
```

Choose an output file with `--output results/my_run.csv`. For a larger experiment, increase matches, hand limit, and EquityAgent simulations; runtime grows accordingly. No performance claim is made from the default quick run.

## Development

Python 3.12+ is required.

```sh
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
```

Tests live in `tests/`. Monte Carlo tests use fixed seeds; deterministic enumeration is tested against a board that guarantees a tie.

## Scope and roadmap

Phase 2 does not implement CFR, neural networks, opponent modeling, multiway tables, or a Streamlit arena. Those remain future work. PokerLab is an educational simulator using play-money chips and is not designed for real-money gambling.
