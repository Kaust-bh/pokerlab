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
PYTHONPATH=src python -m pokerlab.benchmark_cli --matches 20 --hands 100 --equity-simulations 100 --seed 2026
```

Choose an output file with `--output results/my_run.csv`. Installing the project also registers the equivalent `pokerlab-benchmark` console command. For a larger experiment, increase matches, hand limit, and EquityAgent simulations; runtime grows accordingly. No performance claim is made from the default quick run.

## Kuhn Poker and CFR research

`src/pokerlab/research/kuhn/` is a separate imperfect-information game and does not depend on the No-Limit Hold'em engine. It implements the three-card J/Q/K game with one-chip antes, one-chip bets, explicit public histories, private-card information sets, exact chance enumeration, and vanilla Counterfactual Regret Minimization (CFR).

Utilities are net chips after antes: check-check showdown is ±1, called-bet showdown is ±2, and a fold awards the bettor +1. Player 0's game value at equilibrium is −1/18 chips per hand. A full game-tree state contains both private cards; an `InformationSet` contains only the acting player's card, player identity, and public betting history.

For player i at information set I, instantaneous counterfactual regret is accumulated as

```text
r_i(I,a) = Σ_h π_-i(h) [v_i(h,a) − v_i(h)]
```

Here `π_-i` includes chance and opponent reach, but excludes player i's own reach. Regret matching normalizes positive cumulative regrets, falling back to uniform play when none are positive. The reported strategy is the cumulative average, weighted by the acting player's own reach. CFR fixes the strategy profile for a whole iteration and applies regret deltas after traversing all six ordered deals.

The evaluator computes exact best responses constrained to information sets. It reports **NashConv** as the sum of both players' unilateral improvement gains, and **exploitability** as `NashConv / 2`. Vanilla CFR is regret minimization on the full tree; it is not MCCFR or Deep CFR.

Run a reproducible training experiment with CSV metrics, an average-strategy JSON snapshot, and an HTML convergence plot:

```sh
PYTHONPATH=src python -m pokerlab.research.kuhn.experiments \
  --iterations 10000 --seed 7 --evaluation-interval 100 \
  --logging-interval 1000 --output-dir results/kuhn
```

The result includes expected game value, exploitability, NashConv, information-set count, and runtime. Kuhn has a family of equilibria, so individual action probabilities may vary while value and exploitability converge. For a primary description of Kuhn's game and equilibrium family, see [Kuhn (1950), *A Simplified Two-Person Poker*](https://sites.math.rutgers.edu/~zeilberg/akherim/PokerPapers/Kuhn1951.pdf); for counterfactual regret minimization, see [Zinkevich et al. (2007)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).

## Leduc Poker and CFR

`src/pokerlab/research/leduc/` implements a separate two-player fixed-limit Leduc environment. It uses the canonical Southey et al. variant: two suits of J/Q/K, one-chip antes, one private card per player, and one public card after the first betting round. The first-round bet size is two chips and the second-round size is four. Each round has a **two-bet maximum**: one opening bet and at most one raise, after which the facing player may fold or call. Player 0 acts first pre-flop and Player 1 acts first post-flop. This seating convention fixes the otherwise interchangeable dealer labels.

Utilities are net chips, zero-sum. Antes and all matched contributions are included; folding awards the pot to the opponent, and showdowns compare a pair made with the public rank against all high-card hands, then compare private ranks. Equal hand ranks split the pot. A full game-tree state stores both private cards for the solver, while an information set stores only the acting player's physical private card, any revealed public card, the round, and both public betting histories.

The solver enumerates all 30 ordered private deals and every legal public-card outcome. It holds one strategy profile fixed during a complete iteration and applies regret deltas afterward. At information set `I` for player `i`, instantaneous counterfactual regret is

```text
r_i(I,a) = Σ_h π_-i(h) [v_i(h,a) − v_i(h)]
```

where `π_-i` includes chance and opponent reach but excludes the acting player's reach. Average strategy mass uses the acting player's own reach, coalescing the physically distinct hidden states represented by each observation. The exact best-response evaluator shares one response action across all full histories in each information set. As in Kuhn, NashConv is the sum of unilateral gains and exploitability is `NashConv / 2`.

Run a reproducible Leduc experiment:

```sh
PYTHONPATH=src python -m pokerlab.research.leduc.experiments \
  --iterations 1000 --seed 7 --evaluation-interval 1000 \
  --logging-interval 1000 --output-dir results/leduc --compare-kuhn
```

This writes actual convergence metrics to CSV, the average strategy to JSON, and an HTML plot for exploitability, NashConv, and expected value. Records include the exact tree node count, information-set count, measured training throughput, and best-response evaluation runtime. The first iteration visits 9,450 tree nodes and 936 information sets for this variant. `--compare-kuhn` trains the Kuhn solver with the same seed and iteration count and writes a comparison CSV. Exact best-response evaluation is more expensive than one CFR iteration; increase the evaluation interval for long runs.

Recorded comparison with seed `7`, 5,000 iterations, and evaluation every 500 iterations:

| Game | Expected value | Exploitability | Information sets | Nodes/iteration | Iterations/s | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Leduc | −0.008098 | 0.014781 | 936 | 9,450 | 17.42 | 287.00 s |
| Kuhn | −0.055506 | 0.003302 | 12 | 54 | 2,769.23 | 1.81 s |

Exploitability and values are in each game's chip units and should be interpreted with each game's scale. The Leduc exploitability fell from `0.057890` at iteration 500 to `0.014781` at 5,000; expected value settled near `−0.008` from iterations 3,000–5,000. Generated files are under `results/leduc/`, including the CSV, JSON, HTML plot, and Kuhn comparison CSV.

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

Phases 3A and 3B implement vanilla CFR for Kuhn and fixed-limit Leduc Poker. Phase 3C adds external-sampling MCCFR for those same research games. The project does not implement NLHE CFR, CFR+, Deep CFR, neural networks, opponent modeling, or a Streamlit arena. PokerLab is an educational simulator using play-money chips and is not designed for real-money gambling.

## Phase 3C: External Sampling MCCFR

PokerLab includes separate External Sampling Monte Carlo CFR solvers for Kuhn
and Leduc. Each iteration freezes one behavioral profile, samples a private
deal and samples chance/opponent decisions, while enumerating all actions at
the traverser's information sets. The sampled action-value differences are
counterfactual regret estimates: expectation over chance and opponent samples
recovers their reach weights, so regret updates do not multiply by traverser
reach.

Average strategy follows vanilla CFR's own-reach weighting. The implementation
updates that average exactly from each information set's own action sequence,
including sets absent from sampled regret trajectories. This separate
information-set pass preserves average-strategy comparability but adds cost to
each MCCFR iteration.

Run a modest same-seed comparison with exact best-response evaluation:

```sh
PYTHONPATH=src python -m pokerlab.research.mccfr_experiments \
  --games kuhn leduc --seeds 7 19 --iterations 1000 \
  --evaluation-interval 250 --output-root results/mccfr
```

Each run writes to a new timestamped subdirectory under `results/mccfr/`,
including convergence and comparison CSV files, final strategies, experiment
configuration, Git provenance and artifact hashes, and plots for exact
exploitability versus iterations and training-only wall time. A third plot
shows the summed L1 distance between same-seed average strategies at matching
checkpoints. Exact evaluation time is reported separately and excluded from
training throughput. The approximate strategy-table memory estimate is not
process RSS. Results from small runs are diagnostics, not a claim that MCCFR
always converges faster.

The 1,000-iteration pilot used seeds 7 and 19. At the final checkpoint,
vanilla CFR's exploitability was 0.007269 in Kuhn and 0.035678 in Leduc.
MCCFR's was 0.028629–0.035237 in Kuhn and 1.231762–1.326018 in Leduc.
Mean training times for the two seeds were 0.294 s vanilla versus 0.088 s
MCCFR in Kuhn, and 70.02 s versus 7.55 s in Leduc. Thus MCCFR took less
training time per 1,000 iterations in this run but had substantially higher
exact exploitability at the same iteration count. The Leduc MCCFR Player 0
values ranged from −0.096894 to 0.104598, indicating high run variance at
this budget. These measurements do not show MCCFR reaching a useful Leduc
equilibrium at 1,000 iterations. The full per-checkpoint curves, paired
strategy distances, configurations, and hashes are in
[results/mccfr/run-20261030T213116Z/](results/mccfr/run-20261030T213116Z/).
For denser wall-time checkpoints with the same seeds and training budget, see
[results/mccfr/run-20261030T213604Z/](results/mccfr/run-20261030T213604Z/).
At approximately matched training time near the end of the MCCFR runs, vanilla
had lower exact exploitability: in Leduc, about 0.167 at its 100-iteration
checkpoint versus 1.23–1.33 for MCCFR at 1,000 iterations; in Kuhn, about
0.0135 at roughly 300 vanilla iterations versus 0.0286–0.0352 for MCCFR at
1,000 iterations. These are pilot observations from two seeds and one machine.

See [docs/phase3c_mccfr.md](docs/phase3c_mccfr.md) for the estimator derivation,
validation details, run artifacts and limitations. The implementation follows
Lanctot et al. (2009), [*Monte Carlo Sampling for Regret Minimization in
Extensive Games*](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf).

## Phase 3D: CFR+ and DCFR

Kuhn and Leduc also include separate full-tree CFR+ and DCFR solvers. CFR+
uses alternating player regret passes, nonnegative cumulative regrets, and
iteration-weighted own-reach strategy averaging. A CFR+ iteration is one full
Player 0 pass followed by one full Player 1 pass. DCFR uses the defaults
`alpha=1.5`, `beta=0`, and `gamma=2`; these parameters are configurable.
Vanilla CFR and baseline MCCFR remain separate implementations.

Run a three-seed, 20-iteration comparison for all four algorithms in both
games. Each run creates a new timestamped directory and exact evaluator plots:

```sh
PYTHONPATH=src python -m pokerlab.research.cfr_variant_experiments \
  --iterations 20 --seeds 11 23 47
```

The comparison records full-tree passes, player updates, training time,
evaluation time, exact NashConv and exploitability, and average-strategy JSON.
The budget is a smoke-scale comparison, not a general performance claim. See
[docs/phase3d_cfr_variants.md](docs/phase3d_cfr_variants.md) for the algorithm
conventions, measured results, and limitations.

## Phase 4: Abstracted heads-up NLHE

An initial engine-backed HUNL prototype is available under
`pokerlab.research.nlhe`. It uses a configurable half-pot/pot/all-in action
menu, 169 preflop hand classes, deterministic approximate postflop buckets,
player-visible information sets, and on-demand external-sampling MCCFR. The
full 100 BB environment is the game default; the documented smoke experiment
uses a smaller stack and only one training iteration because the traverser
still enumerates abstract actions. Evaluation against random, heuristic and
Monte Carlo equity baselines is exploratory and does not establish equilibrium
quality. No full-game exploitability is claimed.

Run the smoke experiment with:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.experiments \
  --iterations 1 --seed 7 --hands 4 --stack-bb 5
```

Each invocation creates a fresh artifact folder under
`results/nlhe_abstraction/`. See
[docs/phase4_abstracted_hunl.md](docs/phase4_abstracted_hunl.md) for the game,
estimator, abstraction and limitations.

## Phase 5A: configurable HUNL abstractions

Phase 5A adds opt-in sampled showdown-equity buckets, deterministic draw and
board-texture features, and `legacy`, `standard`, and `wide` betting profiles.
The Phase 4 defaults remain in place. For example:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.experiments \
  --iterations 1 --seed 105 --hands 1 --stack-bb 3 \
  --postflop-method sampled_equity --postflop-buckets 8 \
  --equity-samples 8 --opponent-range premium --action-profile wide
```

See [docs/phase5_abstraction.md](docs/phase5_abstraction.md) for the estimator,
configuration choices, measured profiling, and limitations.

## Phase 5B: experimental abstraction benchmarks

The Phase 5B runner trains the same five abstraction variants with paired
seeds, writes resumable strategy and solver checkpoints, and evaluates against
Random, Heuristic, Equity and compatible earlier checkpoints using fixed-hand
batches from both button positions. Check the runtime estimate before the
default multi-seed suite:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark --estimate-only
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark
```

Outputs go to fresh directories under `results/nlhe_abstraction/`. The report
documents the measured comparisons and their statistical limits:
[docs/phase5_evaluation.md](docs/phase5_evaluation.md). These evaluations do
not calculate HUNL exploitability.

## Phase 6A: Deep CFR prototype

Install the optional neural dependency and run a small Kuhn comparison against
the tabular solvers:

```sh
python -m pip install -e '.[deep-cfr]'
PYTHONPATH=src python -m pokerlab.algorithms.deep_cfr.experiments \
  --game kuhn --iterations 100 --seed 7 --traversals-per-player 4 \
  --evaluation-interval 20 --weighting uniform
```

The prototype also supports Leduc exact evaluation. It has not been applied to
HUNL. See [docs/phase6_deep_cfr.md](docs/phase6_deep_cfr.md) for the estimator,
weighting, tests, pilot results and limitations.

## Phase 6B: Deep CFR convergence study

Run the paired-seed Kuhn study, including controlled traversal, network-fit,
and reservoir ablations:

```sh
PYTHONPATH=src python -m pokerlab.algorithms.deep_cfr.study \
  --game kuhn --iterations 500 --evaluation-interval 100 \
  --seeds 11 23 47
```

Use 100 iterations with evaluation every 50 for Leduc. The studies compare
against vanilla CFR, CFR+, and DCFR using exact exploitability. See
[docs/phase6b_deep_cfr_evaluation.md](docs/phase6b_deep_cfr_evaluation.md) for
seed-level results, confidence intervals, strategy distances, runtime,
memory measurements, provenance notes, and limitations. No HUNL Deep CFR
training is included.
