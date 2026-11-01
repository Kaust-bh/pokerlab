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
PYTHONPATH=src python -m pokerlab.benchmark_cli --matches 20 --hands 100 --equity-simulations 100 --seed 2026
```

Benchmark output is explicitly preliminary. The framework does not infer agent strength from a small sample.

## Recorded Phase 2 benchmark

The recorded workspace benchmark used 250 matches per pairing, a 50-hand cap, 5,000 starting chips, 1/2 blinds, 25 Monte Carlo samples per EquityAgent decision, and seed `20261031`. Aggregate match win rates (with approximate 95% intervals) were:

| Matchup | Hands | Win rate | 95% interval |
|---|---:|---:|---:|
| Random A vs Random B | 1,298 | 0.536 | 0.474–0.598 |
| Heuristic vs Random | 4,268 | 0.664 | 0.605–0.723 |
| Equity vs Random | 1,942 | 0.684 | 0.626–0.742 |
| Equity vs Heuristic | 12,500 | 0.622 | 0.562–0.682 |
| Heuristic A vs Heuristic B | 12,500 | 0.442 | 0.381–0.503 |

The complete CSV is `results/phase2_benchmark.csv`. These are exploratory results, not evidence that one strategy is generally stronger. Random-agent matches frequently ended before the hand cap, and the baseline policies and equity sample budget are intentionally simple.

## Phase 3A: Kuhn Poker vanilla CFR

### Implemented

- `research/kuhn/game.py`: J/Q/K chance deals, legal action histories, terminal detection, and net-chip zero-sum utilities.
- `research/kuhn/information_sets.py`: stable acting-player observations that omit the opponent's card.
- `research/kuhn/strategy.py`: regret matching, own-reach-weighted average-strategy normalization, and JSON snapshots.
- `research/kuhn/cfr.py`: simultaneous full-tree vanilla CFR updates over all six ordered deals.
- `research/kuhn/evaluation.py`: exact expected value, information-set-consistent pure best responses, NashConv, and exploitability.
- `research/kuhn/experiments.py`: reproducible training, CSV and strategy export, CLI, and Plotly convergence output.

### Mathematical conventions

Utilities are Player 0's net chips after both antes: showdown after check-check is ±1; a called bet showdown is ±2; a fold awards the bettor +1. Player 1 receives the negative utility. Counterfactual regret uses chance reach times opponent reach, excluding the acting player's reach. To compute average strategy, each information set is coalesced once per iteration and weighted by the acting player's own reach. This is the behavioral-strategy average, not the final current strategy.

NashConv is the sum of Player 0's and Player 1's best-response gains against the profile. This project reports exploitability as NashConv/2; some sources use “exploitability” for NashConv itself, so both values are exported.

The six chance deals and public action branches are traversed exactly. Best responses enumerate the Kuhn player's pure information-set policies, so choices cannot condition on hidden opponent cards. The implementation is vanilla full-tree CFR, not MCCFR or Deep CFR.

### Validation and recorded experiment

Tests validate every terminal history and payoff, information-set identity/privacy, first-iteration counterfactual regret (including the 1/8 Jack-root bet regret), average reach weighting, regret matching, strategy snapshots, exact known equilibrium value and best responses, and a 5,000-iteration convergence run. The full suite passes **62 tests**; Ruff checks and formatting pass.

The reproducible 10,000-iteration experiment used seed `7`, evaluation every 100 iterations, and logging every 1,000 iterations. At iteration 10,000, Player 0's average-profile expected value was `−0.055546` chips per hand (reference `−1/18 ≈ −0.055556`), NashConv was `0.004636`, and exploitability was `0.002318`. Exploitability declined from `0.007269` at iteration 1,000. The learned first-player opening frequencies were J `0.208`, Q `0.000`, K `0.613`; this is one approximate member of Kuhn's equilibrium family rather than a unique target strategy.

Generated outputs are `results/kuhn/convergence.csv`, `results/kuhn/average_strategy.json`, and `results/kuhn/convergence.html`.

### Limitations and next steps

Kuhn's solver and exact best-response enumeration are specialized to its small tree. It does not implement CFR+, sampled CFR, subgame solving, or NLHE CFR. Leduc is implemented separately below and does not alter this Kuhn code.

References: [Kuhn, “A Simplified Two-Person Poker” (1950)](https://sites.math.rutgers.edu/~zeilberg/akherim/PokerPapers/Kuhn1951.pdf); [Zinkevich et al., “Regret Minimization in Games with Incomplete Information” (NIPS 2007)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf).

## Phase 3B: Leduc Poker vanilla CFR

### Selected rules

The implementation follows the two-bet-maximum Leduc Hold'em described by Southey et al. The deck has two physical suits of J, Q, and K; both players ante one chip and receive one private card. Player 0 acts first in round one. A check-check or a called bet advances to the public-card chance node; a fold ends the hand. Player 1 acts first in round two. Bet sizes are two chips in round one and four in round two. A round permits one opening bet and one raise maximum; after the raise, only fold and call are legal. This defines “two-bet maximum” precisely and prevents mixing it with variants that permit additional raises. There are no blinds, rake, stacks, or side pots.

An opening bet adds one round unit to the bettor's contribution. A raise first matches the outstanding wager and adds one more unit, so it contributes twice the round's bet size; the caller then matches the additional unit. A fold winner's net payoff is the folded player's contribution. At showdown the winner's net is the opponent's total contribution, and a tie returns zero to both. Utilities are therefore net chips after ante and action contributions and sum to zero in every terminal state.

### Information sets and CFR

`research/leduc/game.py` contains the full hidden state and explicit public-card chance node. `information_sets.py` groups states using the acting player, their physical private card, the public card if revealed, round index, and the public action histories. The other private card and undealt cards never enter the key. Histories are full game-tree states; information sets are player observations that may contain many histories.

The solver visits all 30 ordered private deals and all four available public cards after a non-folded first round. At `I` for player `i`, regret uses chance reach and opponent reach, excluding `i`'s own reach. It adds `π_-i(h) * (v_i(h,a) - v_i(h))` for each compatible history. Average behavioral strategy accumulates `π_i(I) * σ_i(I,a)`, excluding opponent reach. Compatible physical hidden-card states share own reach for this observation, so the solver coalesces them once per iteration. Strategy deltas are committed only after all chance outcomes have been traversed, matching the simultaneous Kuhn convention.

Exact best responses use backward information-set optimization. For each candidate response information set, the evaluator sums action values across its full histories with chance and opponent reach weights, then chooses one action for the entire information set. It does not enumerate opponent private cards as if they were known. NashConv sums both best-response gains; reported exploitability is NashConv divided by two.

Run training and evaluation:

```sh
PYTHONPATH=src python -m pokerlab.research.leduc.experiments \
  --iterations 1000 --seed 7 --evaluation-interval 1000 \
  --logging-interval 1000 --output-dir results/leduc --compare-kuhn
PYTHONPATH=src python -m pokerlab.research.kuhn.experiments \
  --iterations 1000 --seed 7 --evaluation-interval 1000 \
  --logging-interval 1000 --output-dir results/kuhn
```

CSV records include iteration, expected value, NashConv, exploitability, information-set count, runtime, tree node count, training iterations per second, and best-response runtime. Average strategy snapshots are JSON; plots show exploitability, NashConv, and expected value. `--compare-kuhn` runs Kuhn at the same iteration count and seed and saves a side-by-side CSV. The selected variant traverses 9,450 nodes and has 936 information sets per complete iteration. Exact best-response passes take longer than a training iteration, so long runs should use a wider evaluation interval.

### Recorded 5,000-iteration comparison

Both solvers were run with seed `7`, 5,000 iterations, and evaluation every 500 iterations. Leduc's average-profile Player 0 value was `−0.008098`, NashConv `0.029563`, and exploitability `0.014781` at the end. Exploitability declined from `0.057890` at 500 iterations; value was between `−0.007798` and `−0.008098` from iterations 3,000–5,000. Kuhn reached value `−0.055506`, NashConv `0.006605`, and exploitability `0.003302`. Leduc traversed 9,450 nodes per iteration and had 936 information sets; Kuhn traversed 54 nodes per iteration and had 12 information sets. Their chips-per-hand scales differ, so raw exploitability is contextual rather than a direct strength comparison. Leduc trained at 17.42 iterations/s and took 287.00 seconds including exact best-response evaluations. The paired comparison Kuhn run trained at 2,769 iterations/s and took 1.81 seconds. CSV, strategy, and plot outputs are under results/leduc/; the comparison CSV is in the same directory.

Reference: [Southey et al., “Bayes' Bluff: Opponent Modeling in Poker” (UAI 2005)](https://poker.cs.ualberta.ca/publications/UAI05.pdf), which defines the six-card deck, ante, two-bet maximum, and round bet sizes.

## Phase 3C: External Sampling MCCFR

The Kuhn and Leduc `mccfr.py` modules are separate from their vanilla CFR
solvers. A training iteration performs one external-sampling regret traversal
for each player. Each traversal uses the same frozen strategy profile, samples
the private deal, public chance and opponent actions, and enumerates the
traverser's legal actions. The sample's probability already supplies chance and
opponent reach in expectation; regret deltas receive no additional importance
weight and exclude traverser reach.

The average strategy is updated separately using the acting player's own
action sequence. This is exact own-reach mass at every information set, not
sample visitation frequency. It preserves the vanilla solver's behavioral
average when chance or opponent sampling misses an information set. The
catalog-based pass has per-iteration cost proportional to the information-set
count.

Validation is independent of the MCCFR regret update. Kuhn enumerates all six
deals and all deterministic opponent-action assignments under a fixed
behavioral profile; its expected sampled regrets for both players are compared
with a separate full-tree reference. Kuhn and Leduc average increments are
compared with independent own-reach full-tree references. Leduc also has fixed
profile, seeded statistical tests for selected information sets for both
players. Tests state their chance/opponent probabilities and analytically
derived action utilities, and use six-standard-error bounds plus a small
numerical allowance. A non-unit traverser own reach verifies that it is
excluded from regret. Other tests check hidden-state information-set sharing,
frozen simultaneous updates, identical-seed reproducibility, and
different-seed trajectories.

Run the same-seed vanilla/MCCFR comparison and exact evaluations:

```sh
PYTHONPATH=src python -m pokerlab.research.mccfr_experiments \
  --games kuhn leduc --seeds 7 19 --iterations 1000 \
  --evaluation-interval 250 --output-root results/mccfr
```

The command creates a fresh timestamped run folder and fails rather than
overwriting an existing run. It writes per-checkpoint values for both players,
NashConv, exact exploitability, cumulative training and evaluation time,
training-only iterations per second, seed and an approximate strategy-table
memory size. Comparison output includes L1 distance from the same-seed vanilla
average strategy. HTML plots compare exploitability against both iterations
and training time. The experiment manifest stores configuration, Git HEAD,
dirty status, source hashes and output hashes. Runtime comparisons include
strategy averaging in training and exclude exact best-response evaluation.

See [docs/phase3c_mccfr.md](docs/phase3c_mccfr.md) for derivation, interpretation
and limitations. The implementation applies the external-sampling construction
from Lanctot et al. (2009), [*Monte Carlo Sampling for Regret Minimization in
Extensive Games*](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf).

### Recorded pilot

The recorded comparison used 1,000 iterations, checkpoints every 250
iterations, and seeds 7 and 19. At iteration 1,000, vanilla CFR's exact
exploitability was 0.007269 in Kuhn and 0.035678 in Leduc. MCCFR's ranged
0.028629–0.035237 in Kuhn and 1.231762–1.326018 in Leduc. Mean training time
was 0.294 s vanilla versus 0.088 s MCCFR in Kuhn, and 70.02 s versus 7.55 s
in Leduc. Training time excludes exact evaluation and includes the exact
own-reach average-strategy update.

The pilot therefore shows lower training cost per iteration for MCCFR but
substantially worse exploitability at equal iteration count. Leduc MCCFR's
Player 0 expected value differed by seed (−0.096894 and 0.104598), and its
exploitability remained high. This is an implementation and reproducibility
pilot, not evidence of Leduc equilibrium convergence. Both solvers retained
the same approximate strategy-table footprint: 13,984 bytes for Kuhn and
1,174,640 bytes for Leduc under the documented Python-size estimate; this is
not whole-process memory.

Artifacts are under
`results/mccfr/run-20261030T213116Z/`: checkpoint CSVs, final strategies,
same-seed strategy-distance comparisons, plots, configuration, run summary,
and a manifest with Git status and hashes. The benchmark is reproducible with
the command above.

A second run with the same seeds and iteration budget but 50-iteration
evaluation checkpoints is saved under
`results/mccfr/run-20261030T213604Z/`. Its denser wall-time curves allow a
direct approximate comparison where the measured training-time ranges overlap.
Near the end of each 1,000-iteration MCCFR run, the closest vanilla checkpoint
had lower exploitability: approximately 0.167 at 100 vanilla Leduc iterations
versus 1.23–1.33 for MCCFR, and approximately 0.0135 at 300 vanilla Kuhn
iterations versus 0.0286–0.0352 for MCCFR. Those points are matched by elapsed
training time, not by iteration count. They show that cheaper MCCFR iterations
did not produce better play quality at this pilot budget.

## Phase 3D: Full-tree CFR variants

The CFR+ and DCFR implementations are separate from vanilla CFR and MCCFR.
CFR+ alternates two complete player update passes within each counted
iteration. Player 1's pass observes Player 0's newly updated regret-matching+
strategy. Linear average-strategy mass is added separately for each player
using that player's policy at the beginning of its own pass and its own reach.
DCFR performs one simultaneous full-tree pass per iteration, discounts old
regrets by sign and prior average-strategy mass by the configured factor, then
adds the current full-tree contributions.

Run the reproducible comparison with:

```sh
PYTHONPATH=src python -m pokerlab.research.cfr_variant_experiments \
  --iterations 20 --seeds 11 23 47
```

This writes a fresh run folder in `results/cfr_variants/`; the runner will not
reuse an existing timestamped directory. Its convergence CSV stores per-seed
records at iterations 5, 10, and 20, including exact evaluation time separately
from training time, player update counts, and full-tree traversal counts. The
manifest captures the Git HEAD, dirty status, and hashes of the generated
artifacts. Increase iteration budgets only after reviewing the diagnostic
pilot in `docs/phase3d_cfr_variants.md`.

## Phase 4: abstracted HUNL smoke run

The engine-backed abstract game is in `src/pokerlab/research/nlhe/`. Run its
small reproducible training and baseline-evaluation experiment with:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.experiments \
  --iterations 1 --seed 7 --hands 4 --stack-bb 5
```

The runner creates a new folder under `results/nlhe_abstraction/`, records
configuration, seed, Git provenance and artifact hashes, and compares against
Random, Heuristic and low-simulation Equity agents. Keep smoke budgets small
until traverser-branching and importance-weight variance have been measured.
The documented 5 BB smoke profile is not the 100 BB default game configuration.
See `docs/phase4_abstracted_hunl.md` for estimator details and known limits.

## Phase 5A: abstraction profiles

The default HUNL profile remains the Phase 4 legacy configuration. To run the
optional sampled-equity hand abstraction and wide betting menu:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.experiments \
  --iterations 1 --seed 105 --hands 1 --stack-bb 3 \
  --postflop-method sampled_equity --postflop-buckets 8 \
  --equity-samples 8 --opponent-range premium --action-profile wide
```

Runs record the full abstraction configuration and source/artifact hashes in a
fresh results directory. Do not compare different action profiles as the same
abstract game. See `docs/phase5_abstraction.md` for measured performance and
limitations.

### Phase 5B: paired abstraction evaluation

Estimate and run the multi-seed MCCFR comparison with:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark --estimate-only
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark
```

The benchmark tests strength/equity hand features and legacy/wide action
profiles under the same seeds. It evaluates fixed fresh-stack hand batches
with each policy on the button, preventing tournament bust-outs from reducing
the requested sample. Checkpoint state supports deterministic continuation
via `--resume-checkpoint`; every invocation creates a new artifact directory.
See `docs/phase5_evaluation.md` for the design, statistics, result files and
measured limitations. No exact HUNL exploitability is claimed.

### Phase 6A: Deep CFR

Deep CFR uses an optional PyTorch dependency; install it with
`python -m pip install -e '.[deep-cfr]'`. Run a Kuhn comparison against
vanilla CFR and CFR+ with:

```sh
PYTHONPATH=src python -m pokerlab.algorithms.deep_cfr.experiments \
  --game kuhn --iterations 100 --seed 7 --traversals-per-player 4 \
  --evaluation-interval 20 --weighting uniform
```

Use `--game leduc` for the small Leduc comparison. Outputs are written to
fresh `results/deep_cfr/run-*` directories and include exact exploitability,
training diagnostics, strategy snapshots and Deep CFR checkpoints.
`DeepCFRTrainer` supports checkpoint resume independently through
`load_checkpoint`. No HUNL neural training is part of Phase 6A. See
`docs/phase6_deep_cfr.md` for the formulation and measured limitations.

### Phase 6B: multi-seed convergence and ablations

Run the controlled, paired-seed Kuhn study with:

```sh
PYTHONPATH=src python -m pokerlab.algorithms.deep_cfr.study \
  --game kuhn --iterations 500 --evaluation-interval 100 \
  --seeds 11 23 47
```

For Leduc, use `--game leduc --iterations 100 --evaluation-interval 50`.
The study process launches each seed/configuration in its own worker so peak
RSS is isolated. It compares vanilla CFR, CFR+, and DCFR, and writes raw and
aggregated CSVs, paired confidence intervals, strategy distances, training
diagnostics, plots, and hash manifests under a new `results/deep_cfr/study-*`
directory. See `docs/phase6b_deep_cfr_evaluation.md` before increasing budgets;
the measured Deep CFR policies remain substantially more exploitable than
tabular references at the tested checkpoints.

### Phase 7: HUNL solver validation, generalization, and Local Best Response

Phase 7 resolves four methodological issues in the abstracted HUNL stack:

1. **MCCFR average-strategy estimator (`src/pokerlab/research/nlhe/mccfr.py`)**:
   In `reach_weighted` mode, `_average_traverse` samples opponent actions from
   $q_{-i}(a \mid h) = 1 / |A(h)|$ and weights traverser strategy increments by
   `average_strategy_reach_weight(own_reach, opponent_proposal_reach)` =
   $\pi_i(h) / q_{-i}(h)$. Because the chance prior $C(I) = \sum_{h \in I} p_c(h)$
   factors out of each information set $I$, normalized cumulative strategy sums
   recover the exact sequence-form average strategy without opponent-reach bias
   or $1 / p_c(h)$ weight explosion.
2. **Suit isomorphism and postflop regret sharing (`abstraction.py`, `information_set.py`)**:
   `_suit_signature` canonicalizes suits jointly across `(hole_cards, board)`
   and sorts flop cards while keeping turn and river order intact. During
   training, `_abstract_regrets` pools postflop regrets by
   `postflop_abstraction_key(board_mode)` so distinct deals share postflop
   experience, while `regret_delta` is applied before `_average_traverse`.
3. **Hierarchical policy generalization and lookup telemetry (`agent.py`, `benchmark.py`)**:
   `AbstractHUNLAgent` weights generalized candidate information sets by
   checkpoint `strategy_sums`, replaces uniform/blind fallback with a
   bucket-strength-aware `_tactical_fallback`, and exports `exact_hits`,
   `generalized_hits`, `fallback_hits`, and their rates to `evaluation.csv` and
   `summary.csv`.
4. **Local Best Response (`src/pokerlab/evaluation/lbr.py`)**:
   `BayesianRangeTracker` updates posterior hand weights using non-mutating
   `agent.action_probabilities(obs)`. `candidate_action_evs` accounts for sunk
   versus incremental street contributions, effective stack caps, uncallable
   overbet return, and opponent fold/call/raise probabilities. `evaluate_lbr_kuhn`
   and `evaluate_lbr_leduc` are validated independently against exact best
   responses. `evaluate_lbr_hunl` is documented explicitly as a finite-sample
   1-step Monte Carlo tactical probe rather than an exact global lower bound.

The 6-variant, 4-seed, 50-iteration benchmark is recorded under
`results/nlhe_abstraction/run-20261101T220341Z-6f72de15/` (`1,200` training
iterations, `528` evaluation batches, `52,800` hands). At iteration 50, learned
policy coverage (`exact + generalized`) reached 64.1%–79.2% across variants
(79.2% for `equity_8`, 72.1% for `isomorphic`, 71.9% for `equity_20`, 70.4% for
`baseline`). See `docs/phase7_hunl_validation.md` for full derivations, tables,
and limitations.
