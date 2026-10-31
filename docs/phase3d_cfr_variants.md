# Phase 3D: CFR+ and DCFR

## Scope and implementation

This phase adds full-tree CFR+ and DCFR solvers for Kuhn and fixed-limit
Leduc. Existing vanilla CFR, baseline ES-MCCFR, game rules, and payoff
conventions remain unchanged. Both new variants use the current exact
information-set evaluators and the same expected-value and exploitability
conventions as the existing solvers.

The common helpers in `src/pokerlab/research/cfr_variant_utils.py` implement
Regret Matching+, stable DCFR discount factors, and training records that
separate training from exact evaluation. Each game has its own traversal and
configuration modules:

- `kuhn/cfr_plus.py`, `leduc/cfr_plus.py`
- `kuhn/dcfr.py`, `leduc/dcfr.py`

### CFR+

At information set `I`, Regret Matching+ chooses an action in proportion to
its positive cumulative regret, with uniform play if all regrets are zero.
After the current player's full-tree regret pass, each action's regret is
updated as `max(0, old_regret + instantaneous_counterfactual_regret)`. The
counterfactual update weights by chance and opponent reach, excluding the
acting player's own reach.

One counted CFR+ iteration consists of two full-tree traversals: Player 0 is
updated first, then Player 1 is updated against the strategy that includes
Player 0's newly clipped regrets. Regret deltas are accumulated over every
chance outcome before they are committed for that player. This is the
alternating update convention described by Tammelin; it is not equivalent to
the simultaneous update used by vanilla CFR. CFR+ alternation has practical
motivation, while its convergence analysis has additional subtleties.

Each player's average strategy receives a separate own-reach-weighted
contribution from the profile at the start of that player's pass. At iteration
`t`, that contribution is multiplied by `t`. Thus, the average does not use
the player's post-update policy for the regret update that just completed.
An iteration counter increments once after both passes; player updates and
traversals are counted separately in benchmark records.

### DCFR

The full-tree DCFR traversal freezes one profile for the complete iteration,
computes all chance-weighted counterfactual regret increments, and computes
the current own-reach average-strategy contribution. For iteration `t`, each
old regret is first multiplied by a sign-specific factor, then its current
instantaneous regret is added:

```text
R_t(I,a) = R_(t-1)(I,a) * t^alpha / (t^alpha + 1) + r_t(I,a),  if R_(t-1) > 0
R_t(I,a) = R_(t-1)(I,a) * t^beta  / (t^beta  + 1) + r_t(I,a),  otherwise
S_t(I,a) = S_(t-1)(I,a) * (t / (t + 1))^gamma + pi_i(I) * sigma_i(I,a)
```

Regret matching uses the resulting cumulative regrets. Defaults are
`alpha=1.5`, `beta=0`, and `gamma=2`; all are configurable. `alpha` and `beta`
must be finite. `gamma` must be finite and nonnegative because it controls a
discount factor. Stable log-domain calculations avoid overflowing `t**alpha`
or `t**beta` at large iteration counts.

The factors are applied once per full-tree iteration, not once per player
update. DCFR uses one simultaneous full-tree traversal per iteration, updating
both players' regrets from the same frozen profile. Its average-strategy mass
is discounted independently from regret, as in Brown and Sandholm's
description of the algorithm.

## Validation

Focused tests cover RM+ positive-regret normalization and uniform fallback,
clipping after player updates, alternating-profile behavior, own-reach
averaging across two linearly weighted iterations, positive/negative DCFR
discounts, strategy-mass discounting, configurable/default parameters,
large-iteration numerical stability, deterministic seeds, normalized
strategies, and exact evaluator compatibility. The Leduc CFR+ own-reach test
reuses the repository's independent full-tree average-strategy reference.
Existing vanilla CFR, MCCFR, Kuhn equilibrium, and Leduc best-response tests
remain in place.

The exact best-response evaluators return the evaluation time separately from
training time. The training throughput in the experiment CSV excludes exact
evaluation.

## Reproducing the pilot

From the repository root, run:

```sh
PYTHONPATH=src python -m pokerlab.research.cfr_variant_experiments \
  --iterations 20 --seeds 11 23 47
```

The runner creates a new timestamped directory under `results/cfr_variants/`
and fails if that directory already exists. It writes per-seed checkpoint
records at iterations 5, 10, and 20, aggregate CSV, final average-strategy
snapshots, configuration files, exploitability plots against iterations and
training time, and a Git/artifact hash manifest. Records include expected
values for both players, NashConv, exact exploitability, elapsed training and
evaluation time, player updates, and traversal counts.

## Measured pilot

The measured run is
`results/cfr_variants/run-20261030T231813Z/`. It used three seeds, 20
iterations, and exact evaluations at 5, 10, and 20 iterations. The table gives
the across-seed mean and sample standard deviation at iteration 20. Full
per-seed values and intermediate checkpoints are in `convergence.csv`.

The earlier local pilot `results/cfr_variants/run-20261030T231556Z/` is
preserved, but superseded and excluded from this report. It was produced before
the CFR+ average-strategy pass was corrected to accumulate each player's
pre-update policy. The later run above uses the corrected code.

| Game | Algorithm | Exploitability (mean ± SD) | Training seconds (mean ± SD) | Traversals / player updates |
|---|---|---:|---:|---:|
| Kuhn | Vanilla CFR | 0.06628 ± 0.00000 | 0.0059 ± 0.0001 | 20 / 40 |
| Kuhn | CFR+ | 0.01011 ± 0.00000 | 0.0161 ± 0.0004 | 40 / 40 |
| Kuhn | DCFR | 0.05173 ± 0.00000 | 0.0062 ± 0.0001 | 20 / 40 |
| Kuhn | ES-MCCFR | 0.27369 ± 0.09195 | 0.0018 ± 0.0000 | 40 / 40 |
| Leduc | Vanilla CFR | 0.62415 ± 0.00000 | 1.3591 ± 0.0544 | 20 / 40 |
| Leduc | CFR+ | 0.14130 ± 0.00000 | 3.2719 ± 0.0295 | 40 / 40 |
| Leduc | DCFR | 0.43090 ± 0.00000 | 1.3721 ± 0.0044 | 20 / 40 |
| Leduc | ES-MCCFR | 2.51188 ± 0.11881 | 0.1488 ± 0.0021 | 40 / 40 |

The exact full-tree CFR, CFR+, and DCFR results were identical across seeds;
their seeds only affect the order in which exact chance deals are traversed.
ES-MCCFR samples hidden deals and therefore has seed-dependent results.

At this small iteration count, CFR+ had lower exploitability than vanilla CFR
in both games, with higher training cost. DCFR was slightly cheaper than CFR+
and had lower exploitability than vanilla CFR in this pilot. ES-MCCFR was
cheapest per iteration and had the highest exploitability at the same
iteration count. These results describe only this small diagnostic budget.
The Leduc wall-clock plot has no common measured-time range between the
20-iteration MCCFR endpoint (about 0.15 s) and the first full-tree checkpoint
(about 0.34 s); it should not be read as a time-matched winner comparison.
Kuhn has overlapping checkpoint times, but only a few milliseconds of
training were measured. Neither plot supports a general algorithm ranking.

## Limitations and next phase

The 20-iteration Leduc run is too short to assess convergence rates or
asymptotic behavior. Exact exploitability is expensive relative to Kuhn
training, and the short run does not establish robustness over seeds,
machines, or parameter settings. The benchmark uses exact full-tree
traversals, so these runtime results do not predict performance in an
abstracted HUNL tree.

Proceed to a Phase 4 abstracted HUNL prototype with CFR+ as an additional
solver candidate, while retaining vanilla CFR/DCFR and MCCFR as comparison
baselines. Validate the abstraction and payoff accounting independently
before treating the resulting exploitability estimates as representative of
full HUNL.

## References

- Oskari Tammelin (2014), [*Solving Large Imperfect Information Games Using
  CFR+*](https://arxiv.org/abs/1407.5042). The paper describes regret-matching+
  clipping, alternating player updates, and linearly weighted averaging used
  here; PokerLab uses a deliberately small exact game traversal.
- Noam Brown and Tuomas Sandholm (2019), [*Solving Imperfect-Information
  Games via Discounted Regret
  Minimization*](https://ojs.aaai.org/index.php/AAAI/article/view/4007). The
  paper defines sign-dependent cumulative-regret discounts and separate
  average-strategy discounts. PokerLab implements the specified DCFR
  recurrence in small Kuhn and Leduc games, without sampling or pruning.
