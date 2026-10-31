# Phase 3C.1: MCCFR convergence and variance investigation

## Scope and outcome

This investigation audited the existing Kuhn and Leduc external-sampling MCCFR
solvers, measured regret-estimator variation at fixed average strategies, and
compared exact exploitability across seeds, iteration counts, and training time.
No game rules, vanilla CFR equations, or MCCFR production equations were
changed. No confirmed production correctness defect was found.

The data supports estimator variance and sparse sampled coverage as important
contributors to Leduc MCCFR's poor finite-budget convergence. It does not show
that variance is the only cause: no controlled variance-reduction ablation was
run. Both MCCFR games improved at larger budgets, but Leduc remained far behind
vanilla CFR at the measured budgets and matched training times.

The completed experiment is in
[`results/mccfr/diagnostics/run-20261030T214955Z`](../results/mccfr/diagnostics/run-20261030T214955Z).
Its manifest records Git HEAD, dirty-tree state, relevant source hashes, and
artifact hashes. Seeds were `7, 19, 31, 43, 59`.

## Mathematical audit

The reviewed implementations are
[`kuhn/mccfr.py`](../src/pokerlab/research/kuhn/mccfr.py),
[`leduc/mccfr.py`](../src/pokerlab/research/leduc/mccfr.py), the corresponding
vanilla CFR modules, exact evaluators, strategy catalogs, and experiment code.

At a traverser information set, each external-sampling traversal enumerates all
legal traverser actions while sampling chance and opponent actions from their
true current distributions. If an information set is not encountered on the
sampled external path, that trial contributes a zero increment. For a reached
set, the trial's action-value difference has no extra reach multiplier. Taking
expectation over the sampled chance/opponent block gives

```text
E[delta_i(I,a)] = sum_{h in I} pi_chance(h) pi_opponent(h)
                  * (v_i(h,a) - v_i(h))
```

which is the vanilla counterfactual regret increment. Traverser reach is
excluded from this regret weight. Leduc's initial ordered private deal and
later public card are sampled according to their true chance probabilities.
Regret deltas for both players are accumulated against one frozen strategy
profile and committed after both traversals.

Average strategy is a separate calculation. The implementations enumerate the
cataloged information sets and add `own_reach(I) * sigma_i(I,a)` each iteration,
including sets omitted by sampled regret paths. This matches the repository's
vanilla averaging convention; it is not an estimator that inherits the regret
sampling variance. Hidden cards are absent from information-set keys and do not
select an action.

The convergence runner uses each solver's average strategy and exact expected
value/best-response evaluators. NashConv and exploitability use the existing
zero-sum payoff convention; the reported Player 1 expected value is the
negative of Player 0's value. Evaluation time is separate from training time.

## Fixed-profile regret estimator

The probe fixes the seed-7, 1,000-iteration MCCFR average strategy from the
existing 50-checkpoint experiment. For each selected information set it draws
independent private deals and external trajectories, records a zero for every
miss, and calculates sample means, sample variances, standard errors, and
nominal 95% Wald intervals. Expected increments and counterfactual reach are
computed by an independent full-tree recursion in
[`mccfr_investigation.py`](../src/pokerlab/research/mccfr_investigation.py);
the reference does not call either production CFR regret update. The stored
exact reference values were independently recomputed from the snapshots with
zero discrepancy. Saved standard errors equal `sqrt(sample_variance / trials)`.

The 25,000-trial table covers 3 Kuhn information sets (6 action records) and 5
Leduc information sets (12 action records). The chosen Leduc states span
pre-flop and flop decisions, including shared hidden-state histories:

| Leduc information set | Exact counterfactual reach | Visits / 25,000 | Visit rate |
|---|---:|---:|---:|
| `p0|J0|-|r0|root|-` | 0.166667 | 4,173 | 16.69% |
| `p1|Q0|-|r0|bet|-` | 0.060933 | 1,575 | 6.30% |
| `p1|J0|J1|r1|check/check|-` | 0.020637 | 517 | 2.07% |
| `p0|J0|K0|r1|check/check|bet` | 0.004610 | 100 | 0.40% |
| `p1|J0|J1|r1|check/check|bet/raise` | 0.006373 | 152 | 0.61% |

The empirical visit rates track the independently calculated external reach.
The deepest selected sets therefore have zero increments on about 99.4–99.6%
of trials. The root bet estimate has sample variance 5.106 and is zero on
83.3% of trials. The Leduc pre-flop raise estimate at `p1|Q0|-|r0|bet|-` was
the closest 25,000-trial mean to the nominal 95% boundary: sample mean
`-0.036857`, exact mean `-0.018158`, standard error `0.009423`, standardized
error `-1.984`. This is a single borderline comparison among 18 action
estimates, so it is not evidence of estimator bias by itself.

The same fixed profile and information set were sampled for 100,000 trials.
For the raise action, the empirical mean was `-0.020488`, the exact reference
remained `-0.018158`, standard error was `0.004703`, and standardized error was
`-0.495`; it was visited 6,140 times (6.14%, versus exact external reach
6.093%). This supports convergence of the estimator mean to the exact update
as samples increase. It does not establish a uniform error bound for every
information set or strategy profile.

See [`regret_variance.csv`](../results/mccfr/diagnostics/run-20261030T214955Z/regret_variance.csv),
[`regret_variance_leduc_100k_robustness.csv`](../results/mccfr/diagnostics/run-20261030T214955Z/regret_variance_leduc_100k_robustness.csv),
and [`data_validation.json`](../results/mccfr/diagnostics/run-20261030T214955Z/data_validation.json).

## Information-set coverage

Coverage probes used 500 sampled traversals per traverser for each recorded
profile. The canonical table with training-seed attribution is
[`information_set_coverage_by_seed.csv`](../results/mccfr/diagnostics/run-20261030T214955Z/information_set_coverage_by_seed.csv).
Each Leduc profile has 936 information sets. Under the five 10k MCCFR average
profiles, 163–201 sets were not seen in 500 probe traversals; the median
per-traversal visit probability across sets was 0.4–0.6%. Under the four fresh
5k vanilla profiles, 307–338 sets were missed in the same probe budget and the
median was 0.2–0.4%. Seed 7's first coverage record used its saved 1k pilot
profile; a separate probe of the recovered historical 5k profile is saved in
[`information_set_coverage_seed7_vanilla_5k_recovered.csv`](../results/mccfr/diagnostics/run-20261030T214955Z/information_set_coverage_seed7_vanilla_5k_recovered.csv).

These numbers describe how often a particular information set appears under
the *external-sampling probe* for the recorded strategy. The vanilla profiles
were given the same sampling probe for comparison; the counts are not vanilla
CFR's actual iteration coverage, since vanilla CFR traverses the full tree.
The probe does not imply that an information set was never visited during
training. It shows that low-reach sets receive few independent sampled regret
signals at these budgets.

## Multi-seed convergence

All values below are exact exploitability in chips per hand. Intervals are
95% Student-t intervals across seeds. Vanilla Leduc 10k was not run: existing
measurements put it near eleven minutes per seed, while 5k provides a matched
five-seed comparison. The historical seed-7 5k trajectory was reused rather
than rerun.

| Game | Solver | Iterations | Seeds | Mean | Median | SD | Min–max | 95% CI for mean | Mean training seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kuhn | Vanilla CFR | 1k | 5 | 0.007269 | 0.007269 | 0 | 0.007269–0.007269 | 0.007269–0.007269 | 0.284 |
| Kuhn | ES-MCCFR | 1k | 5 | 0.033510 | 0.035237 | 0.004834 | 0.028216–0.038952 | 0.027509–0.039511 | 0.088 |
| Kuhn | Vanilla CFR | 5k | 5 | 0.003302 | 0.003302 | 0 | 0.003302–0.003302 | 0.003302–0.003302 | 1.417 |
| Kuhn | ES-MCCFR | 5k | 5 | 0.011883 | 0.011615 | 0.003041 | 0.007821–0.016031 | 0.008108–0.015658 | 0.437 |
| Kuhn | Vanilla CFR | 10k | 5 | 0.002318 | 0.002318 | 0 | 0.002318–0.002318 | 0.002318–0.002318 | 2.841 |
| Kuhn | ES-MCCFR | 10k | 5 | 0.006754 | 0.006149 | 0.004039 | 0.002215–0.013224 | 0.001740–0.011769 | 0.873 |
| Leduc | Vanilla CFR | 1k | 5 | 0.035678 | 0.035678 | <0.000001 | 0.035678–0.035678 | 0.035678–0.035678 | 67.323 |
| Leduc | ES-MCCFR | 1k | 5 | 1.246572 | 1.235498 | 0.047067 | 1.200131–1.326018 | 1.188141–1.305004 | 7.272 |
| Leduc | Vanilla CFR | 5k | 5 | 0.015669 | 0.015679 | 0.000636 | 0.014781–0.016366 | 0.014879–0.016459 | 325.491 |
| Leduc | ES-MCCFR | 5k | 5 | 0.468099 | 0.473014 | 0.017103 | 0.448703–0.488854 | 0.446866–0.489332 | 36.542 |
| Leduc | ES-MCCFR | 10k | 5 | 0.300868 | 0.308789 | 0.027763 | 0.257926–0.327567 | 0.266400–0.335334 | 72.940 |

MCCFR's mean Player 0 value varied substantially in Leduc: its across-seed
standard deviation was 0.1033 at 1k, 0.0369 at 5k, and 0.0259 at 10k. Vanilla
Leduc values were effectively identical at 1k and much less variable at 5k.
The multi-seed exact exploitability results show improvement as MCCFR receives
more iterations, but its mean at 10k remained about 19 times the five-seed
vanilla mean at 5k and about 8.4 times the vanilla mean at 1k. (The iteration
counts differ.)

The 10k vanilla Kuhn and Leduc MCCFR profiles are saved as per-seed JSON files
alongside the raw per-seed CSVs. The historical 5k Leduc seed-7 profile is
documented below.

## Iterations and wall-clock time

The exact aggregate curves are in
[`aggregated_comparison.csv`](../results/mccfr/diagnostics/run-20261030T214955Z/aggregated_comparison.csv)
and [`time_aggregated.csv`](../results/mccfr/diagnostics/run-20261030T214955Z/time_aggregated.csv).
Plots show the mean and seed-level 95% intervals:

- [Kuhn exploitability versus iterations](../results/mccfr/diagnostics/run-20261030T214955Z/kuhn_exploitability_vs_iterations.html)
- [Kuhn exploitability versus training time](../results/mccfr/diagnostics/run-20261030T214955Z/kuhn_exploitability_vs_training_time.html)
- [Leduc exploitability versus iterations](../results/mccfr/diagnostics/run-20261030T214955Z/leduc_exploitability_vs_iterations.html)
- [Leduc exploitability versus training time](../results/mccfr/diagnostics/run-20261030T214955Z/leduc_exploitability_vs_training_time.html)

At about 70 training seconds, both algorithms had five-seed support: vanilla
Leduc's interpolated mean exploitability was 0.0354, while MCCFR's was 0.3139.
At 70 seconds MCCFR had completed close to 10k iterations; vanilla had just
passed 1k. At a matched 1k iterations, MCCFR was roughly nine times faster but
had roughly 35 times the exploitability. The Kuhn comparison was much smaller
but pointed the same way: at 1k iterations, MCCFR was faster and less accurate;
at 0.5 training seconds, interpolated means were 0.0111 MCCFR (n=5) versus
0.0065 vanilla (n=5).

Time curves linearly interpolate only between each seed's recorded exact
evaluation checkpoints and never extrapolate beyond that seed's measured
range. The seed-7 vanilla 5k training time was reconstructed from its recorded
total runtime by subtracting cumulative best-response durations; the other
four 5k vanilla training times were measured directly. Leduc vanilla has only
one seed supporting very early wall-clock times; five-seed support begins
around its 1k checkpoint. Individual seed data is
available in `per_seed_all_runs.csv` and the `raw/` directory. Exploitability
and strategy quality need not improve monotonically at every checkpoint.

## Historical 5k vanilla Leduc snapshot

The historical average strategy was recovered, so no substitute strategy was
used. [`results/leduc/average_strategy.json`](../results/leduc/average_strategy.json)
has SHA-256
`55153bfdf8f2315ec4ffa7178feaa97b2bdd6fcc39be74b8259cdc018fb520cd`, exactly
matching `BASELINE_MANIFEST.json` in the Phase 3B baseline archive. The archived
configuration records seed 7, 5,000 iterations, evaluation every 500, and
logging every 500; the Leduc CFR, evaluator, game, information-set,
serialization, and experiment-runner source hashes all match the archived
source at Git HEAD `f5153543824efa567dfdd804b204e6d3795e4969`. The original
shell invocation and dependency environment were not embedded in the CSV, but
the trained strategy, convergence output, config, source revision, and relevant
source modules are hash-verified.

The recovered copy is
[`strategy_leduc_vanilla_cfr_seed_7_iter_5000_historical.json`](../results/mccfr/diagnostics/run-20261030T214955Z/strategy_leduc_vanilla_cfr_seed_7_iter_5000_historical.json).
The exact evaluator returned Player 0 value `-0.008097667521084559`, NashConv
`0.02956281041436874`, and exploitability `0.01478140520718437`, exactly equal
to the iteration-5,000 historical CSV row (absolute differences are zero in
the saved record). See
[`seed7_vanilla_5k_revalidation.json`](../results/mccfr/diagnostics/run-20261030T214955Z/seed7_vanilla_5k_revalidation.json).

## Findings and next experiments

**Confirmed:** both MCCFR implementations use sampled chance/opponent actions,
enumerate traverser actions, omit traverser reach from regret weighting, freeze
the profile across player traversals, and keep exact own-reach strategy
averaging separate from sampled regret updates. Fixed-profile empirical means
agree with independently computed exact increments at the measured scale; the
100k follow-up resolves the one borderline 25k estimate. Exact average-profile
exploitability improves with budget in these runs but is materially worse than
vanilla CFR at both equal iterations and the measured equal-time points.

**Supported explanation, not an isolated causal proof:** Leduc has 936
information sets, while one MCCFR iteration samples only two private-deal
traversals. Low-probability states therefore produce many zero regret
increments and high conditional variance. This explains seed sensitivity and
slow finite-budget learning without implying bias. Exact averaging rules out
sampled visitation bias in the average-strategy update as the source of the
observed gap. Other causes, including step-size effects from unweighted
stochastic regret accumulation, were not separately ablated.

Recommended next research step is a controlled **stratified external-sampling**
experiment before implementing Outcome Sampling:

| Candidate | Expected benefit | Cost | Main correctness risk | Validation |
|---|---|---|---|---|
| Random-permutation/cyclic coverage of private deals in batches | Reduce deal-level variance and seed sensitivity for card-specific sets | Low | A deterministic schedule can correlate with changing strategies across iterations | Compare batch-summed increments to exact CFR, test random-permutation batches, and measure variance at equal traversal count |
| Selective public-card enumeration at rare/high-variance nodes | Rao-Blackwellize chance noise for chosen states | Moderate to high | Accidentally omit chance weights or double-count information-set histories | Exact mean comparison and variance/runtime ablation on fixed profiles |
| Adaptive full-support proposal for chance/opponent branches | Spend samples on rare states | Moderate | Missing or unstable importance ratios can bias or explode variance | Verify support, independently calculate proposal ratios, compare sample means and tails to exact updates |
| Outcome Sampling MCCFR | Lower traversal cost per iteration | High | More complex importance-weighted regret estimator; sparse high-variance updates may worsen | Do not adopt until estimator tests cover zero-probability support and exact expectation on Kuhn/Leduc |

Do not change average-strategy weighting: it already matches vanilla's own
reach convention. Performance profiling can follow after sampling variance is
addressed; current MCCFR updates visit only about 38–44 nodes per Leduc
iteration versus 9,450 for vanilla, yet wall time improves by about ninefold,
partly because the exact average-strategy pass still processes the catalog.

## Reproducibility and limitations

Start a new, non-overwriting diagnostics run with:

```sh
PYTHONPATH=src python -m pokerlab.research.mccfr_investigation \
  --seeds 7 19 31 43 59 \
  --kuhn-iterations 10000 \
  --leduc-mccfr-iterations 10000 \
  --leduc-vanilla-iterations 5000 \
  --variance-trials 25000 \
  --coverage-trials 500
```

The command creates a new timestamped subdirectory and refuses to overwrite an
existing run. The completed run used only five seeds, did not run 10k vanilla
Leduc, and reused seed-7 historical convergence points. Wall-time values are
machine-specific. Per-trial regret increments were reduced to means/variances
rather than retained as a large trace; the seed, sample count, reach, visits,
variance, confidence interval, and exact reference are retained. No peak RSS
measurement was made. Confidence intervals summarize these finite seed/sample
sets and are not guarantees about future seeds.
