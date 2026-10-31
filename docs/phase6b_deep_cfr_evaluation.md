# Phase 6B: Deep CFR convergence and optimization study

## Scope and audit

Phase 6B evaluates the corrected small-game Deep CFR implementation in Kuhn
and Leduc. It does not train Deep CFR on HUNL. Existing game definitions and
tabular CFR, CFR+, DCFR, and MCCFR implementations were left unchanged.

The advantage target is an external-sampling estimate of

\[
 r_i^t(I,a)=\sum_{h\in I}\pi_{-i}^{\sigma^t}(h)
 \left(v_i^{\sigma^t}(h,a)-v_i^{\sigma^t}(h)\right).
\]

Conditioned on reaching `I`, sampling chance and opponent actions gives an
estimate scaled by the reciprocal of counterfactual reach
`pi_-i(I)`. That factor is common to all actions at `I`; traverser reach is
excluded from regret targets. For the average strategy, chance and the
strategy owner's sampled actions contribute their reach through traversal
frequency. Because traverser actions are enumerated, each strategy sample is
also weighted by traverser reach. The tests retain the independent
0.25:0.75 branch-weight check for this distinction.

No additional regret-equation defect was found in this audit. Training still
refits each network from scratch each iteration, as in the selected Deep CFR
formulation. The optimizer is iteration-local and has no persistent state to
checkpoint. Checkpoints preserve models, reservoirs, Python RNG state,
iteration, configuration, and diagnostics; an uninterrupted-versus-resumed
CPU test checks equivalence.

## Experiment design

The paired studies used seeds `11`, `23`, and `47`. Every seed/configuration
ran in its own subprocess so peak RSS is a per-run process high-water mark.
Kuhn ran for 500 iterations with exact evaluations every 100; Leduc ran for
100 iterations with exact evaluations every 50. Deep CFR used CPU, 64x64 ReLU
MLPs, Adam at `1e-3`, batch size 128, one training epoch per iteration,
20,000-sample advantage and strategy reservoirs, one traversal per player per
iteration, and linear iteration weighting. Strategy and advantage network
settings can now be tuned independently.

The hyperparameter ablations change one setting at a time from that baseline:

| Variant | Change |
|---|---|
| `traversals_x4` | Four traversals per player per iteration |
| `advantage_epochs_x4` | Four advantage-network epochs; strategy remains at one |
| `strategy_epochs_x4` | Four strategy-network epochs; advantage remains at one |
| `reservoir_512` | Both reservoir capacities reduced from 20,000 to 512 |
| Kuhn only: `advantage_lr_3e-4` | Advantage learning rate reduced to `3e-4`; strategy stays at `1e-3` |
| Kuhn only: `wider_advantage_network` | Advantage MLP changed from 64x64 to 128x128; strategy stays at 64x64 |

The reservoir comparison changes both related reservoir capacities together;
it cannot identify which reservoir is responsible for a change. The studies
use exact exploitability and compare policies over identical information sets
and legal actions. Strategy distance is the unweighted mean per-information-set
total variation distance, `mean_I(0.5 * sum_a |p(a|I)-q(a|I)|)`. It is a
descriptive approximation metric, not a reach-weighted exploitability proxy.

Confidence intervals are two-sided Student-t intervals over paired seeds
(critical value 4.303 for three seeds). Paired configuration deltas use the
same seed and iteration for the variant and baseline. The ablations are
exploratory; intervals are not adjusted for multiple comparisons. Tabular
solvers are deterministic in these full-tree runs, so their across-seed
intervals collapse to the same value and should not be mistaken for broad
uncertainty bounds.

## Exact convergence and tabular comparisons

At the final checkpoints, mean exact exploitability was:

| Game/checkpoint | Deep CFR baseline | Vanilla CFR | CFR+ | DCFR |
|---|---:|---:|---:|---:|
| Kuhn, 500 | 0.43062 (95% CI 0.42052–0.44072) | 0.01068 | 0.000173 | 0.00796 |
| Leduc, 100 | 2.35888 (95% CI 2.31894–2.39881) | 0.16651 | 0.01646 | 0.07199 |

Lower is better. Deep CFR remained substantially more exploitable than every
tabular reference at both measured budgets. In Kuhn, baseline Deep CFR moved
from 0.45385 at iteration 100 to 0.43062 at iteration 500. In Leduc, it moved
from 2.36746 at iteration 50 to 2.35888 at iteration 100. These are modest
changes and do not establish convergence.

Selected ablations:

| Game | Variant | Deep CFR exploitability at final checkpoint (95% CI) | Paired delta vs baseline (95% CI) | Deep training time |
|---|---|---:|---:|---:|
| Kuhn, 500 | Four traversals/player | 0.32530 (0.31466–0.33595) | −0.10531 (−0.12092 to −0.08971) | 12.09 s |
| Kuhn, 500 | Four strategy epochs | 0.33014 (0.32043–0.33985) | −0.10048 (−0.11903 to −0.08193) | 6.58 s |
| Kuhn, 500 | Four advantage epochs | 0.42345 (0.40941–0.43748) | −0.00717 (−0.02202 to 0.00767) | 6.99 s |
| Kuhn, 500 | Reservoirs of 512 | 0.44998 (0.44740–0.45256) | +0.01936 (+0.01073 to +0.02799) | 2.88 s |
| Leduc, 100 | Four traversals/player | 1.97600 (1.49095–2.46105) | −0.38288 (−0.90637 to +0.14061) | 3.00 s |
| Leduc, 100 | Four strategy epochs | 2.03061 (1.79453–2.26669) | −0.32826 (−0.54141 to −0.11511) | 1.54 s |
| Leduc, 100 | Four advantage epochs | 2.31812 (2.22735–2.40890) | −0.04075 (−0.16115 to +0.07965) | 1.45 s |
| Leduc, 100 | Reservoirs of 512 | 2.37403 (2.36120–2.38686) | +0.01515 (−0.01515 to +0.04545) | 0.92 s |

The paired intervals suggest that additional strategy fitting improved mean
exploitability in these small runs, especially Kuhn. Four traversals also
improved Kuhn but cost roughly three times the baseline Deep CFR training
time. On Leduc, the traversal ablation had high seed variance and its paired
interval includes zero. Neither result establishes that the setting will
generalize to longer training.

Lower supervised loss was not treated as proof of better play. In Kuhn,
four strategy epochs lowered mean strategy loss at iteration 500 from about
0.207 to 0.137 while improving exploitability, but other loss reductions did
not consistently improve exact exploitability. Strategy TV also did not track
exploitability monotonically: the better Kuhn traversal and strategy-epoch
variants had larger unweighted TV distance from tabular strategies than the
baseline. This reinforces that TV and training loss are diagnostics, not
equilibrium metrics.

## Runtime and memory

Mean Deep CFR training time at the final checkpoint was 4.06 s versus 0.157 s
for vanilla CFR, 0.418 s for CFR+, and 0.172 s for DCFR in Kuhn. In Leduc it
was 1.69 s versus 6.64 s, 16.70 s, and 7.07 s respectively. Training time
excludes exact evaluation. The experiment folders plot exploitability against
both iterations and training seconds; saved checkpoints are relatively sparse,
so the time plot is a curve comparison rather than interpolation at identical
wall-clock instants.

The isolated memory comparison found little process-RSS difference because
the PyTorch runtime dominates these small games:

| Game/checkpoint | Both reservoirs 20,000 | Both reservoirs 512 | Observation |
|---|---:|---:|---|
| Kuhn, 500 | 321.8 MiB mean peak RSS | 329.7 MiB | 512-item reservoirs filled and replaced samples; no measured RSS reduction |
| Leduc, 100 | 299.2 MiB mean peak RSS | 321.4 MiB | Strategy reservoir filled; advantage reservoir filled in only one seed |

RSS variation across fresh processes exceeded any expected small-game memory
saving. Do not interpret these small differences as evidence that smaller
reservoirs consume more memory. In the Kuhn capacity runs, advantage memory
replacements were 301–338 and strategy replacements 352–466 by iteration 500.
In Leduc, strategy replacements were 253–373 by iteration 100; advantage
memory had 0–26 replacements. These counts expose actual reservoir saturation.

The host's Apple Silicon MPS backend is built into PyTorch but was unavailable
at runtime, so all measured runs used CPU. Peak RSS is whole-process memory,
not isolated model or reservoir allocation. No CUDA was assumed.

## Artifacts and provenance

The principal study directories are:

- `results/deep_cfr/study-kuhn-20261031T025240Z-3e445d19/` — seven 500-iteration
  configurations across three seeds, including strategy distances, losses,
  paired deltas, and plots.
- `results/deep_cfr/study-leduc-20261031T025512Z-59ab6563/` — five 100-iteration
  configurations across three seeds.
- `results/deep_cfr/study-kuhn-20261031T030607Z-9ec9707a/` and
  `results/deep_cfr/study-leduc-20261031T030651Z-408e6139/` — isolated-process
  reservoir-capacity comparisons with per-run RSS.
- `results/deep_cfr/study-kuhn-20261031T031113Z-c921e000/` and
  `results/deep_cfr/study-leduc-20261031T031137Z-73ebd9ec/` — final paired
  baseline comparisons including DCFR.

Every complete run has its own configuration, convergence/evaluation CSV,
training diagnostics, strategy snapshots, checkpoints, summary, and SHA-256
manifest. Study directories contain paired-seed summaries, confidence
intervals, plots, and run indexes. Artifact hashes were rechecked after
analysis.

The first broad multi-seed studies ran all configurations in one parent
process, so their `ru_maxrss` values are cumulative and are not used for the
memory conclusions above. During those runs, only the study orchestration file
changed; the per-run Deep CFR trainer, traversal, network, encoder, and game
source hashes match the implementation used. Each broad study contains a
provenance note documenting that orchestration limitation. The later
capacity-specific and four-solver baseline studies ran each configuration in
an isolated process with stable source files.

Reproduce the standard paired study with:

```sh
PYTHONPATH=src python -m pokerlab.algorithms.deep_cfr.study \
  --game kuhn --iterations 500 --evaluation-interval 100 \
  --seeds 11 23 47 --output-root results/deep_cfr
```

Use 100 iterations and an evaluation interval of 50 for the documented Leduc
budget. The default study CLI compares the baseline with more traversals,
additional advantage epochs, additional strategy epochs, and smaller paired
reservoir capacities.

## Conclusions and limits

Deep CFR is operational, reproducible under the tested CPU setup, and its
small-game estimator and average-strategy weighting are covered by independent
tests. More strategy fitting and traversal improved Kuhn results but did not
close the large gap to tabular CFR. In Leduc, the apparent gains are noisier.
The 512-capacity comparison saved training time in the tested configurations
but did not improve exploitability or reduce measured process RSS.

Only three seeds and modest iteration counts were tested. The intervals are
wide or underpowered for some Leduc comparisons, ablation intervals are
unadjusted for multiple testing, only one primary architecture and optimizer
family was used, and long-budget behavior remains unknown. Input normalization,
warm-up schedules, larger seed counts, isolated advantage-versus-strategy
reservoir capacity sweeps, and longer Kuhn/Leduc runs remain open. This phase
does not support HUNL training or a claim that Deep CFR outperforms the
tabular solvers.
