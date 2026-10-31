# Phase 5B: HUNL abstraction evaluation

## Design

The benchmark compares five engine-backed abstract HUNL configurations with
the same 10 BB starting stack, 0.5/1 BB blinds, MCCFR implementation, training
budget and seeds:

| Name | Hand abstraction | Action profile |
|---|---|---|
| `baseline` | 20-bucket sampled strength | legacy |
| `equity_8` | 8-bucket sampled equity | legacy |
| `equity_20` | 20-bucket sampled equity | legacy |
| `action_wide` | 20-bucket sampled strength | wide |
| `combined_wide` | 20-bucket sampled equity | wide |

The sampled-equity configurations use 16 equity samples per feature by default.
Baseline opponents are Random, Heuristic and Equity (32 simulations per
decision). Those three agents receive a fixed legacy action menu for every
variant. Previous-MCCFR comparisons use the preceding saved checkpoint from
the same configuration, so those comparisons necessarily stay within the
same abstract game.

Each evaluation batch consists of independent hands with equal fresh stacks.
The button is fixed for that batch; a paired batch is run with the trained
agent on the button and another with its opponent on the button. Deck and
agent seeds are deterministic and shared across variants for the same training
seed, checkpoint, opponent and button role. This design always evaluates the
requested hand count even if one policy loses a hand's full starting stack.
It does not simulate a continuous bankroll or tournament elimination.

The default budget is 20 MCCFR iterations, checkpoints 5/10/20, four training
seeds (11, 23, 47, 89), and 100 hands per opponent/button batch. All variants
use identical seeds. Config JSON records the concrete game settings and
evaluation plan. The manifest records source revision, dirty-tree status,
platform, Python version and SHA-256 hashes for source and generated artifacts.
Every run receives a new timestamped directory; existing Phase 4/5A results
are not reused or replaced.

## Measurements and statistics

Training CSV rows record iteration timing, sampled nodes, traversals,
information-set count, cumulative positive and negative regret, average
strategy entropy and approximate regret/strategy table sizes. Checkpoint
metadata additionally records elapsed training time and process peak RSS.
Table-size values are Python object-size estimates, not total process memory.
RSS is a process high-water mark and may include earlier work in that process.

For each fixed-hand batch, evaluation CSV records total trained-agent net
chips, BB/100, hand wins/ties, strict batch win/tie, button role, actual hand
count, agent decision time and wall time. CSV summary means and Student-t 95%
intervals are calculated across the independent training-seed batches, not
across individual hands. Paired differences match training seed, checkpoint,
opponent, hand seed and button role. Strict batch-win proportions use Wilson
score intervals. Four seeds provide weak uncertainty estimates; the intervals
are descriptive and should not be treated as evidence of equilibrium quality.
Poker hands within a batch are also not independent of the policy's evolving
private RNG stream, so pooled hand-level binomial intervals are not used.

The benchmark does not calculate exact HUNL exploitability. Better results
against these three fixed baselines show performance only against those
opponents and this finite abstract game. They do not establish Nash
convergence. Action profiles change the legal action tree and remain distinct
abstract games, despite common evaluation-opponent menus.

## Reproduction

Estimate cost before running the full suite:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark --estimate-only
```

Run the default multi-seed comparison:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark
```

For a smaller pipeline validation, for example:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark \
  --iterations 2 --checkpoints 1 2 --seeds 11 \
  --variants baseline equity_8 --evaluation-hands 10
```

Each run writes `config.json`, `manifest.json`, `training.csv`,
`evaluation.csv`, `summary.json`, statistical summaries, a performance plot,
logs, and independent strategy/state/metadata files at every checkpoint.
Checkpoint state contains the Python RNG state and supports exact continuation
with `--resume-checkpoint` for one variant and one seed. Resumed runs go into a
new output directory.

## Results

The default run is saved at
`results/nlhe_abstraction/run-20261031T002211Z-40fe79f2/`. It completed all
400 training iterations and 440 fixed-hand batches (44,000 hands). Summed
iteration training time was 23.76 seconds; summed evaluation batch wall time
was 238.05 seconds. Training throughput was 16.83 iterations/second, calculated
using training time only. Evaluation averaged 0.54 seconds per 100-hand batch
(about 185 hands/second). The full run took about 267 seconds from its timestamped
directory creation to summary generation. Peak process RSS recorded in
checkpoint metadata was 120,307,712 bytes (about 114.7 MiB); the checkpoint
directory occupies about 175 MB. Estimated regret/strategy tables were at most
about 1.53 MB for a seed/configuration. These are pilot-scale measurements on
this machine.

At iteration 20, the table reports mean BB/100 after first averaging the two
button-role batches within each seed. Intervals are two-sided 95% Student-t
intervals over the four independent training seeds (df=3):

| Opponent | Baseline | Equity 8 | Equity 20 | Wide actions | Equity 20 + wide |
|---|---:|---:|---:|---:|---:|
| Random | 6.19 [−48.19, 60.57] | 6.19 [−48.19, 60.57] | 6.19 [−48.19, 60.57] | 10.00 [−46.84, 66.84] | 10.00 [−46.84, 66.84] |
| Heuristic | 12.62 [−14.84, 40.09] | 12.62 [−14.84, 40.09] | 12.62 [−14.84, 40.09] | 9.31 [−20.13, 38.75] | 9.31 [−20.13, 38.75] |
| Equity | −21.25 [−64.83, 22.33] | −21.25 [−64.83, 22.33] | −21.25 [−64.83, 22.33] | −31.94 [−61.34, −2.53] | −31.94 [−61.34, −2.53] |

Paired changes versus baseline, again averaged over button role within each
seed, were 0.00 BB/100 for both equity-only variants against every baseline
opponent. The wide-action and combined-wide variants had the same observed
results at this budget. Their paired differences versus baseline were +3.81
[−63.33, 70.95] against Random, −3.31 [−29.11, 22.48] against Heuristic, and
−10.69 [−43.62, 22.24] against Equity. These intervals all include zero.

At iteration 20, trained-hand win rates excluding tied hands were 49.3%, 70.3%
and 60.4% for the baseline versus Random, Heuristic and Equity respectively;
the wide action policy rates were 53.9%, 73.3% and 61.7%. Across the four
seeds, strict 100-hand batch wins were 4/8, 5/8 and 3/8 for baseline, versus
Random, Heuristic and Equity; the wide policy went 5/8, 6/8 and 1/8. Positive
hand win rates alongside negative BB/100 against Equity show why hand win
percentage alone is not a sufficient performance measure: losses can be larger
than wins.

The equity configurations and baseline had identical observed batch outcomes
in this run; the combined-wide and wide-only configurations also matched. This
does not show that their strategies or information sets are identical. They
were trained and evaluated with distinct saved configurations, but this
20-iteration, 100-hand-per-seed budget did not produce distinguishable sampled
outcomes for those pairs. The wide menu changed the action tree and produced
different outcomes, but its intervals remain broad. There is no evidence here
that one abstraction is generally stronger. Exact HUNL exploitability is not
available.

All 189 artifact hashes listed in the run manifest verified after generation.
The report and CSVs preserve all checkpoints and the full 5/10/20-iteration
curves; the table above is only the final checkpoint summary. No performance
conclusion is made from the Phase 5A one-hand smoke runs.

## Limitations and next step

The 10 BB, 20-iteration suite is a controlled implementation benchmark, not a
strong-play study. Four seeds and 100 hands per seed/batch will still leave
large poker outcome variance. The same seed set improves pairing but does not
make the game trees identical across abstractions. The equity baseline is also
an approximate fixed policy. Measurements do not estimate exploitability.

`app.py` is empty and there is no existing dashboard architecture to extend;
the experiment artifacts and CSV/HTML plot are the research interface for this
phase. A next research step should increase training duration and independent
seeds after reviewing runtime and memory, and use a stronger evaluation design
with more hands and more varied opponent policies. Exact exploitability is not
available for this HUNL abstraction, so any equilibrium claim would need a
separate justified method.
