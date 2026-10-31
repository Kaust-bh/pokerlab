# Phase 3C.2: Stratified External Sampling and Selective Chance Enumeration

## Scope

This phase adds isolated experimental chance-sampling variants beside the
existing ES-MCCFR solvers. Vanilla CFR and the Phase 3C baseline remain
unchanged. Every variant preserves the same frozen strategy profile across
both traversers and the exact own-reach average-strategy update.

Results and source provenance are saved under
`results/mccfr/phase3c2/`. This document separates mathematical guarantees from
measured results; no variance reduction is presumed from the names of the
methods.

## Target regret and sampling notation

For traverser `i` and information set `I`, let `g_i(I,a,ω)` be the signed
action-value difference for one external block `ω`, after enumerating the
traverser's legal actions. The desired vanilla CFR update is

```text
R_i(I,a) = sum_h pi_chance(h) pi_-i(h)
             * [v_i(h,a) - v_i(h)]
```

where `pi_-i` is the opponent's reach. Traverser reach is excluded. In ordinary
external sampling, chance and opponent choices are drawn from their true
distributions, so the sampled local difference has the desired expectation;
an information set absent from that sample contributes zero.

### A. Private-deal stratification

Partition private deals into disjoint nonempty strata `S_j`. Let
`p_j = sum_{x in S_j} p(x)`, and draw `m_j` deals independently from
`q_j(x) = p(x) / p_j` within each stratum. Conditional on a private deal, the
solver samples public chance (when present) and opponent actions from their
true probabilities. For one traverser, use

```text
R_hat_i(I,a) = sum_j p_j / m_j
                   * sum_{k=1..m_j} g_i(I,a, x_jk, sampled_external_choices)
```

For each stratum,
`E[g | x ~ q_j] = sum_{x in S_j} p(x) E[g | x] / p_j`; multiplying by `p_j`
and summing strata recovers the full chance sum. There are no proposal-ratio
weights because sampling inside each stratum uses the true conditional chance
law. The per-sample factor is `p_j / m_j`, not the traverser's own reach.

Poker-specific strata are chosen from private-card strength structure:

- **Kuhn:** Player 0 holds a King (`2/6`) versus a Jack or Queen (`4/6`).
- **Leduc:** the two private cards have the same rank (`6/30`) versus different
  ranks (`24/30`).

These partitions separate materially different private-hand classes. Their
probabilities are unequal; the implementation must retain those masses. One
draw per stratum costs two private-deal traversals per traverser instead of
one. `m_j > 1` is supported using the same `p_j / m_j` factor; a dedicated
multi-sample-per-stratum test is still needed.

This construction is unbiased, but it is not assumed to have lower variance
for every information set. With one independent draw in each stratum, its
variance is `sum_j p_j^2 Var(g | S_j)`, while `K` i.i.d. baseline draws have
variance `Var(g) / K`. Between-stratum variation is removed, but the within-
stratum variances and unequal masses determine the result. The variance
comparison is measured at equal root-sample and node-touch budgets.

### B. Selective chance enumeration

For a chance variable `X` with true outcomes `x` and probabilities `p(x)`,
replace its sampled contribution by the conditional sum

```text
g_bar = sum_x p(x) g(x, sampled_other_choices)
```

This is a Rao-Blackwellized estimator: averaging over the enumerated chance
outcome preserves its expected contribution, and conditional expectation cannot
increase variance for the same coupled remaining random choices. Descendant
regret increments receive the enumerated chance probability; returned action
values are probability-weighted expectations. Upstream counterfactual regret
updates must not apply that chance weight a second time. Opponent actions that
remain sampled continue to use their true policy probabilities without extra
weights.

Leduc has 30 equally likely ordered private deals and, after a non-folded
pre-flop round, four equally likely public cards. The four separately measured
configurations are:

| Private deal | Public card | Chance contribution at a local regret update |
|---|---|---|
| Sample | Sample | True sampling law; no added weight |
| Enumerate | Sample | `1/30` for each private deal; public law sampled |
| Sample | Enumerate | Private law sampled; `1/4` for each available public card |
| Enumerate | Enumerate | `1/30 * 1/4` for each compatible deal/board pair |

The board has four cards remaining for every legal private deal. If betting
folds before the board, no public chance node is visited. Exact enumeration
costs more work and is not automatically preferable; its variance reduction
must be judged against elapsed time and node touches. Kuhn has no public card,
so its chance-enumeration variant enumerates its six private deals.

### Combined configuration

Private-deal stratification combined with public-card enumeration uses
`p_j / m_j * p(public | private)` at a sampled private stratum path. Its
expectation follows by applying the two sums successively. It is evaluated only
after both component configurations pass their separate exact-expectation
tests. Full private-deal enumeration plus public-card enumeration is the
enumerate/enumerate configuration above, not an additional proposal.

## Average strategy and solver state

Every experimental solver adds the existing exact
`own_reach(I) * sigma_i(I,a)` mass for every cataloged information set once per
iteration. Chance-sampling settings affect regret estimates only. Both player
traversals use a single frozen profile and all regret deltas are committed after
both traversals. Strategy choice continues to depend only on player-observable
information-set keys.

## Independent validation status

The Kuhn tests exhaustively sum private deals and all opponent action maps for
both traversers. Their reference is the existing test-only full-tree CFR
calculation; the production CFR update is not called. The Kuhn stratified and
enumerated weights are thereby checked against that reference. The Leduc
variant tests currently verify that private-deal weighting schemes agree under
an exhaustive all-deal/all-board traversal and deterministic opponent policies.
The pre-existing Leduc tests separately cover hand-derived regret fixtures,
non-unit own reach exclusion, shared information sets, exact own-reach
averaging, and sampled external regret. **The new Leduc scheme test is not yet
an independent regret oracle for each chance configuration**: its traversal
uses the experimental solver implementation. A disputed prior fixture also
failed to agree with that full-tree sum, so I did not hard-code it as an oracle.
This is a remaining validation blocker, not a confirmed production equation
defect.

The experimental average-strategy test confirms the exact own-reach delta is
unchanged by rank stratification and public enumeration. Baseline tests remain
intact. The new tests do not yet cover all four Leduc combinations with an
independent expected-regret oracle; the convergence results below are
exploratory only and cannot close that gap.

## Measured pilot

Artifacts are under
[`results/mccfr/phase3c2/pilot-20261030T224631Z`](../results/mccfr/phase3c2/pilot-20261030T224631Z/).
The run used Leduc, seeds 17 and 29, 100 iterations, and exact exploitability
evaluation at the final strategy. The experiment is intentionally too short
and has too few seeds for an algorithm ranking.

| Method | Mean exploitability at 100 iterations | Mean training seconds | Mean nodes on final iteration |
|---|---:|---:|---:|
| Vanilla CFR | 0.1665 | 7.44 | full-tree count not recorded |
| Baseline ES-MCCFR | 2.3167 | 0.746 | 46 |
| Public-card enumeration | 2.1027 | 0.785 | 157 |
| Rank strata + public enumeration | 1.7143 | 0.938 | 258 |

At 100 iterations, variants had lower wall-clock cost than vanilla CFR but
substantially higher exploitability. Public enumeration and the combined
configuration used more sampled nodes and showed lower mean exploitability
than baseline MCCFR in this two-seed snapshot; the result is not enough to
establish a repeatable gain. No monotonicity claim is made. Per-seed exact
values, training/evaluation times, and strategy snapshots are in `convergence.csv`
and the adjacent JSON files. `exploitability_vs_time.html` plots each measured
seed. The historical Phase 3C and Phase 3C.1 result directories were not
modified.

### Fixed-profile regret variance

The estimator pilot used the saved Phase 3C.1 Leduc MCCFR seed-7, 10,000-step
average strategy. For each of three information sets it recorded 500 sampled
traversals per scheme and seed (17, 29), including zero increments when the
information set was absent. Exact target regrets came from the independent
full-tree reference helper in `mccfr_investigation.py`. Raw increments are in
`variance_raw.csv`; per-action means, variances, standard errors, and normal
95% intervals are in `variance_summary.csv`.

Mean per-action variance ratios versus baseline, averaged across action/seed
cells (ratios are undefined for the third target because its baseline action
variance was zero):

| Target | Scheme | Per-traversal variance ratio | Node-cost ratio | Approx. equal-node-budget variance ratio |
|---|---|---:|---:|---:|
| P0 J0 preflop root | Private enumeration | 0.034 | 30.17 | 1.023 |
| P0 J0 preflop root | Public enumeration | 0.059 | 2.88 | 0.169 |
| P0 J0 preflop root | Rank strata + public enumeration | 0.042 | 5.76 | 0.243 |
| P1 Q0 facing preflop bet | Private enumeration | 0.027 | 29.96 | 0.815 |
| P1 Q0 facing preflop bet | Public enumeration | 0.085 | 3.09 | 0.264 |
| P1 Q0 facing preflop bet | Rank strata + public enumeration | 0.048 | 5.94 | 0.284 |

The first target's average traversal node counts were 621 (private enumeration),
59 (public enumeration), and 119 (rank strata plus public enumeration), versus
21 for baseline. The second target was similar (584, 60, 116 versus 20). The
equal-node-budget column multiplies the per-traversal variance ratio by the
node-cost ratio, assuming independent draws and a fixed total node budget.
Public enumeration and the combined variant show lower variance per node for
these two targets, while private enumeration is roughly neutral. At the rare
third target, several action/seed cells have zero baseline sample variance, so
those ratios are undefined and no beneficial claim is made. This uses node
visits as a compute proxy, not measured CPU time, and must not be generalized
to other information sets. `variance_by_target.html`
visualizes the action-averaged variances; `variance_comparison.csv` preserves
the computed ratios. The fixed-profile samples are diagnostic and not a
convergence experiment.

### Research decision and limitations

Keep baseline ES-MCCFR as the only supported solver. Public enumeration is the
most promising experimental candidate in these two information-set variance
measurements, but the short convergence pilot is inconclusive and the new
Leduc estimator configurations still need an independent exact regret oracle.
Private enumeration is especially costly. A follow-up should first resolve
the independent Leduc oracle discrepancy, then measure repeated batches with
runtime and variance at matched wall-clock budgets before any five-seed
convergence comparison.

## Research context

Lanctot et al. establish external sampling as an MCCFR estimator whose expected
updates match CFR. Gibson et al. analyze the role of bounded unbiased value
estimators and their variance in CFR updates. These results motivate testing
variance reduction but do not imply that any particular partition is useful in
PokerLab. A 2026 preprint on persistent correlated chance streams studies
cross-iteration chance allocation; that is distinct from the fresh,
within-iteration stratification measured here and is not silently conflated
with it.

- Lanctot et al. (2009), [Monte Carlo Sampling for Regret Minimization in
  Extensive Games](https://proceedings.neurips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html).
- Gibson et al. (2012), [Generalized Sampling and Variance in Counterfactual
  Regret Minimization](https://poker.cs.ualberta.ca/publications/AAAI12-generalmccfr.pdf).
- Li, Chen, and Huang (2026), [Correlated Chance Sampling for Monte Carlo
  Counterfactual Regret Minimization](https://arxiv.org/abs/2607.27035).
