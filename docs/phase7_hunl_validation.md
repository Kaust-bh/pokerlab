# Phase 7: HUNL Solver Validation, Generalization & Local Best Response

## 1. Overview and Scope

Phase 7 audits, corrects, and validates PokerLab's abstracted Heads-Up No-Limit Texas Hold'em (HUNL) solver, policy generalization pipeline, and evaluation methodology:

1. **Unbiased Own-Reach MCCFR Average-Strategy Estimator (`reach_weighted`)**: Replaces opponent-reach-biased strategy accumulation and high-variance $1 / p_c(h)$ importance weights in `src/pokerlab/research/nlhe/mccfr.py` with an unbiased sequence-form own-reach estimator under a full-support uniform opponent proposal distribution.
2. **Suit-Isomorphic Hand & Board Abstraction**: Canonicalizes suits jointly across `(hole_cards, board)` using rank-only street signatures (`_suit_signature`) and sorts the three flop cards while preserving turn and river street order in `src/pokerlab/research/nlhe/abstraction.py` and `src/pokerlab/research/nlhe/information_set.py`.
3. **Postflop Cross-Deal Regret Sharing & Hierarchical Generalization**: Pools postflop regret updates across equivalent bucket-and-betting-context states via `_abstract_regrets` during training, and resolves unseen evaluation states via a 3-tier lookup hierarchy (`exact` $\to$ `generalized` weighted by `strategy_sums` $\to$ card-aware `_tactical_fallback`) with per-decision lookup-rate telemetry in `src/pokerlab/research/nlhe/agent.py`.
4. **Local Best Response (LBR) Evaluation Framework**: Implements `BayesianRangeTracker`, `LocalBestResponseAgent`, `evaluate_lbr_kuhn`, `evaluate_lbr_leduc`, and `evaluate_lbr_hunl` in `src/pokerlab/evaluation/lbr.py`, validated independently against exact Kuhn and Leduc best-response solvers.
5. **Multi-Seed Abstraction Benchmark**: Evaluates all six abstraction variants (`baseline`, `equity_8`, `equity_20`, `action_wide`, `combined_wide`, and `isomorphic`) across four independent seeds (`11, 23, 47, 89`) and checkpoints `10, 25, 50` in `results/nlhe_abstraction/run-20261101T220341Z-6f72de15/`.

---

## 2. Mathematical Derivations

### 2.1 MCCFR Average-Strategy Estimator (`reach_weighted`)

In extensive-form games with perfect recall, the sequence-form average behavioral strategy $\bar{\sigma}_i^T(I, a)$ at information set $I$ for player $i$ weights each iteration's behavioral strategy $\sigma_i^t(I, a)$ by player $i$'s own reach probability $\pi_i^{\sigma^t}(I) = \pi_i^{\sigma^t}(h)$ (which is identical for all $h \in I$ under perfect recall):

$$
\bar{\sigma}_i^T(I, a) = \frac{\sum_{t=1}^T \pi_i^{\sigma^t}(I)\,\sigma_i^t(I, a)}{\sum_{t=1}^T \pi_i^{\sigma^t}(I)}
$$

In `ExternalSamplingMCCFR._average_traverse` (`average_strategy_mode="reach_weighted"`), each strategy-accumulation pass for traverser $i$:
- Samples chance transitions according to the true chance distribution $p_c(h)$.
- Enumerates all legal actions $a \in A(h)$ at traverser decision nodes $h \in \mathcal{H}_i$.
- Samples a single action $a \in A(h)$ uniformly with probability $q_{-i}(a \mid h) = 1 / |A(h)| > 0$ at opponent decision nodes $h \in \mathcal{H}_{-i}$.

Under this sampling distribution $Q$, the probability of reaching history $h \in I$ is:

$$
Q(h) = p_c(h)\, q_{-i}(h), \quad \text{where } q_{-i}(h) = \prod_{(h', a') \sqsubseteq_{-i} h} \frac{1}{|A(h')|} > 0
$$

Weighting each visited node $h \in I$ by `average_strategy_reach_weight(own_reach, opponent_proposal_reach)`:

$$
w_i(h) = \frac{\pi_i^{\sigma^t}(h)}{q_{-i}(h)}
$$

gives the expected single-iteration increment at information set $I$:

$$
\mathbb{E}_Q\!\left[\sum_{h \in I} \mathbb{I}[h \text{ reached}]\, w_i(h)\,\sigma_i^t(I, a)\right]
= \sum_{h \in I} p_c(h)\, q_{-i}(h)\, \frac{\pi_i^{\sigma^t}(h)}{q_{-i}(h)}\, \sigma_i^t(I, a)
= \pi_i^{\sigma^t}(I)\,\sigma_i^t(I, a) \sum_{h \in I} p_c(h)
$$

Because the constant chance prior $C(I) = \sum_{h \in I} p_c(h)$ depends only on $I$ and cancels upon normalization across $a \in A(I)$, the normalized cumulative strategy sums converge to the exact sequence-form average strategy $\bar{\sigma}_i^T(I, a)$:
- Unlike raw visitation weighting ($w_i(h) = \pi_i^{\sigma^t}(h)$ under opponent policy $\sigma_{-i}^t$), it does not multiply by $\pi_{-i}^{\sigma^t}(h)$ and never drops zero-opponent-reach branches.
- Unlike full inverse-reach importance sampling ($w_i(h) = \pi_i^{\sigma^t}(h) / (p_c(h)\,q_{-i}(h))$), it does not divide by $p_c(h) \approx 3.2 \times 10^{-15}$ across 9 dealt cards, avoiding $10^{15}$ weight variance.

### 2.2 Joint `(hole, board)` Suit Isomorphism and Flop Order Invariance

For any suit $s \in \{\text{clubs}, \text{diamonds}, \text{hearts}, \text{spades}\}$, `_suit_signature(s, hole, board)` computes the rank-only street tuple:

$$
\operatorname{sig}(s; \text{hole}, \text{board}) = \Big(
\operatorname{sort}_{\downarrow}\{r \mid (r, s) \in \text{hole}\},\;
\operatorname{sort}_{\downarrow}\{r \mid (r, s) \in \text{board}_{0:3}\},\;
(r \mid (r, s) \in \text{board}_{3:4}),\;
(r \mid (r, s) \in \text{board}_{4:5})
\Big)
$$

Sorting present suits in descending order of $\operatorname{sig}(s; \text{hole}, \text{board})$ and sorting the three flop cards by `(-rank, canonical_suit)` while keeping turn (`board[3]`) and river (`board[4]`) in street order guarantees:
1. **Strict invariance under all $4! = 24$ suit permutations**, including when `hole` is a pocket pair (where the two hole suits tie on hole rank and are broken deterministically by their flop/turn/river suit signatures).
2. **Strict invariance under all $3! = 6$ flop orderings**, while distinguishing turn and river arrival order.

### 2.3 Local Best Response (LBR) Candidate-Action Expected Values

Let $P_0 = \text{pot}$ denote the current total pot (which in PokerLab's engine already includes both players' current-street contributions $c_{\text{hero}}$ and $c_{\text{opp}}$), $S_{\text{hero}}$ and $S_{\text{opp}}$ denote remaining stacks, $\text{to\_call} = \max(0, c_{\text{opp}} - c_{\text{hero}})$, and $E \in [0, 1]$ denote hero's showdown equity under the posterior opponent hand range $B(h_{-i})$ and uniform board runouts:

1. **Fold (`ActionType.FOLD`)**:
   $$
   \text{EV}(\text{FOLD}) = 0.0
   $$
   (Sunk contributions are already inside $P_0$; folding forfeits the pot with zero additional chip loss.)

2. **Check / Call (`ActionType.CHECK`, `ActionType.CALL`)**:
   With incremental call cost $\Delta_{\text{call}} = \min(\text{to\_call}, S_{\text{hero}})$ and resulting pot $P_{\text{call}} = P_0 + \Delta_{\text{call}}$:
   $$
   \text{EV}(\text{CHECK/CALL}) = E \cdot P_{\text{call}} - \Delta_{\text{call}}
   $$

3. **Bet / Raise / All-In (`ActionType.BET`, `ActionType.RAISE`, `ActionType.ALL_IN`)**:
   For target street contribution $T = c_{\text{hero}} + \Delta_{\text{hero}}$ (with $\Delta_{\text{hero}} > \Delta_{\text{call}}$), the opponent's maximum callable incremental amount is $\Delta_{\text{opp}} = \min(\max(0, T - c_{\text{opp}}), S_{\text{opp}})$, and hero's effective incremental risk is $\Delta_{\text{hero,eff}} = c_{\text{opp}} + \Delta_{\text{opp}} - c_{\text{hero}}$ (any uncallable excess above $c_{\text{opp}} + S_{\text{opp}}$ is returned to hero):
   - If $\Delta_{\text{opp}} = 0$ (opponent is already all-in after hero matches):
     $$
     \text{EV}(a) = E \cdot (P_0 + \Delta_{\text{hero,eff}}) - \Delta_{\text{hero,eff}}
     $$
   - If $\Delta_{\text{opp}} > 0$, querying the opponent's action distribution $(p_{\text{fold}}, p_{\text{call}}, p_{\text{raise}})$ across sampled posterior hands $h_{-i} \sim B$ under check/call continuation yields:
     $$
     \text{EV}(a \mid h_{-i}) = p_{\text{fold}}\, P_0 + (p_{\text{call}} + p_{\text{raise}})\Big(E(h_{-i})\,(P_0 + \Delta_{\text{hero,eff}} + \Delta_{\text{opp}}) - \Delta_{\text{hero,eff}}\Big)
     $$

---

## 3. Toy-Game Exact Validation (Kuhn and Leduc Poker)

`evaluate_lbr_kuhn` and `evaluate_lbr_leduc` in `src/pokerlab/evaluation/lbr.py` implement independent forward Bayesian posterior belief tracking over private deals without calling `kuhn.evaluation.best_response_value` or `leduc.evaluation.best_response_value`:
- **Kuhn Poker (`test_lbr_matches_exact_best_response_on_kuhn_independently`)**: Across CFR checkpoints (`1, 10, 50` iterations) and both player seats, `evaluate_lbr_kuhn` matches the exact best-response value within `1e-9`.
- **Leduc Poker (`test_lbr_matches_and_bounds_exact_best_response_on_leduc_independently`)**: Across CFR checkpoints (`1, 5, 15` iterations) and both player seats:
  - `evaluate_lbr_leduc(..., mode="local_br")` matches the exact best-response value within `1e-9`.
  - `evaluate_lbr_leduc(..., mode="check_call_rollout")` (1-step local lookahead on the active street followed by passive check/call continuation on later streets) satisfies $\text{EV}(\sigma_i) \le \text{EV}(\text{LBR}_{\text{check\_call}}) \le \text{EV}(\text{BR}_i)$ at every checkpoint.

---

## 4. Multi-Seed HUNL Abstraction Benchmark (`run-20261101T220341Z-6f72de15`)

### 4.1 Experimental Design

The Phase 7 benchmark (`results/nlhe_abstraction/run-20261101T220341Z-6f72de15/`) compares six abstract HUNL configurations across four independent seeds (`11, 23, 47, 89`), 50 MCCFR training iterations, checkpoints at iterations `10, 25, 50`, a 10 BB starting stack (0.5/1 BB blinds), and 100 evaluation hands per opponent/button-role batch:

| Variant | Postflop Hand Method | Postflop Buckets | Board Abstraction | Action Profile |
|---|---|---:|---|---|
| `baseline` | `sampled_strength` | 20 | `exact` | `legacy` |
| `equity_8` | `sampled_equity` | 8 | `exact` | `legacy` |
| `equity_20` | `sampled_equity` | 20 | `exact` | `legacy` |
| `action_wide` | `sampled_strength` | 20 | `exact` | `wide` |
| `combined_wide` | `sampled_equity` | 20 | `exact` | `wide` |
| `isomorphic` | `sampled_equity` | 20 | `isomorphic` | `legacy` |

### 4.2 Training Diagnostics at Iteration 50

Unlike Phase 5B (where `baseline`, `equity_8`, and `equity_20` produced identical trajectories because postflop keys never collided across deals), cross-deal postflop regret sharing via `postflop_abstraction_key` produces distinct information-set growth, cumulative regret $L_1$ norms, and average-strategy entropies across all six variants:

| Variant | Info Sets (`seed=11`) | Info Sets (`seed=23`) | Info Sets (`seed=47`) | Info Sets (`seed=89`) | Mean Regret $L_1$ | Mean Strategy Entropy (nats) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 3,816 | 3,977 | 3,647 | 3,769 | 14,422.39 | 0.9437 |
| `equity_8` | 3,863 | 3,981 | 3,647 | 3,946 | 15,599.91 | 0.8954 |
| `equity_20` | 3,849 | 4,298 | 3,499 | 3,753 | 15,582.14 | 0.9424 |
| `action_wide` | 14,020 | 13,608 | 13,039 | 14,849 | 59,794.41 | 1.0229 |
| `combined_wide` | 12,709 | 13,627 | 13,037 | 12,585 | 60,186.66 | 1.0014 |
| `isomorphic` | 3,912 | 3,910 | 3,799 | 3,896 | 15,554.65 | 0.8837 |

### 4.3 Match Evaluation Results and Confidence Intervals at Iteration 50

Each seed evaluates two 100-hand batches per opponent (one with the trained agent on the button, one with the opponent on the button). Averaging the two button-role batches within each seed and computing two-sided 95% Student-$t$ intervals across the 4 independent seeds ($df = 3$) at iteration 50 yields:

| Variant | vs. `Random` BB/100 (95% CI) | Paired $\Delta$ vs. `baseline` (`Random`) | vs. `Heuristic` BB/100 (95% CI) | vs. `Equity` BB/100 (95% CI) | vs. `PreviousMCCFR` BB/100 (95% CI) |
|---|---:|---:|---:|---:|---:|
| `baseline` | $-32.75$ $[-47.57, -17.93]$ | $0.00$ | $-23.06$ $[-66.15, +20.02]$ | $-16.12$ $[-87.67, +55.42]$ | $-0.12$ $[-49.86, +49.61]$ |
| `equity_8` | $+26.38$ $[-3.45, +56.20]$ | **$+59.12$ $[+40.82, +77.43]$** | $-43.00$ $[-105.74, +19.74]$ | $-27.88$ $[-81.25, +25.50]$ | $-34.50$ $[-68.51, -0.49]$ |
| `equity_20` | $+18.50$ $[-26.60, +63.60]$ | **$+51.25$ $[+5.15, +97.35]$** | $-47.12$ $[-93.88, -0.37]$ | $-6.00$ $[-86.61, +74.61]$ | $-30.56$ $[-90.48, +29.35]$ |
| `action_wide` | $+14.81$ $[-56.73, +86.35]$ | $+47.56$ $[-31.91, +127.03]$ | $-16.06$ $[-50.89, +18.76]$ | $-7.56$ $[-24.38, +9.26]$ | $-0.88$ $[-92.48, +90.73]$ |
| `combined_wide` | $-8.88$ $[-72.20, +54.45]$ | $+23.88$ $[-45.11, +92.86]$ | $-27.62$ $[-65.21, +9.96]$ | $+4.88$ $[-39.66, +49.41]$ | $+17.75$ $[-52.18, +87.68]$ |
| `isomorphic` | $+10.81$ $[-75.81, +97.44]$ | $+43.56$ $[-31.24, +118.37]$ | $-48.12$ $[-92.40, -3.85]$ | $-31.31$ $[-101.53, +38.90]$ | $-16.69$ $[-97.73, +64.35]$ |

Position-specific rows (`trained_first_button = "trained"` vs. `"opponent"`) and checkpoints `10` and `25` are preserved in full in `summary.csv` and `paired_comparisons.csv`.

### 4.4 Policy Lookup Coverage Telemetry

`AbstractHUNLAgent` records whether each decision is served by an exact information-set hit (`exact_hits`), the `strategy_sums`-weighted hierarchical generalizer (`generalized_hits`), or the card-aware tactical fallback (`fallback_hits`). Aggregated across all evaluation decisions at iteration 50:

| Variant | Total Decisions | Exact Hits (`exact_lookup_rate`) | Generalized Hits (`generalized_lookup_rate`) | Fallback Hits (`fallback_lookup_rate`) | Learned Policy Coverage (`exact + gen`) |
|---|---:|---:|---:|---:|---:|
| `baseline` | 6,332 | 1,983 (31.3%) | 2,472 (39.0%) | 1,877 (29.6%) | **70.4%** |
| `equity_8` | 6,189 | 2,078 (33.6%) | 2,823 (45.6%) | 1,288 (20.8%) | **79.2%** |
| `equity_20` | 6,277 | 2,033 (32.4%) | 2,482 (39.5%) | 1,762 (28.1%) | **71.9%** |
| `action_wide` | 6,635 | 1,985 (29.9%) | 2,267 (34.2%) | 2,383 (35.9%) | **64.1%** |
| `combined_wide` | 6,751 | 1,986 (29.4%) | 2,424 (35.9%) | 2,341 (34.7%) | **65.3%** |
| `isomorphic` | 6,201 | 2,011 (32.4%) | 2,461 (39.7%) | 1,729 (27.9%) | **72.1%** |

Across checkpoints `10 -> 25 -> 50`, fallback reliance drops steadily (e.g., for `equity_8`, from ~`58–63%` at iteration 10 to ~`34–37%` at iteration 25 and **20.8%** at iteration 50).

---

## 5. Limitations and Scientific Interpretation

1. **HUNL LBR Is a Tactical Probe, Not an Exact Exploitability Bound**:
   While `evaluate_lbr_kuhn` and `evaluate_lbr_leduc` enumerate exact posterior ranges over small game trees, `evaluate_lbr_hunl` uses finite Monte Carlo hand and runout samples (`rollout_samples`), 1-step aggressive lookahead, and queries an abstracted policy that may use generalized or tactical fallback on off-tree probes. Consequently, `evaluate_lbr_hunl` measures finite-sample vulnerability to a 1-step range-aware responder and must **not** be interpreted as a formal global exploitability lower bound or a certificate of Nash equilibrium.
2. **Finite-Budget Abstraction Trade-Offs**:
   At 50 MCCFR iterations on a 10 BB stack with 4 seeds, confidence intervals remain wide because of high single-hand payoff variance and finite tree coverage (20.8%–35.9% fallback rate at iteration 50). While `equity_8` and `equity_20` show positive paired improvements over `baseline` against `Random` whose 95% Student-$t$ intervals exclude zero at iteration 50, results against `Heuristic` and `Equity` remain noisy and negative on average at this training budget.
3. **Cross-Deal Postflop Regret Sharing Is an Abstraction Approximation**:
   Sharing regrets across deals via `postflop_abstraction_key` accelerates postflop learning at small iteration budgets by pooling histories with identical player, street, primary hand bucket, board feature, and street action pattern, at the cost of coalescing distinct preflop hand classes (`starting_class`) when an exact postflop information set has not yet been visited.
4. **Provenance Note on `run-20261101T220341Z-6f72de15`**:
   All 225 artifact SHA-256 digests and 79 source SHA-256 digests recorded in `results/nlhe_abstraction/run-20261101T220341Z-6f72de15/manifest.json` were verified at completion (`git_head`: `317ac96adacfbbbaeae5108b2ed05fcf185da14e`). Following that run, a 1-line provenance fix was applied to `src/pokerlab/research/nlhe/benchmark.py` so `summary.json["source_revision"]` also queries `git -C <repo_root> rev-parse HEAD` when invoked from outside the repository root (verified by smoke test and `test_benchmark_estimate_and_smoke_run_write_complete_artifacts`). The historical directory `run-20261101T220341Z-6f72de15` is preserved unmodified.

---

## 6. Reproducibility Instructions

Run the unit and validation test suite:

```sh
pytest -c pyproject.toml tests
ruff check src tests
ruff format --check src tests
git diff --check
```

Re-run the 6-variant, 4-seed, 50-iteration benchmark:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark \
  --iterations 50 --checkpoints 10 25 50 \
  --seeds 11 23 47 89 --evaluation-hands 100 \
  --output-root results/nlhe_abstraction
```

Run a fast smoke benchmark to verify artifact and provenance generation:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.benchmark \
  --iterations 2 --checkpoints 2 --seeds 11 \
  --variants baseline isomorphic --evaluation-hands 4 \
  --output-root results/nlhe_abstraction
```
