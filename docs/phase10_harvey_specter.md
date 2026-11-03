# Phase 10: HarveySpecter — Adaptive Opponent Intelligence & Exploitative Poker Strategy

*"Play the man, not the odds."*

## 1. Overview & Architecture

`pokerlab.opponent_model` implements **HarveySpecter**, a research-grade adaptive opponent modeling, Bayesian hand-range inference, probabilistic archetype classification, and exploitative strategy blending system for Heads-Up No-Limit Texas Hold'em (HUNL).

Without rewriting the poker engine or existing CFR/MCCFR solvers, HarveySpecter wraps `AbstractHUNLAgent` and adapts its baseline strategy online as statistical evidence about the opponent accumulates:

$$\pi_{\text{final}}(a \mid I) = (1 - \lambda)\,\pi_{\text{MCCFR}}(a \mid I) + \lambda\,\pi_{\text{exploit}}(a \mid I, \hat{\theta}_{\text{opp}}, P(H \mid A_{1:t}))$$

### Module Structure (`src/pokerlab/opponent_model/`)

| Module | Primary Classes / Functions | Responsibility |
|---|---|---|
| `models.py` | `ObservedActionEvent`, `ObservedHandRecord`, `SmoothedMetric`, `OpponentStatsSnapshot`, `ArchetypeDistribution`, `RangeComboEntry`, `BayesianRangeSnapshot`, `ExploitativeActionEvaluation`, `HarveyDecisionReport` | Immutable data structures for action/hand telemetry, smoothed statistics, Bayesian posteriors, archetype distributions, and decision reports |
| `collector.py` | `OpponentDataCollector` | Strictly information-safe observation & hand-end collector; enforces card privacy and deduplicates action history via `processed_history_len` |
| `statistics.py` | `BehaviouralProfiler` | Computes 20+ mathematically defined poker statistics with Beta/Laplace shrinkage, Wilson 95% confidence intervals, recency decay, and drift stability |
| `range_model.py` | `BayesianRangeModel`, `ShowdownCalibration`, `postflop_combo_strength`, `ALL_COMBOS` | Exact 1,326-combination ($169$-class) Bayesian range tracker with card removal, sequential action likelihood updates, and showdown calibration |
| `archetypes.py` | `ArchetypeClassifier`, `PROTOTYPES` | Soft Gaussian-prototype posterior over 7 poker archetypes with cold-start prior and stability-modulated confidence |
| `exploit.py` | `ExploitativeStrategyEngine` | Computes $\pi_{\text{exploit}}(a)$, range-weighted hero equity, opponent fold-equity estimates, candidate EVs, and grounded natural-language explanations |
| `adaptive_agent.py` | `HarveySpecterAgent`, `HarveySpecterConfig`, `CallingStationAgent`, `TightPassiveAgent`, `LooseAggressiveAgent`, `ManiacAgent`, `NitAgent` | Adaptive MCCFR + exploit policy wrapper (`adaptive`, `fixed_lambda`, `baseline_only`, `pure_exploit`) and controlled archetype benchmark opponents |
| `profile_store.py` | `OpponentProfile`, `OpponentProfileStore`, `ProfileStorageError` | Atomic JSON persistence, corrupt/missing file recovery, reset, export/import, and isolated multi-opponent profile management |
| `benchmark.py` | `run_harvey_specter_benchmark`, `HarveyBenchmarkConfig` | Paired multi-seed benchmark runner with BB/100, paired gain CIs, archetype accuracy, and Bayesian range Brier/likelihood calibration metrics |

---

## 2. Information Boundary & Data Collection (`collector.py`)

`OpponentDataCollector` enforces strict hidden-information separation:

1. **During live play (`ingest_observation(obs)`)**:
   - Reads only public fields from `Observation`: `obs.history[processed_history_len:]`, `obs.community_cards`, `obs.hole_cards` (hero's own cards, used solely as card-removal blockers in `BayesianRangeModel`), `obs.pot`, `obs.to_call`, `obs.stack`, `obs.opponent_stack`, and `obs.is_button`.
   - **Never** accesses `GameState` or unrevealed opponent hole cards.
   - Deduplicates action history within each hand using `processed_history_len` so repeated calls on the same street or observation never double-count an action.
2. **At hand completion (`finalize_hand(...)`)**:
   - Records `went_to_showdown` and `revealed_hole_cards` **if and only if** `went_to_showdown=True` and a legitimate 2-card tuple is provided.
   - When a hand ends by fold (`went_to_showdown=False`), `revealed_hole_cards` is forced to `None` even if caller code passes cards, preventing information leakage from mucked hands.

---

## 3. Behavioural Statistics & Beta Smoothing (`statistics.py`)

### 3.1 Exact Metric Definitions

Let $h \in \{1, \dots, N\}$ index observed hands with recency weight $w_h = \gamma^{N - h}$ where $\gamma \in (0, 1]$ is `recency_decay` ($\gamma = 1.0$ for unweighted cumulative statistics):

- **VPIP** (`vpip`): Voluntary chips put into the preflop pot (`CALL` facing blind/raise, `BET`, `RAISE`, or `ALL_IN` preflop; checking the big blind option is excluded). Denominator: hands with at least one preflop action or decision.
- **PFR** (`pfr`): Hands where the opponent made at least one preflop `BET`, `RAISE`, or raising `ALL_IN`. Denominator: preflop opportunity hands.
- **VPIP − PFR Gap** (`vpip_minus_pfr`): $\max(0, \text{VPIP}_{\text{smoothed}} - \text{PFR}_{\text{smoothed}})$, measuring preflop passivity/calling tendency.
- **3-Bet Frequency** (`three_bet_freq`): Preflop re-raises when facing an open raise (`facing_raise_count == 1`).
- **Fold to 3-Bet** (`fold_to_three_bet`): Preflop folds when facing a 3-bet (`facing_raise_count >= 2`).
- **Continuation Bet Frequency** (`cbet_freq`): Flop bet/all-in when the opponent was the last preflop aggressor and had the opportunity to open action on the flop (`to_call_before == 0`).
- **Fold to C-Bet** (`fold_to_cbet`): Flop folds when facing a continuation bet from the preflop aggressor.
- **Aggression Frequency** (`aggression_frequency`, $\text{AFq}$):
  $$\text{AFq} = \frac{N_{\text{bet}} + N_{\text{raise}} + N_{\text{aggressive all-in}}}{N_{\text{bet}} + N_{\text{raise}} + N_{\text{call}} + N_{\text{fold}}}$$
- **Aggression Factor** (`aggression_factor`, $\text{AF}$):
  $$\text{AF} = \frac{\alpha_{\text{prior}} + W_{\text{bet+raise}}}{\beta_{\text{prior}} + W_{\text{call}}}$$
  with smoothing constants $(\alpha_{\text{prior}}, \beta_{\text{prior}}) = (1.5, 1.5)$ so $W_{\text{call}} = 0$ remains finite and well-behaved.
- **Fold to Bet** (`fold_to_bet`): Folds across all streets when facing a positive amount to call (`facing_bet == True`).
- **Fold to River Bet** (`fold_to_river_bet`): River folds when facing a bet or raise on the river.
- **Showdown Frequency** (`showdown_frequency`) & **WTSD When Saw Flop** (`wtsd_when_saw_flop`): Fraction of all hands (and flop-seen hands) reaching showdown.
- **Won at Showdown** (`won_at_showdown`, $W\$SD$): Fraction of showdowns where the opponent won a positive share of the pot (`result_chips >= 0`).
- **Average Bet / Raise Size Pot Ratio** (`avg_bet_size_pot_ratio`, `avg_raise_size_pot_ratio`): Shrinkage-smoothed mean of $\Delta \text{chips} / \text{pot}_{\text{before}}$.
- **Positional Splits**: `vpip_button`, `vpip_big_blind`, `pfr_button`, `pfr_big_blind`.

### 3.2 Beta-Binomial Smoothing & Confidence

Each binomial rate metric with prior mean $\mu_0 \in (0, 1)$, prior pseudo-count $M_0 > 0$ ($\alpha_0 = \mu_0 M_0$, $\beta_0 = (1 - \mu_0) M_0$), weighted successes $S_w$, and weighted trials $T_w$ computes:

$$\hat{p}_{\text{smoothed}} = \frac{\alpha_0 + S_w}{\alpha_0 + \beta_0 + T_w}, \qquad \text{Confidence} = \frac{T_w}{T_w + M_0} \in [0, 1)$$

95% intervals use the Wilson score interval over effective sample size $T_w$ blended with the Beta posterior standard deviation when $T_w = 0$.

### 3.3 Strategy Drift & Stability Score

When an opponent shifts behaviour mid-session (e.g., Nit $\to$ Maniac), `BehaviouralProfiler._compute_stability` compares the recent window (last 10 hands) against earlier hands across VPIP, PFR, and postflop fold/aggression rates:

$$\text{Stability} = \exp(-1.8 \cdot \max(0, \Delta_{\text{drift}} - 0.12)) \in [0.15, 1.0]$$

A drop in stability lowers archetype/blending overconfidence while recency weighting ($\gamma < 1$) shifts the profile toward the new regime.

---

## 4. Bayesian Hand Range Estimation (`range_model.py`)

### 4.1 State Space & Card Removal

`BayesianRangeModel` maintains an explicit probability distribution $P(H_i)$ over all $\binom{52}{2} = 1,326$ unordered two-card combinations $H_i = \{c_1, c_2\}$ (mapped to 169 canonical preflop classes `AA`, `AKs`, `AKo`, $\dots$, `72o`).

Whenever hero hole cards or community board cards $B \subset \mathcal{D}$ are observed:
$$P(H_i \mid B) = 0 \quad \text{if } H_i \cap B \neq \emptyset$$
and all remaining unblocked combinations are renormalized so $\sum_{i=1}^{1326} P(H_i) = 1$.

### 4.2 Sequential Bayesian Action Updates

For each newly observed opponent action $a_t$ in context $C_t = (\text{street}, \text{position}, \text{pot\_ratio}, \text{facing\_bet}, \text{facing\_raise\_count}, \text{board})$:

$$P(H_i \mid a_{1:t}, C_{1:t}) = \frac{P(a_t \mid H_i, C_t, \hat{\theta}_{\text{opp}})\,P(H_i \mid a_{1:t-1}, C_{1:t-1})}{\sum_{j: H_j \cap B = \emptyset} P(a_t \mid H_j, C_t, \hat{\theta}_{\text{opp}})\,P(H_j \mid a_{1:t-1}, C_{1:t-1})}$$

where hand strength $s(H_i, B) \in [0.05, 0.999]$ is:
- **Preflop**: Canonical Chen/equity strength `COMBO_PREFLOP_STRENGTH[H_i]`.
- **Postflop**: `postflop_combo_strength(H_i, B)`, which evaluates the exact 5-to-7-card `HandCategory` and primary tiebreaker rank on $B$ (distinguishing overpairs and top pair from bottom/board pairs) plus flush-draw and straight-draw bonuses on flop and turn.
- **Likelihood parameterization**:
  - **Preflop raise / 3-bet / 4-bet**: Sigmoidal threshold centered at $1 - \text{PFR}_{\text{pos}}$ (or $1 - 3\text{Bet}$ for re-raises), plus a calibrated light-3-bet/bluff tail.
  - **Preflop call**: Two-sided band between $1 - \text{VPIP}_{\text{pos}}$ (bottom of calling range) and $1 - 0.65\,\text{PFR}_{\text{pos}}$ (hands that trap/flat rather than 3-bet).
  - **Postflop bet / raise / all-in**: Value component $\sigma((s - \tau_{\text{val}})/0.10)$ where $\tau_{\text{val}}$ increases with $\text{pot\_ratio}$ and street depth, plus a bluff/semi-bluff component weighted by the opponent's empirical $\text{AFq}$ and showdown bluff rate $\hat{\beta}_{\text{SD}}$.
  - **Postflop call**: Condensed medium-to-strong value/draw band adjusted by $(1 - \text{FoldToBet})$.
  - **Postflop check**: Downweights nutted hands that usually bet for value unless the opponent's slowplay/trap frequency $\hat{\tau}_{\text{SD}}$ is high.
  - **Fold**: Recorded as a terminal action event without pretending the folded hole cards were revealed.

### 4.3 Showdown Calibration (`ShowdownCalibration`)

When a hand reaches showdown and $H^* = (c_1^*, c_2^*)$ is legitimately revealed:
1. Before modifying calibration counters, `BayesianRangeModel` records out-of-sample predictive accuracy for that hand:
   - **Posterior vs Uniform-Unblocked Likelihood Ratio**: $P_{\text{post}}(H^*) / P_{\text{unif}}(H^*)$.
   - **Multi-class Brier Score**: $\sum_{i} (P(H_i) - \mathbb{I}[H_i = H^*])^2$ for both posterior and prior.
2. `ShowdownCalibration` updates smoothed empirical counters (`bluff_frequency_at_showdown`, `trap_frequency_at_showdown`, `mean_aggressive_showdown_strength`) keyed by `hand_number` so the same showdown can never be calibrated twice.

---

## 5. Archetype Classification (`archetypes.py`)

`ArchetypeClassifier` evaluates a soft Gaussian-prototype likelihood across 6 normalized behavioural features $(\text{VPIP}, \text{PFR}, \text{VPIP}-\text{PFR}, \text{AFq}, \text{FoldToBet}, \text{WTSD})$ for 7 archetypes:

- `nit`: Ultra-tight preflop ($\text{VPIP} \approx 16\%, \text{PFR} \approx 11\%$), high fold-to-bet ($\approx 68\%$).
- `tight_passive`: Low PFR ($\approx 8\%$), passive postflop ($\text{AFq} \approx 18\%$), folds to large pressure ($\approx 52\%$).
- `calling_station`: High VPIP ($\approx 74\%$), low PFR ($\approx 10\%$), wide gap ($\approx 62\%$), very low fold-to-bet ($\approx 16\%$), high WTSD ($\approx 72\%$).
- `tight_aggressive` (TAG): Selective VPIP ($\approx 36\%$), high PFR/VPIP ratio ($\approx 29\%$), solid aggression ($\text{AFq} \approx 48\%$).
- `loose_aggressive` (LAG): Wide VPIP ($\approx 62\%$), high PFR ($\approx 46\%$), high aggression ($\text{AFq} \approx 58\%$).
- `maniac`: Extreme VPIP ($\approx 86\%$), extreme PFR ($\approx 74\%$), hyper-aggressive ($\text{AFq} \approx 78\%$), low fold-to-bet ($\approx 14\%$).
- `balanced_unknown`: Equilibrium/prior baseline dominant on cold start ($N < 5$ hands).

Features with zero postflop trials receive zero feature-distance weight so preflop-only hands (e.g., against a Nit who folds preflop) are not penalized by uninformative postflop priors.

---

## 6. Exploitative Strategy Engine & Adaptive MCCFR Blending (`exploit.py`, `adaptive_agent.py`)

### 6.1 Range-Aware Equity & Candidate Action EV

Given hero hole cards $H_{\text{hero}}$, board $B$, opponent posterior range $P(H_{\text{opp}} \mid A_{1:t})$, and smoothed opponent tendencies:

1. **Range-Weighted Hero Equity** $\hat{E}_{\text{range}}$:
   Samples opponent hole cards from $P(H_{\text{opp}} \mid A_{1:t})$ (excluding cards in $H_{\text{hero}} \cup B$) and board runouts to estimate hero's showdown equity against the opponent's actual action-conditioned range.
2. **Candidate Action EVs**:
   - $\text{EV}(\text{FOLD}) = 0$ (measuring incremental return relative to current pot decision).
   - $\text{EV}(\text{CHECK}) = \kappa_{\text{realize}} \cdot \hat{E}_{\text{range}} \cdot \text{Pot}$, where equity realization $\kappa_{\text{realize}}$ depends on position and opponent aggression.
   - $\text{EV}(\text{CALL}, C) = \hat{E}_{\text{range, call}} \cdot (\text{Pot} + C) - C$, with $\hat{E}_{\text{range, call}}$ adjusted for the opponent's bet-sizing polarization and bluff frequency.
   - $\text{EV}(\text{BET/RAISE}, \Delta) = \hat{p}_{\text{fold}}(\Delta) \cdot \text{Pot} + (1 - \hat{p}_{\text{fold}}(\Delta)) \cdot \left[\hat{E}_{\text{when called}}(\Delta) \cdot (\text{Pot} + C + 2\Delta_{\text{add}}) - (C + \Delta_{\text{add}})\right]$.
3. **Explicit Archetype & Tendency Adjustments**:
   - **vs Calling Station / Low Fold-to-Bet**: Suppresses pure bluffs ($\hat{E}_{\text{range}} < 0.38$) and widens/sizes up value bets ($\hat{E}_{\text{range}} \ge 0.60$).
   - **vs Nit / High Fold-to-Bet**: Increases profitable bluff and semi-bluff steals when $\hat{p}_{\text{fold}}(\Delta) > \frac{\Delta}{\text{Pot} + \Delta}$, and folds marginal bluff-catchers against rare nit bets/raises.
   - **vs Maniac / Loose-Aggressive**: Widens bluff-catching (`CALL`) with medium-strength hands against wide aggression and prefers trapping/inducing over folding.

### 6.2 Adaptive Mixing Parameter $\lambda$

`HarveySpecterAgent` combines $\pi_{\text{MCCFR}}$ and $\pi_{\text{exploit}}$ via:

$$\pi_{\text{final}}(a) = (1 - \lambda)\,\pi_{\text{MCCFR}}(a) + \lambda\,\pi_{\text{exploit}}(a)$$

In `adaptive` mode:
$$\lambda = \text{clip}\!\left(\lambda_{\max} \cdot w_{\text{sample}}(N) \cdot (0.45 + 0.55\,c_{\text{arch}}) \cdot (0.45 + 0.55\,\text{extremity}) \cdot (0.50 + 0.50\,\text{stability}),\; 0,\; \lambda_{\max}\right)$$
where $w_{\text{sample}}(N) = 0$ for $N < N_{\text{cold}}$ (default 3 hands) and $w_{\text{sample}}(N) = \frac{N}{N + N_{\text{half}}}$ thereafter.
- **Cold start ($N = 0$)**: $\lambda = 0 \implies \pi_{\text{final}} = \pi_{\text{MCCFR}}$ exactly.
- **`baseline_only` mode**: $\lambda = 0$ always (used to verify zero behavioral drift while tracking opponent statistics).
- **`fixed_lambda` mode**: $\lambda = \lambda_{\text{fixed}}$ once $N \ge 1$.
- **`pure_exploit` mode**: $\lambda = 1.0$.

---

## 7. Multi-Seed Paired Benchmark Results

Command executed:

```sh
PYTHONPATH=src python -m pokerlab.opponent_model.benchmark \
  --hands-per-seed 80 --seeds 101 202 303 404 \
  --output-dir results/harvey_specter/run-phase10
```

Artifacts are saved in `results/harvey_specter/run-phase10/benchmark_results.json` and `results/harvey_specter/run-phase10/summary.csv` (`4` seeds $\times$ `80` hands = `320` paired hands per matchup across `5` variants $\times$ `8` opponents = `12,800` hands total; `100 BB` stacks, `1/2` blinds).

### 7.1 BB/100 Performance & Paired Gain vs Baseline MCCFR (`320` Hands / `4` Seeds per Row)

| Opponent | Baseline MCCFR (BB/100) | Harvey Disabled $\lambda=0$ (BB/100) | Harvey Fixed $\lambda=0.5$ (BB/100) | Harvey Adaptive (BB/100) | Adaptive Paired Gain vs Baseline (BB/100, 95% CI) | Pure Exploit $\lambda=1$ (BB/100) |
|---|---:|---:|---:|---:|---:|---:|
| `CallingStation` | `-29.53` | `-29.53` | `+1138.28` | **`+898.75`** | **`+928.28`** `[+251.42, +1605.14]` | `+1242.34` |
| `RandomAgent` | `+12.97` | `+12.97` | `-118.91` | **`+587.66`** | **`+574.69`** `[-6.42, +1155.80]` | `-5.16` |
| `LooseAggressive` | `-500.62` | `-500.62` | `-394.84` | **`-3.28`** | **`+497.34`** `[-11.17, +1005.86]` | `-73.12` |
| `EquityAgent` | `-469.53` | `-469.53` | `+115.47` | **`-115.62`** | **`+353.91`** `[-139.92, +847.74]` | `+298.75` |
| `TightPassive` | `-107.03` | `-107.03` | `+27.97` | **`+114.22`** | **`+221.25`** `[+51.31, +391.19]` | `+155.00` |
| `Maniac` | `-246.72` | `-246.72` | `-153.59` | **`-104.53`** | **`+142.19`** `[-452.31, +736.69]` | `-9.22` |
| `HeuristicAgent` | `-97.81` | `-97.81` | `-204.06` | **`+0.62`** | **`+98.44`** `[-148.44, +345.32]` | `-188.12` |
| `MCCFRBaseline` | `+190.31` | `+190.31` | `+251.88` | **`+211.88`** | **`+21.56`** `[-566.55, +609.68]` | `+382.03` |

### 7.2 Archetype Classification & Bayesian Range Calibration (`HarveySpecter_Adaptive`)

| Opponent | Expected Archetype | Predicted Archetype | Target Archetype Prob | Archetype Accuracy | Showdown Range Likelihood Ratio ($P_{\text{post}} / P_{\text{unif}}$) | Brier Improvement ($\text{Brier}_{\text{prior}} - \text{Brier}_{\text{post}}$) | Mean $\lambda$ / Final $\lambda$ |
|---|---|---|---:|---:|---:|---:|---:|
| `CallingStation` | `calling_station` | `calling_station` | `0.8061` | `100.0%` | `1.2259x` | `-0.0020` | `0.549` / `0.638` |
| `TightPassive` | `tight_passive` | `tight_passive` | `0.5683` | `75.0%` | `1.4134x` | `+0.0036` | `0.520` / `0.669` |
| `Maniac` | `maniac` | `maniac` | `0.8026` | `100.0%` | `0.9791x` | `-0.0067` | `0.555` / `0.627` |
| `RandomAgent` | `loose_aggressive` | `loose_aggressive` | `0.5987` | `75.0%` | `1.0252x` | `-0.0045` | `0.457` / `0.578` |
| `LooseAggressive` | `loose_aggressive` | `calling_station` / `lag` | `0.2731` | `50.0%` (`100%` in Disabled) | `1.2763x` | `-0.0077` | `0.491` / `0.625` |
| `HeuristicAgent` | `tight_passive` | `calling_station` | `0.0143` | `0.0%`* | `1.4691x` | `+0.0033` | `0.483` / `0.593` |
| `EquityAgent` | `tight_aggressive` | `calling_station` | `0.0193` | `0.0%`* | `1.8594x` | `+0.0037` | `0.502` / `0.588` |
| `MCCFRBaseline` | `balanced_unknown` | `calling_station` | `0.1322` | `0.0%`* | `1.0665x` | `-0.0046` | `0.458` / `0.580` |

*\*Observation on `HeuristicAgent` and `EquityAgent` classification*: In 100 BB Heads-Up play with 1/2 blinds, `HeuristicAgent` and `EquityAgent` limp/call preflop with a very wide fraction of hands when facing a 1 BB completion (`to_call = 1`) and call postflop bets whenever static equity exceeds pot odds, producing empirical $(\text{VPIP} > 68\%, \text{PFR} < 18\%, \text{FoldToBet} < 25\%)$ profiles that genuinely match `calling_station` in behavioural space. Against both agents, `BayesianRangeModel` achieves strong showdown likelihood ratios (`1.47x` vs `HeuristicAgent`, `1.86x` vs `EquityAgent`) and positive Brier score improvements (`+0.0033` and `+0.0037`), while `HarveySpecter_Adaptive` improves performance by `+98.44 BB/100` and `+353.91 BB/100` respectively.

---

## 8. Arena & Policy Inspector Usage

Launch PokerLab Arena:

```sh
PYTHONPATH=src python -m pokerlab.arena --host 127.0.0.1 --port 8501
```

- **Quick Presets**:
  - **Human vs HarveySpecter**: Play Heads-Up NLHE directly against `HarveySpecter (Adaptive MCCFR)` while watching it profile your VPIP, PFR, 3-Bet, C-Bet, AFq, Fold-to-Bet, archetype distribution, and Bayesian hand range in real time.
  - **Harvey vs Calling Station**: Watch HarveySpecter identify a Calling Station, ramp up $\lambda$, suppress bluffs, and value-bet relentlessly.
- **Policy Inspector & Opponent Intelligence Drawer**:
  - Displays side-by-side $\pi_{\text{MCCFR}}(a)$, $\pi_{\text{exploit}}(a)$, and $\pi_{\text{final}}(a)$ bars, $\lambda$ mixing weight, range-weighted hero equity, opponent fold-equity estimates, candidate action EVs, and a grounded natural-language strategic explanation citing exact action counts.
  - Includes the **13×13 Canonical Hand Range Heatmap** (`AA`..`22`, suited upper triangle, offsuit lower triangle) and top unblocked combinations, plus **Reset Profile**, **Export JSON**, and **Import JSON** controls backed by `/api/opponent-profile/*`.

---

## 9. Known Limitations & Future Research

1. **1-Step Action EV Approximation**: `ExploitativeStrategyEngine` evaluates candidate actions using range-weighted Monte Carlo rollouts and 1-step fold/call response models rather than full multi-street subgame solving. Against a sophisticated counter-adapting adversary, high $\lambda$ values can themselves become exploitable, which is why `HarveySpecterConfig.max_lambda` caps $\lambda \le 0.80$ by default.
2. **Factored Action Likelihoods**: `BayesianRangeModel` uses smooth parametric likelihood functions calibrated from showdown summaries rather than a full sequence-form equilibrium policy table for all possible opponent types.
3. **High HUNL Variance**: No-limit all-in runouts have ~100 BB single-hand swings, so 320-hand paired confidence intervals remain wide (`±170` to `±680 BB/100`) even when paired seed diffs are consistently positive across all 8 opponent types.
