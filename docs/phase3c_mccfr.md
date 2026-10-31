# Phase 3C: External Sampling MCCFR

PokerLab implements two-player, zero-sum External Sampling Monte Carlo
Counterfactual Regret Minimization (ES-MCCFR) for its Kuhn and fixed-limit
Leduc research games. The sampled solvers live beside the existing vanilla
solvers; game rules, payoff conventions, vanilla CFR, and exact Nash evaluators
are reused without modification.

## Regret estimator

For a traverser i, a sampled external block fixes chance outcomes and the
opponent's behavioral choices. At every traverser information set, the
algorithm explores every legal action and computes:

    sampled_regret(I, a) = sampled_value(I, a)
                           - sum_b sigma_i(b | I) sampled_value(I, b)

The traverser's policy is used at later traverser decisions inside each
continuation. Chance and opponent actions are sampled from their actual current
distributions. Therefore, the probability of sampling an external block is
exactly its chance reach times opponent reach. Summing the sampled regret over
those blocks in expectation gives the vanilla counterfactual regret update:

    E[sampled_regret(I, a)]
      = sum_h pi_-i(h) [v_i(h, a) - v_i(h)]

The sampled regret update receives no extra probability multiplier. In
particular, it does not include traverser reach. At each iteration both players
traverse from independently sampled private deals using the same frozen
strategy profile; regret deltas are applied only after both traversals finish.
At opponent decisions, the sampled action is keyed by the opponent's
information set, so a policy never conditions on a hidden card.

## Average strategy

Vanilla CFR's behavioral average adds
pi_i(I) * sigma_i(a | I) to each information set on every iteration. A sampled
external trajectory may omit an information set because of chance or opponent
actions, which would bias this average if it were updated only when visited.
PokerLab instead precomputes the finite information-set catalog and the acting
player's own action sequence for each set. Each iteration adds the exact
own-reach-weighted mass for every set. This retains the vanilla average
definition, including sets absent from the two sampled regret traversals, and
avoids importance weights in the regret estimator.

This exact average update costs work proportional to the information-set
catalog per iteration. It reduces neither the variance nor the cost of the
average update, and can limit the end-to-end speed benefit in small games.
Training-time comparisons include this work and exclude exact evaluation.

## Validation

Kuhn tests enumerate all chance deals and pure external action assignments,
then compare expected sampled regret increments for both players with an
independent full-tree calculation. Separate full-tree references validate
own-reach average mass.

Leduc tests use fixed profiles and seeded Monte Carlo expectation checks for a
selected information set for each player. Their action utilities, chance
probabilities, opponent sampling probabilities, own-reach exclusion, and
statistical tolerances are stated in the tests. Additional checks cover
shared hidden-state information sets, deterministic seeded trajectories,
different-seed trajectories, and exact evaluator integration.

All convergence measurements use exact expected values, best responses,
NashConv, and exploitability from the existing evaluators. Exploitability is
not sampled. It need not decrease at every checkpoint.

## Running a comparison

    PYTHONPATH=src python -m pokerlab.research.mccfr_experiments \
      --games kuhn leduc --seeds 7 19 --iterations 1000 \
      --evaluation-interval 250 --output-root results/mccfr

Each invocation creates a new timestamped directory and refuses to replace an
existing one. It contains convergence and comparison CSV files, final strategy
JSON for each solver/game/seed, configuration and Git provenance manifests,
and HTML plots for exploitability versus iterations, exploitability versus
training time, and average-strategy distance to the same-seed vanilla run.
The table-size figure is an approximate Python-object estimate, not process
RSS. It should be used as a qualitative observation.

## Limitations

- These implementations target the included finite Kuhn and Leduc games; they
  are not an NLHE solver.
- External sampling has higher variance per update than full-tree CFR.
- The exact average-strategy pass is intentional for mathematical comparability
  and adds per-iteration work.
- Exact best responses are more expensive than a training iteration in Leduc.
- Experiments use modest iteration counts and two seeds; they are diagnostics,
  not broad performance claims.
- No Outcome Sampling MCCFR, CFR+, Deep CFR, function approximation, or
  parallel traversal is implemented.

## Reference

Lanctot, Waugh, Zinkevich, and Bowling (2009), [*Monte Carlo Sampling for
Regret Minimization in Extensive Games*](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf).
Their external-sampling estimator samples chance and opponent actions while
enumerating the updating player's actions. That sampling structure is the
source of the expected counterfactual-regret identity described above; this
implementation's exact average-strategy update is an explicit choice to retain
the repository's vanilla behavioral-average convention.
