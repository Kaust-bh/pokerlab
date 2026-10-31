# Phase 4: Abstracted Heads-Up No-Limit Hold'em

This is a CPU-only research prototype that adapts the existing heads-up NLHE
engine to a reduced action and private-hand game. The engine remains the source
of truth for card dealing, betting transitions, all-ins, showdown and chip
settlement. `AbstractHUNLState` clones the mutable engine state when the solver
branches over a traverser's actions, preserving the sampled deck order across
those branches.

## Game and betting abstraction

The configurable default game uses 100 BB effective stacks, 0.5/1 BB blinds,
no antes or rake, and half-BB integer chip units. A small 5 BB training profile
is used for the checked-in smoke experiment because full no-limit action trees
are expensive even under sampling. Buttons are sampled uniformly during
training; evaluation alternates the button each hand.

The default `legacy` action profile preserves the Phase 4 menu: at a no-bet
node it is check, half-pot bet, pot bet, and all-in; facing a bet it is fold,
call, half-pot raise, pot raise, and all-in. The opt-in `standard` profile adds
minimum legal bet/raise actions. The `wide` profile adds one-third, three-quarter,
one-and-a-half-pot, and two-pot sizes. Bet targets use
`contribution + max(minimum_opening_bet, floor(pot * fraction))`. Raise targets
use `current_bet + max(minimum_raise, floor((pot + call_amount) * fraction))`.
Targets are clipped to the actor's maximum contribution. Sizes that reach the
stack cap are represented by all-in; duplicate sizes are removed. The engine
still validates min-raise, short all-in, and raise-reopening semantics.

## Card abstraction and information sets

Preflop hands use the standard 169 rank/suitedness classes. By default,
postflop features combine the retained starting class, current made-hand
category, deterministic sampled percentile versus random legal opponent
holdings, a flush-draw flag, and a coarse straight-out bucket. This preserves
the original Phase 4 abstraction. Its sampled percentile describes
current-board hand strength; it is not future-street equity.

An opt-in `sampled_equity` method estimates showdown equity over legal sampled
opponent holdings and future runouts. It adds configurable equity buckets,
frontdoor/backdoor flush-draw labels, straight-out buckets, and a compact board
texture label. The opponent prior can be uniform or the static `premium` class
set. That prior is not conditioned on the opponent's observed betting actions.
Equity sampling and opponent range settings are deterministic functions of the
player's cards, visible board and configuration; opponent hole cards are never
inputs. Expensive features use bounded caches.

The information-set key contains button role, street, private bucket, exact
public board, pot, effective stack, role-ordered street contributions, public
action history and abstract legal actions. It has no opponent or undealt cards.
The starting hand class remains part of postflop buckets to retain the player's
own private information across streets. Exact public boards are intentionally
not abstracted, so the state space still grows rapidly.

## MCCFR and average strategy

Each iteration snapshots the behavioral strategy before either player pass.
Each regret pass samples button/deck chance from its true distribution,
samples opponent actions from the frozen profile, and enumerates all actions
at the traverser's information sets. Its action regret is `u(a) - sum_b
sigma(b)u(b)` in the traverser's utility convention. Chance and opponent reach
are represented by the sampling distribution; no traverser-reach multiplier
is used in regret updates. Updates are accumulated and committed after both
passes.

Average strategy uses two separate sampled traversals per iteration. They
enumerate traverser actions and sample opponent actions from a uniform
full-support proposal. A contribution at information set `I` is
`pi_i(I) * sigma_i(I,a) / (p_chance(I) * q_opponent(I))`. Taking expectation
over the sampled external trajectory cancels the denominator and recovers the
own-reach strategy numerator. Multiple concrete hidden states can map to one
abstract information set; they add a fixed multiplicity to all action masses
at that key, which cancels when the policy is normalized. This is distinct
from regret estimation. Importance weights can have high variance, and the
traverser action expansion can still be large. The exact normalization claim
assumes the current abstraction's public history and legal actions identify a
fixed set of hidden histories; changing the abstraction requires revisiting
that assumption.

Tests include a controlled two-outcome regret reference and an independently
enumerated importance-weighted average-policy calculation. These validate the
local equations, not convergence or exploitability in full HUNL. No exact
best-response evaluator exists for the full abstracted game, so this phase does
not report HUNL exploitability or claim equilibrium quality.

## Reproduction and evaluation

Install the development environment and run:

```sh
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
```

Run a smoke-scale experiment (it creates a fresh folder and does not overwrite
prior runs):

```sh
pokerlab-nlhe-mccfr --iterations 1 --seed 7 --hands 4 --stack-bb 5
```

Or use `PYTHONPATH=src python -m pokerlab.research.nlhe.experiments` with the
same options. Outputs under `results/nlhe_abstraction/` include config,
average strategy, training metrics, match-level evaluation CSV, a baseline
comparison CSV, and a manifest with Git HEAD/status and SHA-256 hashes. The
training CSV records per-iteration time, sampled nodes, information-set count,
cumulative regret L1 and average-policy L1 change; those are solver diagnostics,
not exact convergence or exploitability measurements. The
runner compares against random, heuristic and low-simulation equity agents.
The pilot uses only two matches per opponent, so Wilson intervals are wide and
results are exploratory; simple-agent performance does not establish game
theoretic quality. Evaluation net chips and BB/100 are sample outcomes, not
win-rate estimates over the full game distribution.

Select the improved abstraction options explicitly, for example:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.experiments \
  --iterations 1 --seed 105 --hands 1 --stack-bb 3 \
  --postflop-method sampled_equity --postflop-buckets 8 \
  --equity-samples 8 --opponent-range premium --action-profile wide
```

The JSON configuration records every game, hand-abstraction, sampling and
action-profile setting. Existing result folders and Phase 4 defaults are not
rewritten by this command.

## Limitations

- Abstracted actions omit many legal no-limit sizes and can change strategic
  incentives.
- Postflop buckets are approximate and may merge strategically distinct
  holdings; the sampled opponent percentile can be noisy.
- Retaining exact boards and on-demand histories does not make the game small;
  external sampling reduces chance/opponent branching but traverser branching
  and average-strategy importance weights remain costly.
- Training begins from sampled hands and a sparse strategy table; unseen keys
  use a uniform policy.
- The experiment runner is a reproducibility smoke test, not a benchmark study.
- There is no exact full-game exploitability calculation or unrestricted HUNL
  solution claim.
