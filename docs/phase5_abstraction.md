# Phase 5A: HUNL abstraction improvements

## Audit and motivation

The Phase 4 implementation was inspected before changes. Its legacy hand method
ranked the player's current 5–7-card holding against a deterministic sample of
random opponent holdings. That is a current-board rank percentile, not equity
through a future runout. The solver also exposed only half-pot and pot sizing.
The information set already kept the exact public board and excluded opponent
cards; that privacy boundary and the MCCFR update equations are retained.

A cold-cache `cProfile` run of five training iterations used seed 2026, a 5 BB
stack, 20 hand buckets and 128 opponent samples. Before these changes it made
5,218,962 calls and took 1.045 profiled seconds. Whole-state `deepcopy` took
about 0.615 seconds cumulatively; postflop feature evaluation took about 0.384
seconds. This identified branch cloning and repeated straight-out hand
evaluation as the optimization targets. After branch-local cloning and
rank-mask draw evaluation, the same profiled workload made 1,156,282 calls and
took 0.275 profiled seconds. That is a 73.7% reduction for this small workload,
not a general throughput claim. The bounded legacy feature cache had 48 misses
and 77 hits during the after run.

The optional equity path was separately profiled for three iterations with 16
equity samples per feature, seed 2026 and a 5 BB stack. It took 0.229 profiled
seconds with 28 cache misses and 51 hits. More samples and deeper stacks
increase this cost. These measurements depend on sampled paths and cache state.

## Hand abstraction choices

`HUNLGameConfig` retains existing defaults and adds these options:

| Setting | Default | Options / effect |
|---|---|---|
| `postflop_method` | `sampled_strength` | Legacy current-board percentile or `sampled_equity` |
| `postflop_buckets` | 20 | Number of percentile/equity bins |
| `opponent_hand_samples` | 128 | Legacy current-board comparison sample count |
| `equity_samples` | 64 | Opponent holding and future-runout trials in equity mode |
| `opponent_range` | `uniform` | All legal holdings or a fixed premium starting-hand set |

The sampled equity is

\[
\widehat E = \frac{W + \frac{1}{2}T}{N},
\]

where a trial samples a legal opponent holding from the configured static prior,
then a legal completion of the public board uniformly without replacement. It
evaluates both final hands using the existing evaluator. On the river, trials
sample only opponent holdings. Ties contribute one half. A bucket is
`min(B-1, floor(B * E))`, so equity 1.0 maps to the final bucket. The stable
seed is derived from the player's cards, visible board, sample count and range
choice. Suits are canonicalized before sampling, so global suit relabeling does
not change this feature.

The `premium` range is a fixed illustrative prior consisting of AA, KK, QQ,
JJ, TT, AKs, AKo, AQs, AQo and KQs after card removal. It is not inferred from
public actions and is not a learned opponent range. The Monte Carlo estimate is
approximate; its uncertainty is not encoded in the bucket, so low sample counts
can move a hand across a boundary.

The equity method adds a compact board texture descriptor for pairing, suit
concentration and rank connectivity, plus separate flush and straight draw
features. Exact public boards remain in information sets, so texture is a
feature for inspection and bucketing rather than a public-board abstraction.
The legacy method and its private-bucket labels remain the default.

The legacy draw feature was audited. Its prior straight-out count tested
whether a next card made a hand category at least `STRAIGHT`, which also counts
flush completions as straight outs. That legacy behavior remains in the
default bucket for saved-strategy compatibility. The equity method separates
flush-draw codes from straight outs. Straight detection now uses rank masks
instead of invoking the general evaluator for every unseen card; a regression
test compares the legacy result with the evaluator-based reference.

## Action abstraction profiles

The `action_profile` setting is recorded in configuration, observations and
tournament settings. All profiles include check/fold/call as appropriate and a
distinct all-in.

| Profile | Opening bet and raise sizes |
|---|---|
| `legacy` | Half-pot, pot |
| `standard` | Minimum legal, half-pot, pot |
| `wide` | Minimum legal, one-third-pot, half-pot, three-quarter-pot, pot, 1.5-pot, 2-pot |

Sizes use integer chip targets and the engine's target street-contribution
convention. Opening bets use the current pot; raises use the pot after calling.
Duplicate targets collapse to one abstract action. Targets capped at the stack
are represented by all-in. The existing engine continues to enforce minimum
raises, short all-ins and reopening rules. `legacy` remains the default and
preserves the Phase 4 solver menu.

## MCCFR and privacy

No regret or average-strategy equations changed. Regret traversals still use
one frozen profile for both players, sample chance and opponent actions,
enumerate traverser actions, and exclude traverser reach from regret updates.
Average strategy still uses separate uniform full-support opponent proposals
with inverse chance/proposal-reach weighting. New action choices enter through
the same legal action enumeration and the information-set `legal_actions`
field.

Postflop features are computed only from the acting player's cards, visible
board and serialized configuration. The opponent's actual cards are never an
input. The information set still contains role, street, own abstract bucket,
exact public board, pot, effective stack, contributions, public history and
legal abstract actions.

## Performance changes

The game adapter now copies the mutable engine containers needed by a branch:
stacks, deck card order, contributions/pot state, betting sets and public board.
Cards, action/history tuples and immutable configuration are shared. The deck
shuffle RNG is shared safely because later transitions only pop from the
already shuffled branch-local card list. Tests verify that a branch cannot
mutate its parent or sibling.

Both hand-feature methods use bounded LRU caches. The rank-mask draw
calculation preserves the legacy bucket exactly, including its old flush/straight
label conflation. No new dependencies or engine changes were introduced.

## Reproduction

Legacy smoke configuration:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.experiments \
  --iterations 1 --seed 7 --hands 4 --stack-bb 5
```

Sampled equity with premium range and wide actions:

```sh
PYTHONPATH=src python -m pokerlab.research.nlhe.experiments \
  --iterations 1 --seed 205 --hands 1 --stack-bb 3 \
  --postflop-method sampled_equity --postflop-buckets 8 \
  --equity-samples 8 --opponent-range premium --action-profile wide
```

Each run creates a fresh `results/nlhe_abstraction/run-*` directory. Its
manifest records Git HEAD/status, source hashes and artifact hashes. Its config
JSON records the exact game, sampling and abstraction settings. Existing Phase
4 artifacts remain intact.

The sampled-equity pilot at
`results/nlhe_abstraction/run-20261031T000727Z-205/` trained one iteration in
0.0156 seconds, visited 32 information sets and sampled 57 nodes. It evaluated
two one-hand matches against each baseline; MCCFR won one match against each
baseline, with sample BB/100 results of 50 against Random, 0 against Heuristic,
and −50 against Equity. Every 95% Wilson interval was 0.095–0.905. Those
outcomes are too small and noisy for a strength claim. No full-game
exploitability is available for this abstraction. The artifact and source
hashes in its manifest verify.

## Limitations

- Sampled equity costs more than legacy current-board ranking and remains noisy
  at modest sample counts.
- The premium range is static and is not conditioned on observed betting.
- Exact boards, pot sizes, stacks and histories still fragment information
  sets; these changes do not make unrestricted HUNL tractable.
- The `wide` profile increases traverser branching. Different profiles are
  different abstract games and should not be compared as identical trees.
- The legacy method preserves a known flush/straight-draw label conflation for
  compatibility. New experiments can use `sampled_equity` for separate draw
  features.
- This phase does not implement Deep CFR, neural approximation or Phase 5B.
