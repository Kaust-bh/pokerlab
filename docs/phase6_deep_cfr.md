# Phase 6A: Deep CFR prototype

## Scope

Deep CFR is implemented under `src/pokerlab/algorithms/deep_cfr/`. It reuses
the existing Kuhn and Leduc game definitions and exact exploitability
evaluators. Existing tabular CFR, CFR+, DCFR, MCCFR, and Hold'em equity APIs
remain separate. PyTorch is an optional dependency, installed with
`python -m pip install -e '.[deep-cfr]'`; importing PokerLab without that extra
continues to work.

The experiment runner compares Deep CFR against vanilla CFR and CFR+ in Kuhn
and Leduc. It does not start HUNL neural training.

## Algorithm and estimator

The implementation follows Brown, Lerer, Gross, and Sandholm (2019),
[*Deep Counterfactual Regret Minimization*](https://proceedings.mlr.press/v97/brown19b.html),
using their external-sampling traversal and linear iteration weights by
default. Uniform iteration weights are also configurable for comparison with
vanilla CFR. Each CFR iteration performs `K` traversals for Player 0 and `K`
for Player 1.

At a traverser information set `I`, the current behavioral strategy is regret
matching over the traverser's advantage network output. Every legal action is
recursively evaluated; the stored sampled target is

\[
\tilde r(I,a) = \tilde v(I,a) - \sum_{b\in A(I)}\sigma(I,b)\tilde v(I,b).
\]

At chance nodes the traversal samples using the game's actual chance
probabilities. At opponent nodes it records the current strategy target and
samples one legal action from that strategy. At terminal nodes it returns the
traverser's utility. Thus the action branches used for an advantage target are
fully enumerated at the traverser's node, while chance and opponent branches
use external sampling.

For external sampling, the sampled instantaneous advantage conditioned on
visiting `I` is an unbiased estimate of the counterfactual action advantage
scaled by the inverse counterfactual reach of `I`. That scalar is common to all
legal actions at `I`, so regret matching is unchanged by it. The traversal
samples chance and opponent behavior directly; the implementation does not
multiply advantage targets by the traverser's own reach. See Lemma 1 and
Algorithm 2 in the paper for the estimator and its assumptions.

Advantage memories are separate per player. Each item stores the encoded
information set, a full action target vector, a legal-action mask, player,
iteration and training weight. The bounded reservoirs use standard reservoir
sampling: for stream item `n`, replace a uniformly selected slot from `[0,n)`
only when its index is below capacity. Advantage loss is masked MSE over legal
actions. Networks are refit from scratch each iteration, following the paper;
optimizer state therefore does not carry across iterations.

The shared strategy memory stores targets at opponent nodes during each
player's traversal. The chance and sampled-opponent path supplies those parts
of the reach probability. Since the traverser's actions are enumerated rather
than sampled, each stored row is additionally weighted by the traverser's own
reach probability at that history. Thus a row for player `p` has regression
weight `t * pi_traverser(I)` and occurs with probability proportional to
`pi_chance(I) * pi_p(I)`; in expectation this recovers the full
`t * pi_p(I)` average-strategy objective. Zero-reach rows have zero objective
mass and are not stored. With `linear` weighting, `t` matches the paper's
Linear CFR choice. The strategy net is trained by legal-action-masked weighted
MSE and has a separate output head from either advantage net.

Regret matching clamps negative network predictions to zero and normalizes
over legal actions; if no legal prediction is positive, it uses a uniform
legal policy. This is the project's stable baseline convention rather than the
paper's optional deterministic tie-breaking variant. The network's average
strategy output masks illegal actions before softmax, so they receive exactly
zero probability.

## Networks, memory, and reproducibility

Each player has an advantage MLP. A separate shared MLP approximates the
average strategy; player identity is part of each feature vector. Hidden layer
sizes, ReLU/tanh/leaky-ReLU activation, learning rate, Adam/SGD, batch size,
training epochs, gradient clipping and device are configurable. CPU is the
default; CUDA is accepted only when available. The final linear layer is
initialized to zero so the first regret-matched profile is uniform.

Kuhn encoding is 17 floats: player identity, private rank, and three
position-specific one-hot public betting-history slots. Leduc encoding is 57
floats: player identity, the six-card private identity, public-card identity
including a no-card slot, round, and four action slots for each round. Betting
contributions are determined by the public action sequence in these games.
Neither encoding accepts a full state or the opponent's private card. The
HUNL adapter boundary is the `GameCodec` protocol; a future HUNL codec must
include the complete player-visible state (hole cards, public board, position,
history, pot/stack and legal-action context) and prove that encodings do not
merge strategically distinct observations. No HUNL codec or HUNL training is
included here.

Python traversal and reservoir RNGs are seeded independently and checkpointed.
Network initialization and minibatch ordering use deterministic seeds derived
from the configured seed and iteration. Checkpoints include both advantage
networks, the strategy network, memories, RNG state, iteration count and
diagnostics. Resume equivalence is tested against uninterrupted CPU training.
Inference supports deterministic argmax or stochastic action selection.

## Validation

Controlled toy games test chance-sampled advantages and opponent-action
sampling against independently calculated expected values. Tests also cover
reservoir inclusion, capacity, deterministic replay, legal masks, distinct
player networks, information-set encodings, complete Kuhn play, exact profile
evaluation, Leduc inference, and checkpoint/resume.

The seeded comparison runner writes configuration, per-checkpoint convergence
CSV, exact strategy JSON, Deep CFR checkpoints, elapsed training/evaluation
times, memory sizes, losses, traversal counts, a plot, and a hash manifest.
Exact NashConv/exploitability is computed for all three solvers. The Deep CFR
experiment uses the configured (default linear) iteration weighting while
the tabular reference uses its existing vanilla CFR and CFR+ conventions;
iteration counts therefore do not imply identical update rules.

## Initial pilot runs (preserved, pre-correction)

The first two pipeline pilots exposed a missing average-strategy reach factor:
the traverser was enumerating its own actions, but the corresponding reach was
not included in strategy-memory weights. These original outputs and their
manifests are preserved, but their neural strategy results are not valid
evidence for the corrected average-strategy estimator. The code and tests below
include the correction; corrected runs are reported separately.

Kuhn, seed 7, 100 iterations, four traversals/player/iteration, 20,000-item
memories, one epoch, batch size 64 and uniform iteration weights completed in
2.34 seconds total. Deep CFR used 1.68 seconds of training and reached exact
exploitability 0.3967. At the same iteration count, vanilla CFR reached
0.02567 and CFR+ 0.001194. This small neural run did not approach tabular
accuracy; its low traversal and regression budget is inadequate for a claim
of convergence. The CSV includes checkpoints at 20, 40, 60, 80 and 100.

Leduc, seed 19, 25 iterations, one traversal/player/iteration, 20,000-item
memories, one epoch, batch size 128 and linear weighting completed in 12.14
seconds including exact evaluations. Deep CFR reached exploitability 2.3556;
vanilla CFR reached 0.5181 and CFR+ 0.09895 at iteration 25. The Deep CFR
training time was 0.82 seconds, but exact evaluation took about 0.34 seconds
per checkpoint. This run is a pipeline and cost pilot, not evidence of useful
Leduc play.

Artifacts:

- `results/deep_cfr/run-kuhn-20261031T014744Z-fed96d27/`
- `results/deep_cfr/run-leduc-20261031T014809Z-00e87127/`
- `results/deep_cfr/run-kuhn-20261031T015104Z-a5972a34/` (higher traversal/
  regression budget follow-up, also pre-correction)

These artifact directories were not overwritten by the correction.

## Corrected estimator validation runs

After adding the traverser-reach factor, both pilots were rerun with the same
seeds and game budgets. The new outputs are separate from the preserved
pre-correction directories above.

Kuhn, seed 7, 100 iterations, four traversals per player per iteration,
20,000-item memories, one training epoch, batch size 64 and uniform iteration
weights took 2.30 seconds total. Deep CFR used 1.64 seconds of training and
reached exact exploitability 0.4232; vanilla CFR reached 0.02567 and CFR+
0.001194. Process peak RSS was 332,840,960 bytes. This very small run shows
the implementation and exact-evaluation pipeline working; it does not show
useful convergence.

Leduc, seed 19, 25 iterations, one traversal per player per iteration,
20,000-item memories, one epoch, batch size 128 and linear iteration weights
took 12.18 seconds total. Deep CFR used 0.632 seconds of training and reached
exact exploitability 2.3575; vanilla CFR reached 0.5181 and CFR+ 0.09895.
Exact evaluation took about 0.349 seconds per checkpoint. Process peak RSS was
322,322,432 bytes. This is also a pipeline and cost pilot only.

Corrected outputs:

- `results/deep_cfr/run-kuhn-20261031T015327Z-fed96d27/`
- `results/deep_cfr/run-leduc-20261031T015334Z-00e87127/`

The comparison records exact exploitability at each checkpoint and keeps
training and evaluation time separate. Both corrected runs remain much less
accurate than their tabular references at equal iteration counts. The runs use
single seeds and deliberately small optimization budgets, so they do not
support a general conclusion about attainable Deep CFR quality. All
`artifact_sha256` entries in these manifests were checked against the saved
files after generation.

The experiment manifest source hashes reflect the exact implementation used
for each run. Artifact hashes were verified after generation.

## Reproduction and limitations

Install optional support and run the default Kuhn experiment:

```sh
python -m pip install -e '.[deep-cfr]'
PYTHONPATH=src python -m pokerlab.algorithms.deep_cfr.experiments \
  --game kuhn --iterations 100 --seed 7 --traversals-per-player 4 \
  --evaluation-interval 20 --weighting uniform
```

Run the smaller Leduc experiment with:

```sh
PYTHONPATH=src python -m pokerlab.algorithms.deep_cfr.experiments \
  --game leduc --iterations 25 --seed 19 --traversals-per-player 1 \
  --evaluation-interval 5 --weighting linear
```

Each run creates a new directory under `results/deep_cfr/`; existing results
are not overwritten. Resume is available through `DeepCFRTrainer.load_checkpoint`
and `train(additional_iterations)`.

This is a small-game prototype. Approximation quality depends on traversal
count, reservoir size, network capacity, optimization budget and encoding. The
pilot results are much worse than the tabular baselines. Exact evaluation is
available only because Kuhn and Leduc are small. HUNL's very large observation
space, action abstraction and memory requirements have not been profiled here;
do not treat the codec interface as evidence that HUNL Deep CFR is ready.
