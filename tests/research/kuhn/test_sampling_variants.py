import math
from itertools import product

from pokerlab.research.kuhn.game import KuhnGame
from pokerlab.research.kuhn.mccfr import MCCFRConfig, MCCFRSolver
from pokerlab.research.kuhn.sampling_variants import (
    SamplingConfig,
    SamplingVariantSolver,
)

from .test_mccfr import _fixed_profile, _independent_vanilla_updates


def test_stratified_and_enumerated_kuhn_updates_have_exact_expectation():
    reference = MCCFRSolver(MCCFRConfig(iterations=1, seed=3))
    profile = _fixed_profile(reference)
    exact, _ = _independent_vanilla_updates(profile)

    for mode in ("rank_stratified", "enumerate"):
        solver = SamplingVariantSolver(
            SamplingConfig(iterations=1, seed=19, private_deal_mode=mode)
        )
        for traverser in (0, 1):
            aggregate = {}
            if mode == "enumerate":
                samples = solver._private_deal_samples()
            else:
                # Exhaustively average every possible conditional draw in each
                # stratum; p(stratum)/m is the estimator's outer weight.
                samples = [
                    (deal, mass / len(stratum))
                    for mass, stratum in solver._private_strata()
                    for deal in stratum
                ]
            for deal, weight in samples:
                external = tuple(
                    info for info in solver.information_sets if info.player != traverser
                )
                for choices in product(*(solver._legal[info] for info in external)):
                    sampled = dict(zip(external, choices, strict=True))
                    probability = weight
                    for info, choice in sampled.items():
                        probability *= profile[info][choice]
                    if probability == 0:
                        continue
                    delta = {}
                    solver._traverse_variant(
                        KuhnGame.initial_state(deal),
                        traverser,
                        profile,
                        delta,
                        chance_weight=1.0,
                        opponent_choices=sampled,
                    )
                    for info, actions in delta.items():
                        target = aggregate.setdefault(info, dict.fromkeys(actions, 0.0))
                        for action, value in actions.items():
                            target[action] += probability * value
            for info, actions in exact.items():
                if info.player != traverser:
                    continue
                for action, value in actions.items():
                    assert math.isclose(
                        aggregate.get(info, {}).get(action, 0.0),
                        value,
                        abs_tol=1e-12,
                    )


def test_sampling_variant_average_and_seed_reproducibility():
    first = SamplingVariantSolver(
        SamplingConfig(iterations=8, seed=41, private_deal_mode="rank_stratified")
    )
    second = SamplingVariantSolver(
        SamplingConfig(iterations=8, seed=41, private_deal_mode="rank_stratified")
    )
    baseline = MCCFRSolver(MCCFRConfig(iterations=1, seed=5))
    profile = baseline.current_strategy_profile()
    _, average_reference = _independent_vanilla_updates(profile)
    actual = first._average_strategy_delta(profile)
    for info, actions in average_reference.items():
        assert actual[info] == actions

    for _ in range(8):
        first.run_iteration()
        second.run_iteration()
    assert first.regret_sum == second.regret_sum
    assert first.strategy_sum == second.strategy_sum


def test_kuhn_private_strata_have_correct_probability_mass():
    strata = SamplingVariantSolver._private_strata()
    assert [mass for mass, _ in strata] == [1 / 3, 2 / 3]
    assert [len(deals) for _, deals in strata] == [2, 4]
    solver = SamplingVariantSolver(
        SamplingConfig(
            iterations=1,
            seed=4,
            private_deal_mode="rank_stratified",
            samples_per_stratum=3,
        )
    )
    samples = solver._private_deal_samples()
    assert len(samples) == 6
    assert math.isclose(sum(weight for _, weight in samples), 1.0)
    for mass, deals in strata:
        assert math.isclose(
            sum(weight for deal, weight in samples if deal in deals), mass
        )
