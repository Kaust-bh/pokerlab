import math

from pokerlab.research.leduc.game import LeducGame
from pokerlab.research.leduc.sampling_variants import (
    SamplingConfig,
    SamplingVariantSolver,
)

from .test_mccfr import _player0_fixture, _player1_fixture


def _exact_root_average(solver, profile, traverser, *, private_mode, public_mode):
    total = {}
    for deal in LeducGame.deals():
        if private_mode == "rank_stratified":
            # Average each possible draw conditional on its stratum, then
            # weight by the true stratum mass.
            rank_mass = 6 / 30 if deal[0].rank == deal[1].rank else 24 / 30
            private_weight = rank_mass / (6 if deal[0].rank == deal[1].rank else 24)
        else:
            private_weight = 1 / 30
        initial = LeducGame.initial_state(deal)
        delta = {}
        solver._traverse_variant(
            initial,
            traverser,
            profile,
            delta,
            chance_weight=private_weight,
        )
        branches = [(1.0, delta)]
        for branch_probability, delta in branches:
            for info, actions in delta.items():
                target = total.setdefault(info, dict.fromkeys(actions, 0.0))
                for action, value in actions.items():
                    target[action] += branch_probability * value
    return total


def test_leduc_chance_variants_match_hand_derived_regret_fixtures():
    # The reference is a separate, exact sum over all 30 ordered private
    # deals and all public cards. Opponent choices are made deterministic by
    # each fixture, so no sampling remains in this expected value.
    fixtures = ((0, _player0_fixture), (1, _player1_fixture))
    references = {}
    for private_mode in ("sample", "rank_stratified", "enumerate"):
        for public_mode in ("enumerate",):
            solver = SamplingVariantSolver(
                SamplingConfig(
                    iterations=1,
                    seed=7,
                    private_deal_mode=private_mode,
                    public_card_mode=public_mode,
                )
            )
            for player, fixture in fixtures:
                profile, target = fixture(solver)
                for info in solver.information_sets:
                    if info.player != player and all(
                        probability in (0.0, 1.0)
                        for probability in profile[info].values()
                    ):
                        continue
                    if info.player != player:
                        legal = solver._legal[info]
                        profile[info] = {
                            action: float(action is legal[0]) for action in legal
                        }
                estimate = _exact_root_average(
                    solver,
                    profile,
                    player,
                    private_mode=private_mode,
                    public_mode=public_mode,
                )
                key = (player, target)
                observed = estimate.get(target, {})
                if key not in references:
                    references[key] = observed
                for action, value in references[key].items():
                    assert math.isclose(observed[action], value, abs_tol=1e-12)


def test_leduc_variant_averages_are_exact_and_runs_reproduce():
    cfg = SamplingConfig(
        iterations=3,
        seed=91,
        private_deal_mode="rank_stratified",
        public_card_mode="enumerate",
    )
    first = SamplingVariantSolver(cfg)
    second = SamplingVariantSolver(cfg)
    profile = first.current_strategy_profile()
    before = {info: values.copy() for info, values in first.strategy_sum.items()}
    first.run_iteration()
    second.run_iteration()
    assert first.regret_sum == second.regret_sum
    assert first.strategy_sum == second.strategy_sum
    # Chance sampling choices cannot change own-reach accumulation: every
    # catalogued information set receives its exact own-reach contribution.
    expected = first._average_strategy_delta(profile)
    assert first.strategy_sum == {
        info: {
            action: before[info][action] + expected[info][action] for action in values
        }
        for info, values in first.strategy_sum.items()
    }
