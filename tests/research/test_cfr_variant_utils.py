import math

import pytest

from pokerlab.research.cfr_variant_utils import (
    dcfr_negative_discount,
    dcfr_positive_discount,
    dcfr_strategy_discount,
    discount_regret_sum,
    discount_strategy_sum,
    regret_matching_plus,
)


def test_regret_matching_plus_clips_and_has_uniform_fallback():
    assert regret_matching_plus({"a": 3.0, "b": -2.0}, ("a", "b")) == {
        "a": 1.0,
        "b": 0.0,
    }
    assert regret_matching_plus({"a": -1.0, "b": 0.0}, ("a", "b")) == {
        "a": 0.5,
        "b": 0.5,
    }


def test_dcfr_discounts_positive_negative_and_strategy_sums():
    positive_factor = 3**1.5 / (3**1.5 + 1)
    assert math.isclose(dcfr_positive_discount(3, 1.5), positive_factor)
    assert dcfr_negative_discount(3, 0.0) == 0.5
    assert math.isclose(dcfr_strategy_discount(3, 2.0), (3 / 4) ** 2)
    assert math.isclose(
        discount_regret_sum(8.0, 2.0, 3, 1.5, 0), 8 * positive_factor + 2
    )
    assert math.isclose(discount_regret_sum(-8.0, 2.0, 3, 1.5, 0), -2.0)
    assert math.isclose(discount_strategy_sum(8.0, 2.0, 3, 2), 8 * (3 / 4) ** 2 + 2)


def test_dcfr_factors_remain_finite_for_large_iteration_indices():
    iteration = 10**500
    assert dcfr_positive_discount(iteration, 1.5) == 1.0
    assert dcfr_negative_discount(iteration, 0.0) == 0.5
    assert 0.0 < dcfr_strategy_discount(iteration, 2.0) <= 1.0


@pytest.mark.parametrize("values", [(float("nan"), 0.0, 2.0), (1.0, 0.0, -1.0)])
def test_invalid_dcfr_parameters_are_rejected(values):
    from pokerlab.research.kuhn.dcfr import DCFRConfig

    with pytest.raises(ValueError):
        DCFRConfig(alpha=values[0], beta=values[1], gamma=values[2])


def test_dcfr_config_accepts_custom_finite_parameters():
    from pokerlab.research.leduc.dcfr import DCFRConfig

    config = DCFRConfig(alpha=2.25, beta=-0.5, gamma=1.25)
    assert (config.alpha, config.beta, config.gamma) == (2.25, -0.5, 1.25)
