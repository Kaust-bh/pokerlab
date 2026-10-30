import pytest

from pokerlab.algorithms import estimate_equity
from pokerlab.engine import Card


def test_monte_carlo_distribution_bounds_and_reproducibility() -> None:
    hole = (Card(14, "spades"), Card(14, "hearts"))
    first = estimate_equity(hole, simulations=300, seed=123, method="monte_carlo")
    second = estimate_equity(hole, simulations=300, seed=123, method="monte_carlo")
    assert (
        first.win_probability,
        first.tie_probability,
        first.loss_probability,
        first.equity,
        first.simulations,
        first.confidence_interval,
    ) == (
        second.win_probability,
        second.tie_probability,
        second.loss_probability,
        second.equity,
        second.simulations,
        second.confidence_interval,
    )
    assert (
        first.win_probability + first.tie_probability + first.loss_probability
        == pytest.approx(1)
    )
    assert 0 <= first.equity <= 1
    assert (
        0
        <= first.confidence_interval[0]
        <= first.equity
        <= first.confidence_interval[1]
        <= 1
    )
    assert first.simulations == 300


def test_river_exact_enumeration_finds_shared_board_tie() -> None:
    hole = (Card(2, "clubs"), Card(3, "diamonds"))
    board = (
        Card(10, "spades"),
        Card(11, "spades"),
        Card(12, "spades"),
        Card(13, "spades"),
        Card(14, "spades"),
    )
    result = estimate_equity(hole, board, seed=1)
    assert result.method == "exact"
    assert result.win_probability == 0
    assert result.tie_probability == 1
    assert result.loss_probability == 0
    assert result.equity == 0.5
    assert result.confidence_interval == (0.5, 0.5)


def test_rejects_duplicate_or_incomplete_board_cards() -> None:
    duplicate = Card(14, "spades")
    with pytest.raises(ValueError, match="unique"):
        estimate_equity((duplicate, duplicate), simulations=1)
    with pytest.raises(ValueError, match="0, 3, 4, or 5"):
        estimate_equity(
            (Card(2, "clubs"), Card(3, "clubs")), (Card(4, "clubs"),), simulations=1
        )


def test_exact_turn_estimate_is_normalized() -> None:
    result = estimate_equity(
        (Card(14, "spades"), Card(14, "hearts")),
        (Card(2, "clubs"), Card(7, "diamonds"), Card(9, "spades"), Card(11, "clubs")),
    )
    assert result.method == "exact"
    assert result.simulations == 45_540
    assert (
        result.win_probability + result.tie_probability + result.loss_probability
        == pytest.approx(1)
    )
