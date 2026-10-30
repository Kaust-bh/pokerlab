import pytest

from pokerlab.engine import Card, HandCategory, evaluate_hand


def c(rank: int, suit: str = "spades") -> Card:
    return Card(rank, suit)


@pytest.mark.parametrize(
    "cards,category",
    [
        ([c(10), c(11), c(12), c(13), c(14)], HandCategory.ROYAL_FLUSH),
        ([c(14), c(2), c(3), c(4), c(5)], HandCategory.STRAIGHT_FLUSH),
        (
            [c(9), c(9, "hearts"), c(9, "clubs"), c(9, "diamonds"), c(2)],
            HandCategory.FOUR_OF_A_KIND,
        ),
        (
            [c(8), c(8, "hearts"), c(8, "clubs"), c(2), c(2, "hearts")],
            HandCategory.FULL_HOUSE,
        ),
        ([c(2), c(5), c(8), c(11), c(13)], HandCategory.FLUSH),
        (
            [c(2), c(3, "hearts"), c(4, "clubs"), c(5, "diamonds"), c(6)],
            HandCategory.STRAIGHT,
        ),
        (
            [c(7), c(7, "hearts"), c(7, "clubs"), c(4), c(2)],
            HandCategory.THREE_OF_A_KIND,
        ),
        ([c(7), c(7, "hearts"), c(4), c(4, "hearts"), c(2)], HandCategory.TWO_PAIR),
        ([c(7), c(7, "hearts"), c(4), c(3), c(2)], HandCategory.ONE_PAIR),
        (
            [c(14), c(11, "hearts"), c(8, "clubs"), c(4, "diamonds"), c(2)],
            HandCategory.HIGH_CARD,
        ),
    ],
)
def test_all_categories(cards: list[Card], category: HandCategory) -> None:
    assert evaluate_hand(cards).category == category


def test_kickers_wheel_best_of_seven_and_board_tie() -> None:
    assert evaluate_hand(
        [c(14), c(2), c(3), c(4), c(5), c(13, "hearts"), c(13, "clubs")]
    ).tiebreak == (5,)
    better = evaluate_hand(
        [c(14), c(14, "hearts"), c(13), c(13, "hearts"), c(12), c(2), c(3)]
    )
    weaker = evaluate_hand(
        [
            c(14, "clubs"),
            c(14, "diamonds"),
            c(13, "clubs"),
            c(13, "diamonds"),
            c(11),
            c(2, "clubs"),
            c(3, "clubs"),
        ]
    )
    assert better > weaker
    board = [c(2), c(3, "hearts"), c(4, "clubs"), c(5, "diamonds"), c(6)]
    assert evaluate_hand([c(14, "hearts"), c(14, "clubs"), *board]) == evaluate_hand(
        [c(13, "hearts"), c(13, "clubs"), *board]
    )
    assert evaluate_hand([c(14), c(13, "hearts"), *board]) == evaluate_hand(
        [c(12), c(11, "hearts"), *board]
    )


def test_full_house_and_flush_kickers() -> None:
    assert evaluate_hand(
        [c(10), c(10, "hearts"), c(10, "clubs"), c(2), c(2, "hearts")]
    ) > evaluate_hand([c(9), c(9, "hearts"), c(9, "clubs"), c(14), c(14, "hearts")])
    assert evaluate_hand([c(2), c(5), c(8), c(11), c(14)]) > evaluate_hand(
        [
            c(2, "hearts"),
            c(5, "hearts"),
            c(8, "hearts"),
            c(11, "hearts"),
            c(13, "hearts"),
        ]
    )


def test_two_pair_kicker_and_invalid_hand() -> None:
    assert evaluate_hand(
        [c(14), c(14, "hearts"), c(13), c(13, "hearts"), c(12)]
    ) > evaluate_hand(
        [c(14, "clubs"), c(14, "diamonds"), c(13, "clubs"), c(13, "diamonds"), c(11)]
    )
    with pytest.raises(ValueError):
        evaluate_hand([c(2)] * 5)
