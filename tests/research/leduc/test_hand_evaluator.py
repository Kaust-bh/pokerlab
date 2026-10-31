from pokerlab.research.leduc.cards import Card, Rank
from pokerlab.research.leduc.hand_evaluator import compare_hands, hand_rank


def test_pair_beats_every_high_card_and_higher_pairs_win():
    board = Card(Rank.QUEEN, 0)
    jack = Card(Rank.JACK, 0)
    queen = Card(Rank.QUEEN, 1)
    king = Card(Rank.KING, 0)
    assert hand_rank(queen, board) > hand_rank(king, board)
    assert compare_hands(queen, king, board) == 1
    assert compare_hands(king, jack, board) == 1


def test_same_private_rank_splits_and_pair_public_rank_ties():
    board = Card(Rank.QUEEN, 0)
    assert compare_hands(Card(Rank.JACK, 0), Card(Rank.JACK, 1), board) == 0
    assert compare_hands(Card(Rank.QUEEN, 0), Card(Rank.QUEEN, 1), board) == 0
