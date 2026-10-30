from pokerlab.engine import Pot


def test_side_pots_folded_contributors_and_odd_chip() -> None:
    pot = Pot()
    for player, amount in {"a": 100, "b": 60, "c": 30, "d": 100}.items():
        pot.contribute(player, amount)
    pot.fold("d")
    layers = pot.layers()
    assert [layer.amount for layer in layers] == [120, 90, 80]
    ranks = {"a": 1, "b": 3, "c": 2}
    payouts = pot.distribute(ranks, ["a", "b", "c", "d"])
    assert payouts == {"a": 80, "b": 210, "c": 0, "d": 0}
    assert sum(payouts.values()) == pot.total


def test_tie_splits_with_odd_chip_by_seat_order() -> None:
    pot = Pot()
    pot.contribute("a", 2)
    pot.contribute("b", 2)
    pot.contribute("c", 1)
    pot.fold("c")
    assert pot.distribute({"a": 5, "b": 5}, ["b", "a", "c"]) == {"a": 2, "b": 3, "c": 0}
