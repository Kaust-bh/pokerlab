from pokerlab.engine import ActionType, Card, Deck, GameState, Street


def test_complete_hand_chip_conservation() -> None:
    game = GameState({"a": 100, "b": 100}, deck=Deck(seed=42))
    assert sum(game.stacks.values()) + game.current_pot == 200
    game.act(ActionType.CALL)
    game.act(ActionType.CHECK)
    assert game.street == Street.FLOP
    for action in (
        ActionType.CHECK,
        ActionType.CHECK,
        ActionType.CHECK,
        ActionType.CHECK,
        ActionType.CHECK,
        ActionType.CHECK,
    ):
        if game.complete:
            break
        game.act(action)
    assert game.complete
    assert sum(game.stacks.values()) == 200
    assert sum(game.winners.values()) == 4


def test_fold_wins_before_showdown_and_invalid_action_is_unchanged() -> None:
    game = GameState({"a": 100, "b": 100}, deck=Deck(seed=5))
    before = (dict(game.stacks), game.current_pot, game.current_player)
    try:
        game.act(ActionType.RAISE, 1)
    except ValueError:
        pass
    assert before == (game.stacks, game.current_pot, game.current_player)
    game.act(ActionType.FOLD)
    assert game.complete
    assert sum(game.stacks.values()) == 200


def test_all_in_runout_and_shared_board_tie() -> None:
    # Ordered deck deals hole cards first, then a board that plays for both players.
    cards = [
        Card(14, "clubs"),
        Card(14, "diamonds"),
        Card(13, "clubs"),
        Card(13, "diamonds"),
        Card(2, "clubs"),
        Card(3, "diamonds"),
        Card(4, "hearts"),
        Card(5, "spades"),
        Card(6, "clubs"),
    ]
    deck = Deck(seed=1)
    deck._cards = list(reversed(cards))
    game = GameState({"a": 10, "b": 10}, small_blind=1, big_blind=2, deck=deck)
    game.act(ActionType.ALL_IN)
    game.act(ActionType.CALL)
    assert game.complete
    assert game.community_cards == cards[4:9]
    assert game.stacks == {"a": 10, "b": 10}
