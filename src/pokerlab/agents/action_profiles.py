"""Named bet sizing menus shared by observations and research abstractions."""

from __future__ import annotations

from typing import Final

# Each size is (abstract action label, pot fraction); None means the minimum
# legal opening bet or raise. Legacy exactly preserves the Phase 4 abstract
# action set. Other profiles are opt-in.
ACTION_PROFILES: Final = {
    "legacy": {
        "bets": (("bet_half_pot", 0.5), ("bet_pot", 1.0)),
        "raises": (("raise_half_pot", 0.5), ("raise_pot", 1.0)),
    },
    "standard": {
        "bets": (
            ("bet_min", None),
            ("bet_half_pot", 0.5),
            ("bet_pot", 1.0),
        ),
        "raises": (
            ("raise_min", None),
            ("raise_half_pot", 0.5),
            ("raise_pot", 1.0),
        ),
    },
    "wide": {
        "bets": (
            ("bet_min", None),
            ("bet_third_pot", 1 / 3),
            ("bet_half_pot", 0.5),
            ("bet_three_quarter_pot", 0.75),
            ("bet_pot", 1.0),
            ("bet_one_half_pot", 1.5),
            ("bet_two_pot", 2.0),
        ),
        "raises": (
            ("raise_min", None),
            ("raise_third_pot", 1 / 3),
            ("raise_half_pot", 0.5),
            ("raise_three_quarter_pot", 0.75),
            ("raise_pot", 1.0),
            ("raise_one_half_pot", 1.5),
            ("raise_two_pot", 2.0),
        ),
    },
}
