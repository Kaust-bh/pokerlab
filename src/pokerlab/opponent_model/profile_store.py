"""Opponent Profile Manager and safe JSON persistence for HarveySpecter."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pokerlab.agents.base import ActionRecord, Observation
from pokerlab.engine import Card
from pokerlab.opponent_model.archetypes import ArchetypeClassifier
from pokerlab.opponent_model.collector import OpponentDataCollector
from pokerlab.opponent_model.models import (
    MODEL_VERSION,
    ArchetypeDistribution,
    HandObservationRecord,
    ObservedActionEvent,
    OpponentStatsSnapshot,
    RangeSnapshot,
)
from pokerlab.opponent_model.range_model import BayesianRangeModel
from pokerlab.opponent_model.statistics import BehaviouralProfiler


class ProfileStorageError(ValueError):
    """Raised when an opponent profile file or payload is invalid or corrupted."""


class OpponentProfile:
    """In-memory and persistent intelligence profile for a single opponent."""

    def __init__(
        self,
        opponent_id: str = "opponent",
        *,
        recency_decay: float = 0.94,
        cold_start_hands: float = 8.0,
    ) -> None:
        self.opponent_id = opponent_id.strip() or "opponent"
        self.model_version = MODEL_VERSION
        self.collector = OpponentDataCollector(default_opponent_id=self.opponent_id)
        self.profiler = BehaviouralProfiler(recency_decay=recency_decay)
        self.classifier = ArchetypeClassifier(cold_start_hands=cold_start_hands)
        self.range_model = BayesianRangeModel()
        self.last_hand_number: int = 0
        self._cached_stats: OpponentStatsSnapshot | None = None
        self._cached_archetype: ArchetypeDistribution | None = None

    def _invalidate_cache(self) -> None:
        self._cached_stats = None
        self._cached_archetype = None

    def reset(self) -> None:
        """Reset all collected hands, statistics, and Bayesian calibration for this opponent."""
        self.collector.reset(self.opponent_id)
        self.range_model.reset_all()
        self.last_hand_number = 0
        self._invalidate_cache()

    def start_hand(
        self,
        *,
        hand_number: int | None = None,
        opponent_is_button: bool,
        starting_stack: int = 200,
        effective_stack: int | None = None,
        big_blind: int = 2,
        hero_hole: Sequence[Card] = (),
        community_cards: Sequence[Card] = (),
    ) -> HandObservationRecord:
        record = self.collector.start_hand(
            opponent_id=self.opponent_id,
            hand_number=hand_number,
            opponent_is_button=opponent_is_button,
            starting_stack=starting_stack,
            effective_stack=effective_stack,
            big_blind=big_blind,
        )
        self.last_hand_number = max(self.last_hand_number, record.hand_number)
        self.range_model.start_hand(
            hand_number=record.hand_number,
            hero_hole=hero_hole,
            community_cards=community_cards,
        )
        self._invalidate_cache()
        return record

    def ingest_observation(
        self,
        observation: Observation,
        *,
        hand_number: int | None = None,
    ) -> tuple[HandObservationRecord, list[ObservedActionEvent]]:
        """Update collector and Bayesian range model from a public `Observation`."""
        prev_active = self.collector.get_active_hand(self.opponent_id)
        prev_num = prev_active.hand_number if prev_active is not None else None

        record, new_opp_events = self.collector.ingest_observation(
            observation,
            opponent_id=self.opponent_id,
            hand_number=hand_number,
        )
        self.last_hand_number = max(self.last_hand_number, record.hand_number)

        if (
            prev_num != record.hand_number
            or self.range_model.hand_number != record.hand_number
        ):
            self.range_model.start_hand(
                hand_number=record.hand_number,
                hero_hole=observation.hole_cards,
                community_cards=observation.community_cards,
            )
            # Replay all opponent actions in the new hand onto the fresh range model
            stats_now = self.stats_snapshot()
            for ev in record.opponent_actions:
                self.range_model.update_from_action(
                    ev,
                    stats=stats_now,
                    community_cards=observation.community_cards,
                )
        else:
            self.range_model.condition_on_cards(
                hero_hole=observation.hole_cards,
                community_cards=observation.community_cards,
            )
            if new_opp_events:
                stats_now = self.stats_snapshot()
                for ev in new_opp_events:
                    self.range_model.update_from_action(
                        ev,
                        stats=stats_now,
                        community_cards=observation.community_cards,
                    )

        self._invalidate_cache()
        return record, new_opp_events

    def finalize_hand(
        self,
        *,
        hand_number: int | None = None,
        history: Sequence[ActionRecord] | None = None,
        hero_player: str | None = None,
        community_cards: Sequence[Card] | None = None,
        reached_showdown: bool = False,
        revealed_opponent_cards: tuple[Card, Card] | None = None,
        opponent_net_chips: int = 0,
        opponent_won: bool | None = None,
        final_pot: int | None = None,
    ) -> HandObservationRecord:
        """Finalize a hand and update showdown calibration if opponent cards were revealed."""
        record = self.collector.finalize_hand(
            opponent_id=self.opponent_id,
            hand_number=hand_number,
            history=history,
            hero_player=hero_player,
            community_cards=community_cards,
            reached_showdown=reached_showdown,
            revealed_opponent_cards=revealed_opponent_cards,
            opponent_net_chips=opponent_net_chips,
            opponent_won=opponent_won,
            final_pot=final_pot,
        )
        self.last_hand_number = max(self.last_hand_number, record.hand_number)

        # Ensure any final opponent actions are applied to the current hand's range model
        if self.range_model.hand_number == record.hand_number:
            if record.community_cards:
                self.range_model.condition_on_cards(
                    community_cards=record.community_cards
                )
            stats_now = self.stats_snapshot()
            for ev in record.opponent_actions:
                self.range_model.update_from_action(
                    ev,
                    stats=stats_now,
                    community_cards=record.community_cards,
                )

        if record.reached_showdown and record.revealed_hole_cards is not None:
            self.range_model.record_showdown(
                hand_number=record.hand_number,
                revealed_cards=record.revealed_hole_cards,
                community_cards=record.community_cards,
                opponent_actions=record.opponent_actions,
            )

        self._invalidate_cache()
        return record

    def stats_snapshot(self) -> OpponentStatsSnapshot:
        if self._cached_stats is None:
            hands = self.collector.list_hands(self.opponent_id)
            self._cached_stats = self.profiler.compute_snapshot(
                hands, opponent_id=self.opponent_id
            )
        return self._cached_stats

    def archetype_distribution(self) -> ArchetypeDistribution:
        if self._cached_archetype is None:
            self._cached_archetype = self.classifier.classify(self.stats_snapshot())
        return self._cached_archetype

    def range_snapshot(self) -> RangeSnapshot:
        return self.range_model.snapshot()

    def export_dict(self) -> dict[str, Any]:
        """Export full profile state (including persistent history and calibration) to a dict."""
        stats = self.stats_snapshot()
        archetype = self.archetype_distribution()
        return {
            "model_version": self.model_version,
            "opponent_id": self.opponent_id,
            "recency_decay": self.profiler.recency_decay,
            "cold_start_hands": self.classifier.cold_start_hands,
            "last_hand_number": self.last_hand_number,
            "collector": self.collector.to_dict(self.opponent_id),
            "range_model": self.range_model.to_dict(),
            "summary": {
                "stats": stats.to_dict(),
                "archetype": archetype.to_dict(),
                "range_snapshot": self.range_model.snapshot().to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OpponentProfile:
        """Restore an `OpponentProfile` from a serialized dictionary."""
        if not isinstance(payload, dict):
            raise ProfileStorageError("Profile payload must be a JSON object")
        version = payload.get("model_version")
        if version != MODEL_VERSION:
            raise ProfileStorageError(
                f"Unsupported opponent profile version {version!r} (expected {MODEL_VERSION!r})"
            )
        opp_id = str(payload.get("opponent_id", "")).strip()
        if not opp_id:
            raise ProfileStorageError("Profile payload missing non-empty 'opponent_id'")

        decay = float(payload.get("recency_decay", 0.985))
        if not 0.0 < decay <= 1.0:
            raise ProfileStorageError("Invalid recency_decay in profile payload")
        cold_start = float(payload.get("cold_start_hands", 12.0))

        profile = cls(
            opponent_id=opp_id,
            recency_decay=decay,
            cold_start_hands=cold_start,
        )
        collector_data = payload.get("collector")
        if not isinstance(collector_data, dict):
            raise ProfileStorageError("Profile payload missing 'collector' dictionary")
        profile.collector.load_dict(collector_data)

        range_data = payload.get("range_model")
        if isinstance(range_data, dict):
            profile.range_model.load_dict(range_data)

        profile.last_hand_number = int(
            payload.get(
                "last_hand_number",
                profile.collector._hand_counter_by_opponent.get(opp_id, 0),
            )
        )
        profile._invalidate_cache()
        return profile


class OpponentProfileStore:
    """Manages multiple isolated `OpponentProfile` instances with safe JSON persistence."""

    def __init__(
        self,
        storage_dir: Path | str | None = None,
        *,
        recency_decay: float = 0.94,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        self.recency_decay = recency_decay
        self._profiles: dict[str, OpponentProfile] = {}

    @staticmethod
    def _sanitize_filename(opponent_id: str) -> str:
        cleaned = "".join(
            ch if (ch.isalnum() or ch in ("-", "_")) else "_"
            for ch in opponent_id.strip()
        )
        return f"profile_{cleaned or 'opponent'}.json"

    def profile_path(self, opponent_id: str) -> Path:
        if self.storage_dir is None:
            raise ProfileStorageError(
                "No storage_dir configured on OpponentProfileStore"
            )
        return self.storage_dir / self._sanitize_filename(opponent_id)

    def get_or_create(self, opponent_id: str = "opponent") -> OpponentProfile:
        opp_id = opponent_id.strip() or "opponent"
        if opp_id not in self._profiles:
            self._profiles[opp_id] = OpponentProfile(
                opponent_id=opp_id, recency_decay=self.recency_decay
            )
        return self._profiles[opp_id]

    def list_opponents(self) -> list[str]:
        return sorted(self._profiles.keys())

    def reset_profile(self, opponent_id: str = "opponent") -> OpponentProfile:
        opp_id = opponent_id.strip() or "opponent"
        prof = self.get_or_create(opp_id)
        prof.reset()
        return prof

    def save_profile(
        self,
        opponent_id: str = "opponent",
        path: Path | str | None = None,
    ) -> Path:
        """Atomically save an opponent profile to JSON on disk."""
        prof = self.get_or_create(opponent_id)
        target = Path(path) if path is not None else self.profile_path(prof.opponent_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(prof.export_dict(), indent=2, sort_keys=True)

        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=".tmp_harvey_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
            os.replace(tmp_name, target)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        return target

    def load_profile(
        self,
        opponent_id: str = "opponent",
        path: Path | str | None = None,
        *,
        strict: bool = False,
    ) -> tuple[OpponentProfile, str]:
        """Load an opponent profile from disk.

        When `strict=False` (default), missing or corrupted files are handled
        safely by returning a fresh `OpponentProfile` along with a status code
        (`"loaded"`, `"missing_created_new"`, or `"corrupt_recovered_new"`).
        When `strict=True`, raises `FileNotFoundError` or `ProfileStorageError`.
        """
        opp_id = opponent_id.strip() or "opponent"
        target = Path(path) if path is not None else self.profile_path(opp_id)

        if not target.exists():
            if strict:
                raise FileNotFoundError(f"Opponent profile file not found: {target}")
            fresh = OpponentProfile(
                opponent_id=opp_id, recency_decay=self.recency_decay
            )
            self._profiles[opp_id] = fresh
            return fresh, "missing_created_new"

        try:
            raw_text = target.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
            loaded = OpponentProfile.from_dict(payload)
            self._profiles[loaded.opponent_id] = loaded
            return loaded, "loaded"
        except (
            OSError,
            json.JSONDecodeError,
            ProfileStorageError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            if strict:
                raise ProfileStorageError(
                    f"Corrupted opponent profile file {target}: {exc}"
                ) from exc
            recovered = OpponentProfile(
                opponent_id=opp_id, recency_decay=self.recency_decay
            )
            self._profiles[opp_id] = recovered
            return recovered, "corrupt_recovered_new"

    def export_profile(self, opponent_id: str = "opponent") -> dict[str, Any]:
        return self.get_or_create(opponent_id).export_dict()

    def import_profile(self, payload: dict[str, Any] | str) -> OpponentProfile:
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ProfileStorageError(f"Invalid JSON string: {exc}") from exc
        else:
            parsed = payload
        prof = OpponentProfile.from_dict(parsed)
        self._profiles[prof.opponent_id] = prof
        return prof
