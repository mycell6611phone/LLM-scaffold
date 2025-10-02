# goal_manager.py
"""
Tracks and manages agent goals for the AGI loop.
In-memory implementation with thread safety and soft delete.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, List, Optional, Dict
from datetime import datetime
import threading
import uuid

__all__ = ["GoalStatus", "Goal", "GoalManager"]


class GoalStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


@dataclass
class Goal:
    id: str
    title: str
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class GoalManager:
    """
    In-memory manager for agent goals.

    - Thread-safe.
    - Deterministic list order: (priority asc, None last) then created_at asc then id asc.
    - Soft delete by default (moves to internal trash and marks metadata).
    """

    def __init__(self, *, namespace: str = "default", storage_backend: Any | None = None) -> None:
        self.namespace = namespace
        self.storage_backend = storage_backend  # placeholder, unused
        self._lock = threading.RLock()
        self._goals: Dict[str, Goal] = {}
        self._trash: Dict[str, Goal] = {}

    # ---------- helpers ----------

    @staticmethod
    def _now() -> datetime:
        return datetime.utcnow()

    @staticmethod
    def _normalize_tags(tags: Iterable[str] | None) -> List[str]:
        if not tags:
            return []
        seen = set()
        out: List[str] = []
        for t in tags:
            if t is None:
                continue
            s = str(t).strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _sort_key(g: Goal):
        prio_key = (1, None) if g.priority is None else (0, g.priority)
        return prio_key, g.created_at, g.id

    def _require(self, goal_id: str) -> Goal:
        g = self._goals.get(goal_id)
        if g is None:
            raise KeyError(f"goal '{goal_id}' not found")
        return g

    # ---------- API ----------

    def add_goal(
        self,
        title: str,
        description: str | None = None,
        *,
        priority: int | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Goal:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")

        now = self._now()
        gid = self._new_id()
        goal = Goal(
            id=gid,
            title=title.strip(),
            description=(description or "").strip(),
            status=GoalStatus.PENDING,
            priority=priority,
            tags=self._normalize_tags(tags),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            if gid in self._goals or gid in self._trash:
                # Extremely unlikely, but keep strict.
                raise RuntimeError("generated id collision")
            self._goals[gid] = goal
        return goal

    def update_goal(
        self,
        goal_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: GoalStatus | None = None,
        priority: int | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Goal:
        with self._lock:
            g = self._require(goal_id)

            new_tags = g.tags if tags is None else self._normalize_tags(tags)
            new_title = g.title if title is None else title.strip()
            if new_title == "":
                raise ValueError("title cannot be empty")

            new_desc = g.description if description is None else (description or "").strip()
            new_status = g.status if status is None else status
            new_priority = g.priority if priority is None else priority
            new_metadata = g.metadata if metadata is None else dict(metadata)

            updated = replace(
                g,
                title=new_title,
                description=new_desc,
                status=new_status,
                priority=new_priority,
                tags=new_tags,
                metadata=new_metadata,
                updated_at=self._now(),
            )
            self._goals[goal_id] = updated
            return updated

    def complete_goal(self, goal_id: str) -> Goal:
        with self._lock:
            g = self._require(goal_id)
            if g.status is GoalStatus.COMPLETED:
                return g
            updated = replace(g, status=GoalStatus.COMPLETED, updated_at=self._now())
            self._goals[goal_id] = updated
            return updated

    def list_goals(
        self,
        *,
        status: GoalStatus | None = None,
        tag: str | None = None,
        limit: int | None = None,
    ) -> List[Goal]:
        with self._lock:
            items = list(self._goals.values())

        if status is not None:
            items = [g for g in items if g.status is status]

        if tag is not None:
            t = str(tag).strip()
            if t:
                items = [g for g in items if t in g.tags]

        items.sort(key=self._sort_key)
        if limit is not None and limit >= 0:
            items = items[:limit]
        return items

    def remove_goal(self, goal_id: str, *, hard_delete: bool = False) -> None:
        with self._lock:
            g = self._goals.pop(goal_id, None)
            if g is None:
                # If already soft-deleted, allow idempotent remove.
                if hard_delete:
                    self._trash.pop(goal_id, None)
                else:
                    # Nothing to do.
                    return
            else:
                if hard_delete:
                    # Drop on the floor.
                    return
                tombstoned = replace(
                    g,
                    status=GoalStatus.CANCELLED if g.status is not GoalStatus.COMPLETED else g.status,
                    metadata={**g.metadata, "_deleted": True, "_deleted_at": self._now().isoformat()},
                    updated_at=self._now(),
                )
                self._trash[goal_id] = tombstoned

