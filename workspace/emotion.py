'''
emotion.py

Purpose
 Scaffold for optional mood/affect modeling to modulate prompts and routing.
 No business logic. Methods raise NotImplementedError.

Integration
 - core_loop may query render_system_hint() to bias personas/tools
 - mem/debate/trainer may record mood in metadata for audits
'''

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime

try:  # soft import to avoid tight coupling in scaffold stage
    from agent_personas import PersonaSpec  # type: ignore
except Exception:  # pragma: no cover
    PersonaSpec = Any  # fallback type

__all__ = [
    "MoodName",
    "EmotionState",
    "EmotionEngine",
]


class MoodName(Enum):
    NEUTRAL = "NEUTRAL"
    CURIOUS = "CURIOUS"
    FOCUSED = "FOCUSED"
    CAUTIOUS = "CAUTIOUS"
    CREATIVE = "CREATIVE"
    SKEPTICAL = "SKEPTICAL"
    TIRED = "TIRED"
    ENERGETIC = "ENERGETIC"


@dataclass
class EmotionState:
    mood: MoodName = MoodName.NEUTRAL
    intensity: float = 0.5  # 0..1
    tags: List[str] = field(default_factory=list)
    biases: Dict[str, float] = field(default_factory=dict)  # e.g., {"risk": -0.2}
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class EmotionEngine:
    """
    Scaffold API for mood inference and prompt biasing.

    Notes:
     - Keep deterministic mapping rules when implemented.
     - No time-based decay or stochasticity in this scaffold.
    """

    def __init__(self):
        """Initializes the EmotionEngine with a default state."""
        self._state = EmotionState()
        self.namespace = "default"
        self.policy: Dict[str, Any] = {}
        # Do not compute or persist in scaffold.

    def get_state(self) -> EmotionState:
        """Return current EmotionState.
        """
        # In a real implementation, this might be a copy
        return self._state

    def set_state(self, state: EmotionState) -> EmotionState:
        """Replace current state.
        """
        if not isinstance(state, EmotionState):
            raise TypeError("state must be an instance of EmotionState")
        self._state = state
        return self._state

    def set_mood(self, mood: MoodName, *, intensity: float | None = None, tags: Optional[List[str]] = None) -> EmotionState:
        """Convenience to set mood and optional intensity/tags.
        """
        self._state.mood = mood
        if intensity is not None:
            self._state.intensity = max(0.0, min(1.0, intensity))
        if tags is not None:
            self._state.tags = tags
        self._state.updated_at = datetime.utcnow()
        return self._state

    def sense(self, signals: Dict[str, Any]) -> EmotionState:
        """Update state from external signals (latency, errors, success rate, user tone).

        Raises:
            NotImplementedError: Always in scaffold.
        """
        raise NotImplementedError("EmotionEngine.sense is a scaffold.")

    def render_system_hint(self, *, role: Optional[PersonaSpec] = None) -> str:
        """Return a short, deterministic system hint to condition a persona.

        Example: "You are focused and risk-averse today. Prefer concise outputs."

        Raises:
            NotImplementedError: Always in scaffold.
        """
        raise NotImplementedError("EmotionEngine.render_system_hint is a scaffold.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current state for logging or memory.
        """
        state_dict = {
            "mood": self._state.mood.value,
            "intensity": self._state.intensity,
            "tags": self._state.tags,
            "biases": self._state.biases,
            "metadata": self._state.metadata,
            "updated_at": self._state.updated_at.isoformat()
        }
        return state_dict

    def from_dict(self, data: Dict[str, Any]) -> EmotionState:
        """Load state from a dict representation.
        """
        self._state = EmotionState(
            mood=MoodName(data["mood"]),
            intensity=data.get("intensity", 0.5),
            tags=data.get("tags", []),
            biases=data.get("biases", {}),
            metadata=data.get("metadata", {}),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        return self._state
