# memory_debate.py
"""
Purpose
 Facade for the debate-based memory filter.
 Re-exports concrete classes from memeryloop.py when available.
 Falls back to light scaffolds so imports remain stable during bootstrap.

Exports
 - CandidateMemory
 - ModelInterface
 - DebateConsensusEngine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Dict
import re
import time

__all__ = [
    "CandidateMemory",
    "ModelInterface",
    "DebateConsensusEngine",
]

# --- Prefer the real implementation if present ---
try:  # pragma: no cover
    from memeryloop import CandidateMemory as _CandidateMemory  # type: ignore
    from memeryloop import ModelInterface as _ModelInterface  # type: ignore
    from memeryloop import DebateConsensusEngine as _DebateConsensusEngine  # type: ignore

    CandidateMemory = _CandidateMemory
    ModelInterface = _ModelInterface
    DebateConsensusEngine = _DebateConsensusEngine

except Exception:
    # --- Minimal but working scaffolds with debate logic ---
    Decision = str  # "ACCEPT" | "REJECT" | "ERROR"

    @dataclass
    class CandidateMemory:
        id: str
        content: str
        status: str = "PENDING"
        debate_log: List[Dict[str, str]] = field(default_factory=list)
        meta: Dict[str, str] = field(default_factory=dict)

    def add_to_training_set(cand: CandidateMemory) -> None:
        # Replace with your sink
        cand.meta["added_to_training_set"] = "true"

    class ModelInterface:
        def __init__(
            self,
            name: str,
            ollama_model_name: str | None = None,
            temperature: float = 0.5,
        ) -> None:
            self.name = name
            self.model_name = ollama_model_name or "local"
            self.temperature = temperature

            self._bad_markers = re.compile(r"\b(todo|tbd|fixme|draft|lorem ipsum)\b", re.I)
            self._secret_markers = re.compile(r"\b(api[_-]?key|password|secret|token)\b", re.I)
            self._pii_markers = re.compile(r"\b(ssn|credit\s*card|private\s*key)\b", re.I)

        def _heuristic(
            self,
            candidate: CandidateMemory,
            opponent_reasoning: Optional[str],
        ) -> Tuple[Decision, str]:
            text = candidate.content.strip()

            if not text:
                return "REJECT", f"{self.name}: empty"
            if len(text) < 24:
                return "REJECT", f"{self.name}: too short"
            if self._bad_markers.search(text):
                return "REJECT", f"{self.name}: draft markers"
            if self._pii_markers.search(text):
                return "REJECT", f"{self.name}: PII markers"
            if self._secret_markers.search(text):
                return "REJECT", f"{self.name}: secrets detected"

            score = 0.0
            score += min(len(text) / 200.0, 1.0)           # informativeness
            if any(k in text.lower() for k in ("because", "therefore", "so that")):
                score += 0.2                                # causal structure
            if ":" in text or "-" in text:
                score += 0.1                                # formatting hint

            if opponent_reasoning:
                if re.search(r"\b(error|wrong|unsafe|contradiction)\b", opponent_reasoning, re.I):
                    score -= 0.35                           # opponent flagged an issue

            decision = "ACCEPT" if score >= 0.9 else "REJECT"
            return decision, f"{self.name}: score={score:.2f}"

        def evaluate(
            self,
            candidate: CandidateMemory,
            opponent_reasoning: str | None = None,
        ) -> Tuple[str, str]:
            try:
                return self._heuristic(candidate, opponent_reasoning)
            except Exception as e:
                return "ERROR", f"{self.name}: {type(e).__name__}"

    class DebateConsensusEngine:
        def __init__(
            self,
            model_a: ModelInterface,
            model_b: ModelInterface,
            max_rounds: int = 3,
        ) -> None:
            self.model_a = model_a
            self.model_b = model_b
            self.max_rounds = max_rounds

        def debate_candidate(self, candidate: CandidateMemory) -> str:
            print(f"\n--- Debate for Candidate {candidate.id} ---")
            just_a = ""
            just_b = ""

            for r in range(1, self.max_rounds + 1):
                if r == 1:
                    dec_a, just_a = self.model_a.evaluate(candidate)
                    dec_b, just_b = self.model_b.evaluate(candidate)
                else:
                    dec_a, just_a = self.model_a.evaluate(candidate, just_b)
                    dec_b, just_b = self.model_b.evaluate(candidate, just_a)

                candidate.debate_log.append(
                    {
                        "round": str(r),
                        "model_a_decision": dec_a,
                        "model_a_justification": just_a,
                        "model_b_decision": dec_b,
                        "model_b_justification": just_b,
                    }
                )

                print(f"Round {r}: A={dec_a}, B={dec_b}")

                if dec_a == dec_b and dec_a in {"ACCEPT", "REJECT"}:
                    candidate.status = "ACCEPTED" if dec_a == "ACCEPT" else "REJECTED"
                    print(f"*** Consensus: {candidate.status} ***")
                    return candidate.status

                if "ERROR" in {dec_a, dec_b}:
                    candidate.status = "ERROR_DEBATE"
                    return candidate.status

            candidate.status = "RETAINED"
            print("*** No consensus, candidate retained. ***")
            return candidate.status

        def run(self, candidate_pool: List[CandidateMemory]) -> None:
            for cand in candidate_pool:
                status = self.debate_candidate(cand)
                if status == "ACCEPTED":
                    add_to_training_set(cand)
                    time.sleep(10)  # polite delay

