from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Step(BaseModel):
    description: str
    agent: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    steps: List[Step]


class StepResult(BaseModel):
    """Container for the normalized output of a single step."""

    agent: str
    step_index: int
    step_description: str
    summary: str
    scratch_path: Optional[str] = None
    raw_output: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Backwards compatibility alias for callers that prefer AgentResult nomenclature.
AgentResult = StepResult
