from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Step(BaseModel):
    description: str
    agent: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    steps: List[Step]
