# plan_orchestrator.py
"""
Drop this module into your project and call `run_plan(raw_plan)` from your existing entrypoint.
"""
import json
from typing import Any, Dict, List
from dataclasses import dataclass, field
import openai

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from config import OPENAI_API_KEY, LLM_MODEL
openai.api_key = OPENAI_API_KEY

# -----------------------------------------------------------------------------
@dataclass
class Step:
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False

# -----------------------------------------------------------------------------
class PlanBreaker:
    """
    Break a nested plan JSON into ordered steps via LLM.
    """
    def __init__(self, model: str = LLM_MODEL):
        self.model = model

    def split(self, raw_plan: Dict[str, Any]) -> List[Step]:
        prompt = (
            "You are an AI assistant. «Given the following project plan in JSON, "
            "break it down into an ordered list of steps. «Return valid JSON array: "
            "[{ \"description\": str, ... }].»\n" + json.dumps(raw_plan)
        )
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        arr = json.loads(resp.choices[0].message.content)
        return [Step(description=item["description"], data=item) for item in arr]

# -----------------------------------------------------------------------------
class StepOrchestrator:
    """
    Execute each step one at a time, passing accumulated context.
    """
    def __init__(self, model: str = LLM_MODEL):
        self.model = model

    def execute(self, step: Step, context: Dict[str, Any]) -> Any:
        prompt = (
            "You are a helpful AI. Execute the following step given context.\n"
            f"Step: {step.description}\nContext: {json.dumps(context, indent=2)}\n"
            "Return only JSON representing the step output."
        )
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            return json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            return {"result": resp.choices[0].message.content}

# -----------------------------------------------------------------------------
class Orchestrator:
    """
    High-level plan orchestration: break, execute, aggregate.

    Usage:
        from plan_orchestrator import run_plan
        results = run_plan(raw_plan_dict)
    """
    def __init__(self, model: str = LLM_MODEL):
        self.breaker = PlanBreaker(model)
        self.runner = StepOrchestrator(model)

    def run(self, raw_plan: Dict[str, Any]) -> List[Any]:
        steps = self.breaker.split(raw_plan)
        context: Dict[str, Any] = {}
        results: List[Any] = []
        for step in steps:
            out = self.runner.execute(step, context)
            results.append(out)
            context[step.description] = out
            step.completed = True
        return results

# -----------------------------------------------------------------------------
def run_plan(raw_plan: Dict[str, Any]) -> List[Any]:
    """Convenience entrypoint."""
    orch = Orchestrator()
    return orch.run(raw_plan)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from util import load_json, save_json

    if len(sys.argv) < 2:
        print("Usage: python plan_orchestrator.py path/to/plan.json")
        sys.exit(1)

    plan_path = sys.argv[1]
    plan = load_json(plan_path)
    outputs = run_plan(plan)
    save_json("plan_results.json", outputs)
    print("Execution complete. Results written to plan_results.json.")
