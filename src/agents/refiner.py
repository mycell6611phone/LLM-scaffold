from typing import List, Dict, Any
from src.core.llm import OpenAICompat
from src.core.scratchpad import Scratchpad
from src.core.tools import Toolbelt
from src.core.config import Config
from src.core.task_graph import Plan, Step
from .prompts import REFINER_SYS

class Refiner:
    def __init__(self, llm: OpenAICompat, tools: Toolbelt, sp: Scratchpad, cfg: Config):
        self.llm, self.tools, self.sp, self.cfg = llm, tools, sp, cfg

    async def synthesize(self, prompt: str, plan: Plan, timeline: List[Dict[str, Any]]) -> str:
        msg = [
            {"role": "system", "content": REFINER_SYS},
            {
                "role": "user",
                "content": (
                    f"Objective: {prompt}\n"
                    f"Plan: {plan.model_dump()}\n"
                    f"Timeline summary: {timeline}"
                ),
            },
        ]
        return await self.llm.chat(
            msg,
            temperature=0.2,
            max_tokens=1200,
            model=self.cfg.model_refiner or self.cfg.model,
        )

    async def run_step(
        self,
        step: Step,
        plan: Plan = None,
        context: List[Dict[str, Any]] = None,
        budget_calls: int = 1,
    ):
        # Objective comes from the step unless overridden
        prompt = step.inputs.get("objective", step.description)

        # Plan: prefer explicit arg, fall back to step.inputs
        plan_payload = plan or step.inputs.get("plan_snapshot")
        if isinstance(plan_payload, Plan):
            plan_obj = plan_payload
        elif isinstance(plan_payload, dict):
            plan_obj = Plan(**plan_payload)
        else:
            plan_obj = Plan(steps=[])

        # Context: prefer explicit arg, fall back to step.inputs
        timeline = context or step.inputs.get("context", [])

        # Generate final synthesis
        result = await self.synthesize(prompt, plan_obj, timeline)

        # Log and return
        self.sp.append({"type": "result", "step": step.description, "result": result})
        return result

