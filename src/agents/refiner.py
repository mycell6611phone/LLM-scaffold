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


    async def run_step(self, step: Step, budget_calls: int = 1):
        prompt = step.inputs.get("objective", step.description)
        plan_payload = step.inputs.get("plan_snapshot") or {}
        if isinstance(plan_payload, Plan):
            plan_obj = plan_payload
        else:
            plan_obj = Plan(**plan_payload) if plan_payload else Plan(steps=[])
        timeline = step.inputs.get("context", [])
        result = await self.synthesize(prompt, plan_obj, timeline)
        self.sp.append({"type": "result", "step": step.description, "result": result})
        return result

