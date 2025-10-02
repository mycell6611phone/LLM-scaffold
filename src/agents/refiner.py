from src.core.llm import OpenAICompat
from src.core.scratchpad import Scratchpad
from src.core.tools import Toolbelt
from src.core.config import Config
from src.core.task_graph import Plan
from .prompts import REFINER_SYS

class Refiner:
    def __init__(self, llm: OpenAICompat, tools: Toolbelt, sp: Scratchpad, cfg: Config):
        self.llm, self.tools, self.sp, self.cfg = llm, tools, sp, cfg

    async def synthesize(self, prompt: str, plan: Plan, sp: Scratchpad) -> str:
        msg = [
            {"role":"system","content": REFINER_SYS},
            {"role":"user","content": f"Objective: {prompt}\nPlan: {plan.model_dump()}\nTrace: {sp.short_context(20)}"}
        ]
        return await self.llm.chat(msg, temperature=0.2, max_tokens=1200, model=self.cfg.model_refiner or self.cfg.model)

    async def run_step(self, *args, **kwargs):
        return ""
