from typing import List
from src.core.llm import OpenAICompat
from src.core.scratchpad import Scratchpad
from src.core.tools import Toolbelt
from src.core.memory import Memory
from src.core.config import Config
from src.core.task_graph import Step, Plan
from .prompts import ORCH_SYS

class Orchestrator:
    def __init__(self, llm: OpenAICompat, tools: Toolbelt, sp: Scratchpad, mem: Memory, cfg: Config):
        self.llm, self.tools, self.sp, self.mem, self.cfg = llm, tools, sp, mem, cfg

    async def make_plan(self, prompt: str) -> Plan:
        mem_hits = self.mem.query(prompt, k=3)
        msg = [
            {"role":"system","content": ORCH_SYS},
            {"role":"user","content": f"Objective: {prompt}\nRelevant memory: {mem_hits}"}
        ]
        out = await self.llm.chat(msg, temperature=0.5, max_tokens=512, model=self.cfg.model_orchestrator or self.cfg.model_theorist or self.cfg.model)
        data = self.llm.extract_json_block(out) or {"steps":[{"description":"Understand objective", "agent":"theorist"}]}
        steps = [Step(**s) for s in data.get("steps", [])]
        return Plan(steps=steps)

    def choose_agent(self, step: Step) -> str:
        text = step.description.lower()
        if any(k in text for k in ["list","read","write","refactor","implement","run","execute","test"]):
            return "executor"
        if any(k in text for k in ["analyz","hypoth","design","plan","refactor plan","strategy"]):
            return "theorist"
        if any(k in text for k in ["review","crit","check","verify"]):
            return "critic"
        return "executor"
