from src.core.llm import OpenAICompat
from src.core.scratchpad import Scratchpad
from src.core.tools import Toolbelt
from src.core.config import Config
from src.core.task_graph import Step
from .prompts import CRITIC_SYS

class Critic:
    def __init__(self, llm: OpenAICompat, tools: Toolbelt, sp: Scratchpad, cfg: Config):
        self.llm, self.tools, self.sp, self.cfg = llm, tools, sp, cfg

    async def review(self, step: Step, output: str) -> str:
        msg = [
            {"role":"system","content": CRITIC_SYS},
            {"role":"user","content": f"Step: {step.description}\nOutput: {output}\nRecent: {self.sp.short_context(4)}"}
        ]
        return await self.llm.chat(msg, temperature=0.1, max_tokens=400, model=self.cfg.model_critic or self.cfg.model)
