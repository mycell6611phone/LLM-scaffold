from src.core.llm import OpenAICompat
from src.core.scratchpad import Scratchpad
from src.core.tools import Toolbelt
from src.core.config import Config
from src.core.task_graph import Step
from .prompts import THEO_SYS
from .util import parse_action

class Theorist:
    def __init__(self, llm: OpenAICompat, tools: Toolbelt, sp: Scratchpad, cfg: Config):
        self.llm, self.tools, self.sp, self.cfg = llm, tools, sp, cfg

    async def run_step(self, step: Step, budget_calls: int = 4):
        payload = {
            "objective": step.inputs.get("objective"),
            "requirements": step.inputs.get("requirements"),
            "context": step.inputs.get("context"),
        }
        history = [
            {"role": "system", "content": THEO_SYS},
            {
                "role": "user",
                "content": (
                    f"Question: {step.description}\n"
                    f"Inputs: {payload}\n"
                    "Respond with a short synthesis and cite key unknowns."
                ),
            },
        ]
        calls = 0
        result = ""
        while calls < budget_calls:
            out = await self.llm.chat(history, temperature=0.5, max_tokens=self.cfg.max_step_tokens, model=self.cfg.model_theorist or self.cfg.model)
            act = parse_action(out)
            if act and act.get("action") == "tool":
                tool_res = self.tools.dispatch(act)
                self.sp.append({"type":"tool","call":act,"result":tool_res})
                history.append({"role":"assistant","content": out})
                history.append({"role":"user","content": f"TOOL_RESULT: {tool_res}"})
                calls += 1
                continue
            result = out if not act else act.get("content","")
            break
        self.sp.append({"type": "result", "step": step.description, "result": result})
        return result

    async def review(self, step: Step, output: str):
        return ""
