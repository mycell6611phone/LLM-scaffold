from typing import Dict, Any
from src.core.llm import OpenAICompat
from src.core.scratchpad import Scratchpad
from src.core.tools import Toolbelt
from src.core.config import Config
from src.core.task_graph import Step
from .prompts import EXEC_SYS
from .util import parse_action

class Executor:
    def __init__(self, llm: OpenAICompat, tools: Toolbelt, sp: Scratchpad, cfg: Config):
        self.llm, self.tools, self.sp, self.cfg = llm, tools, sp, cfg

    async def run_step(self, step: Step, budget_calls: int = 4):
        context_payload = {
            "objective": step.inputs.get("objective"),
            "requirements": step.inputs.get("requirements"),
            "dependencies": step.inputs.get("dependencies"),
            "artifacts": step.inputs.get("artifacts"),
            "context": step.inputs.get("context"),
        }
        calls = 0
        history = [
            {"role": "system", "content": EXEC_SYS},
            {
                "role": "user",
                "content": (
                    f"Task: {step.description}\n"
                    f"Inputs: {context_payload}\n"
                    "Respond with concise actions. Use FINAL when satisfied."
                ),
            },
        ]
        result = ""
        while calls < budget_calls:
            out = await self.llm.chat(history, temperature=0.5, max_tokens=self.cfg.max_step_tokens, model=self.cfg.model_executor or self.cfg.model)
            act = parse_action(out)
            if not act:
                result = out
                break
            if act.get("action") == "final":
                result = act.get("content","")
                break
            if act.get("action") == "tool":
                tool_res = self.tools.dispatch(act)
                self.sp.append({"type":"tool", "call": act, "result": tool_res})
                history.append({"role":"assistant","content": out})
                history.append({"role":"user","content": f"TOOL_RESULT: {tool_res}"})
                calls += 1
                continue
            result = str(out)
            break
        self.sp.append({"type": "result", "step": step.description, "result": result})
        return result
