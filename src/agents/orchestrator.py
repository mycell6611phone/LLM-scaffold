# orchestrator.py
from typing import List, Dict, Any
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

    async def run_loop(self, user_objective: str, agents: Dict[str, Any], max_iters: int = 10):
        """
        Main orchestration loop. Keeps asking 'what's needed next?' until done or max_iters hit.
        """
        context: List[Dict[str, Any]] = []
        completed_steps: List[Step] = []

        for i in range(max_iters):
            # Ask LLM what to do NEXT, not the whole plan
            msg = [
                {"role": "system", "content": ORCH_SYS},
                {"role": "user", "content": f"Objective: {user_objective}\nContext so far: {context}\n\nDecide the NEXT action only."}
            ]
            out = await self.llm.chat(
                msg,
                temperature=0.7,
                max_tokens=300,
                model=self.cfg.model_orchestrator or self.cfg.model
            )

            data = self.llm.extract_json_block(out) or {"steps":[{"description":"Understand objective","agent":"theorist"}]}
            step = Step(**data["steps"][0])  # only expect ONE step here

            agent_key = step.agent or self.choose_agent(step)
            agent = agents[agent_key]

            # Execute step
            if hasattr(agent, "run_step"):
                result = await agent.run_step(step, budget_calls=self.cfg.max_tool_calls_per_step)
            else:
                # critic or theorist might not use run_step
                result = await getattr(agent, "review", lambda s,r: str(r))(step, context)

            # Record context for next cycle
            context.append({"description": step.description, "agent": agent_key, "result": result})
            completed_steps.append(step)

            # Append to scratchpad
            self.sp.append({"type": "step_result", "idx": i+1, "agent": agent_key, "result": result})

            # Stop if final output signaled
            if "final" in step.description.lower() or agent_key == "refiner":
                break

        return Plan(steps=completed_steps)

    def choose_agent(self, step: Step) -> str:
        """Heuristic fallback for agent assignment."""
        text = step.description.lower()
        if any(k in text for k in ["list","read","write","refactor","implement","run","execute","test"]):
            return "executor"
        if any(k in text for k in ["analyz","hypoth","design","plan","refactor plan","strategy"]):
            return "theorist"
        if any(k in text for k in ["review","crit","check","verify"]):
            return "critic"
        return "executor"

