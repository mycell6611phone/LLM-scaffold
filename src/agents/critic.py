from src.core.llm import OpenAICompat
from src.core.scratchpad import Scratchpad
from src.core.tools import Toolbelt
from src.core.config import Config
from src.core.task_graph import Step
from src.core.memory import Memory
from .prompts import CRITIC_SYS

class Critic:
    def __init__(self, llm: OpenAICompat, tools: Toolbelt, sp: Scratchpad, cfg: Config, mem: Memory):
        self.llm, self.tools, self.sp, self.cfg, self.mem = llm, tools, sp, cfg, mem

    async def review(self, step: Step, output: str) -> dict:
        """Return a structured critique for the provided step output."""
        context_tail = self.sp.short_context(3)
        prompt = (
            "Review the provided work product and respond as JSON with fields "
            "verdict (pass|fail), explanation, and evidence (optional)."
        )
        msg = [
            {"role": "system", "content": CRITIC_SYS},
            {
                "role": "user",
                "content": (
                    f"Instruction: {prompt}\n"
                    f"Step: {step.description}\n"
                    f"Candidate output: {output}\n"
                    f"Recent notes: {context_tail}"
                ),
            },
        ]

        raw = await self.llm.chat(
            msg,
            temperature=0.1,
            max_tokens=400,
            model=self.cfg.model_critic or self.cfg.model,
        )
        parsed = self.llm.extract_json_block(raw) or {}
        verdict = str(parsed.get("verdict", "pass")).lower()
        if verdict not in {"pass", "fail"}:
            verdict = "pass"
        explanation = parsed.get("explanation") or raw
        evidence = parsed.get("evidence")
        summary = parsed.get("summary")
        if not summary:
            snippet = explanation.strip().splitlines()[0][:180]
            summary = f"[{verdict.upper()}] {snippet}"

        await self.mem.store_critique(step.description, output, verdict, explanation)

        critique = {
            "verdict": verdict,
            "explanation": explanation,
            "evidence": evidence,
            "summary": summary,
        }
        return critique
