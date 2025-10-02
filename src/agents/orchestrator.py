# orchestrator.py
import json
import re
from typing import Any, Dict, List, Optional

from src.core.llm import OpenAICompat
from src.core.scratchpad import Scratchpad
from src.core.tools import Toolbelt
from src.core.memory import Memory
from src.core.config import Config
from src.core.task_graph import Step, Plan, StepResult
from .prompts import ORCH_SYS


class Orchestrator:
    def __init__(
        self,
        llm: OpenAICompat,
        tools: Toolbelt,
        sp: Scratchpad,
        mem: Memory,
        cfg: Config,
    ):
        self.llm, self.tools, self.sp, self.mem, self.cfg = llm, tools, sp, mem, cfg
        self.current_plan: Optional[Plan] = None
        self._discoveries: List[Dict[str, Any]] = []
        self._objective: Optional[str] = None
        self.last_context: List[Dict[str, Any]] = []

    async def gather_requirements(self, objective: str, agents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consult supporting agents for quick requirement probes."""

        self._objective = objective
        context_hint = self.sp.short_context(2)
        agent_names = list(agents.keys())
        prompt = (
            "Identify which specialist agents should provide requirement probes before planning. "
            "Respond ONLY as JSON with `queries`, each entry containing `agent` and `question`."
        )
        msg = [
            {"role": "system", "content": ORCH_SYS},
            {
                "role": "user",
                "content": (
                    f"Objective: {objective}\n"
                    f"Agents: {agent_names}\n"
                    f"Context hint: {context_hint}\n"
                    f"Instruction: {prompt}"
                ),
            },
        ]

        raw = await self.llm.chat(
            msg,
            temperature=0.2,
            max_tokens=300,
            model=self.cfg.model_plan_orchestrator
            or self.cfg.model_orchestrator
            or self.cfg.model,
        )
        data = self.llm.extract_json_block(raw) or {}
        queries = data.get("queries") or []
        if not queries:
            queries = [
                {
                    "agent": "theorist",
                    "question": f"List the primary requirements, constraints, and unknowns for: {objective}",
                }
            ]

        discoveries: List[Dict[str, Any]] = []
        for idx, query in enumerate(queries):
            agent_name = query.get("agent")
            question = query.get("question") or f"Outline requirements for: {objective}"
            if (
                not agent_name
                or agent_name not in agents
                or agent_name in {"critic", "refiner"}
            ):
                continue

            agent = agents[agent_name]
            probe_step = Step(
                description=question,
                agent=agent_name,
                inputs={
                    "objective": objective,
                    "context": [],
                    "mode": "requirement_probe",
                },
            )

            if hasattr(agent, "run_step"):
                raw_result = await agent.run_step(probe_step, budget_calls=1)
            else:
                continue

            step_result = self._normalize_result(
                agent_name,
                agent,
                probe_step,
                raw_result,
                step_index=-(idx + 1),
            )
            discoveries.append(
                {
                    "agent": agent_name,
                    "question": question,
                    "insight": step_result.summary,
                    "scratch": step_result.scratch_path,
                    "raw": step_result.raw_output,
                }
            )
            self.sp.append(
                {
                    "type": "requirement_summary",
                    "agent": agent_name,
                    "summary": step_result.summary,
                }
            )

        self._discoveries = discoveries
        return discoveries

    async def build_plan(self, objective: str, discoveries: List[Dict[str, Any]]) -> Plan:
        """Request a structured plan from the LLM using requirement summaries."""

        requirements = [f"{d['agent']}: {d['insight']}" for d in discoveries]
        msg = [
            {"role": "system", "content": ORCH_SYS},
            {
                "role": "user",
                "content": (
                    "Return a JSON array of plan steps. Each step requires a `description` and optional "
                    "`agent`, `inputs`, or `outputs`. Keep steps atomic and execution-ready.\n"
                    f"Objective: {objective}\n"
                    f"Known requirements: {requirements}"
                ),
            },
        ]

        raw = await self.llm.chat(
            msg,
            temperature=0.2,
            max_tokens=500,
            model=self.cfg.model_plan_orchestrator
            or self.cfg.model_orchestrator
            or self.cfg.model,
        )
        parsed = self.llm.extract_json_block(raw) or []
        if isinstance(parsed, dict) and "steps" in parsed:
            parsed = parsed["steps"]
        if not isinstance(parsed, list) or not parsed:
            parsed = [
                {
                    "description": f"Analyze the objective: {objective}",
                    "agent": "theorist",
                }
            ]

        steps: List[Step] = []
        for item in parsed:
            if isinstance(item, str):
                steps.append(Step(description=item))
                continue
            description = item.get("description") or json.dumps(item)
            agent = item.get("agent")
            inputs = item.get("inputs") or {}
            outputs = item.get("outputs") or {}
            steps.append(Step(description=description, agent=agent, inputs=inputs, outputs=outputs))

        plan = Plan(steps=steps)
        snapshot = [
            {
                "agent": step.agent,
                "description": self._truncate(step.description, 120),
            }
            for step in plan.steps
        ]
        self.sp.append({"type": "plan_snapshot", "steps": snapshot})
        self.current_plan = plan
        return plan

    def refine_steps(self, plan: Plan) -> Plan:
        """Split oversized steps and populate structured inputs."""

        refined: List[Step] = []
        for step in plan.steps:
            for segment in self._segment_description(step.description):
                agent = step.agent or self.choose_agent(Step(description=segment))
                inputs = self._prepare_inputs(segment, agent, step.inputs)
                refined.append(
                    Step(
                        description=segment,
                        agent=agent,
                        inputs=inputs,
                        outputs=step.outputs.copy(),
                    )
                )

        refined_plan = Plan(steps=refined)
        self.current_plan = refined_plan
        return refined_plan

    async def run_loop(
        self,
        user_objective: str,
        agents: Dict[str, Any],
        max_iters: int = 10,
    ) -> Plan:
        """Execute a pre-built plan, capturing summaries along the way."""

        self._objective = user_objective
        context: List[Dict[str, Any]] = []
        executed_steps: List[Step] = []

        discoveries = await self.gather_requirements(user_objective, agents)
        plan = await self.build_plan(user_objective, discoveries)
        plan = self.refine_steps(plan)
        plan_payload = plan.model_dump()

        for idx, base_step in enumerate(plan.steps):
            if idx >= max_iters:
                break

            step = base_step.model_copy(deep=True)
            agent_key = step.agent or self.choose_agent(step)
            if agent_key not in agents:
                agent_key = self.choose_agent(step)
            step.agent = agent_key
            agent = agents[agent_key]

            compact_context = [
                {"index": entry["index"], "agent": entry["agent"], "summary": entry["summary"]}
                for entry in context[-3:]
            ]
            dependencies = [entry["index"] for entry in context]
            step.inputs = dict(step.inputs)
            step.inputs.setdefault("objective", user_objective)
            step.inputs["context"] = compact_context
            if dependencies:
                step.inputs["dependencies"] = dependencies
            if discoveries:
                step.inputs.setdefault(
                    "requirements",
                    self._select_requirements(agent_key),
                )
            if agent_key == "refiner":
                step.inputs.setdefault("plan_snapshot", plan_payload)

            raw_result = await self._dispatch_step(agent_key, agent, step, context)
            step_result = self._normalize_result(agent_key, agent, step, raw_result, idx)

            step.outputs.update(
                {
                    "summary": step_result.summary,
                    "scratch_path": step_result.scratch_path,
                    "metadata": step_result.metadata,
                }
            )

            executed_steps.append(step)
            context_entry = {
                "index": idx,
                "agent": agent_key,
                "summary": step_result.summary,
                "result": step_result,
            }
            context.append(context_entry)


            self.sp.append(
                {
                    "type": "step_summary",
                    "index": idx,
                    "agent": agent_key,
                    "summary": step_result.summary,
                    "scratch": step_result.scratch_path,
                }
            )

            if agent_key == "refiner" or "final" in step.description.lower():
                break

        self.last_context = context
        executed_plan = Plan(steps=executed_steps)
        self.current_plan = executed_plan
        return executed_plan

    def summarize_result(self, result: Any, limit: int = 160) -> str:
        if isinstance(result, StepResult):
            base = result.summary
        elif isinstance(result, dict):
            if result.get("summary"):
                base = str(result["summary"])
            elif {"verdict", "explanation"}.issubset(result.keys()):
                base = f"{result['verdict'].upper()}: {self._truncate(result['explanation'], limit - 8)}"
            else:
                base = self._truncate(json.dumps(result, ensure_ascii=False), limit)
        else:
            base = self._truncate(str(result), limit)
        base = base.strip()
        if len(base) > limit:
            base = base[:limit].rstrip() + "…"
        return base

    def _truncate(self, text: str, limit: int) -> str:
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    def _segment_description(self, description: str) -> List[str]:
        description = description.strip()
        if not description:
            return [description]

        sentences = re.split(r"(?<=[.!?])\s+", description)
        segments: List[str] = []
        buffer = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) > 160 or self._has_multiple_tasks(candidate):
                if buffer:
                    segments.append(buffer.strip())
                    buffer = ""
                segments.append(sentence)
            else:
                buffer = candidate
        if buffer:
            segments.append(buffer.strip())
        return segments or [description]

    def _has_multiple_tasks(self, text: str) -> bool:
        lowered = text.lower()
        verbs = re.findall(
            r"\b(plan|implement|refactor|review|write|test|summarize|analyze|design|build|investigate|draft|generate|refine|document|fix)\b",
            lowered,
        )
        return len(verbs) > 1 or " and " in lowered or ";" in lowered

    def _prepare_inputs(
        self,
        description: str,
        agent: Optional[str],
        base_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        inputs = dict(base_inputs or {})
        inputs["objective"] = self._objective
        requirements = self._select_requirements(agent)
        if requirements:
            inputs["requirements"] = requirements
        artifacts = self._extract_artifacts(description)
        if artifacts:
            inputs["artifacts"] = artifacts
        return inputs

    def _select_requirements(self, agent: Optional[str]) -> List[str]:
        if not self._discoveries:
            return []
        filtered = [d["insight"] for d in self._discoveries if d.get("agent") == agent]
        if not filtered:
            filtered = [d["insight"] for d in self._discoveries[:3]]
        return filtered[:3]

    def _extract_artifacts(self, text: str) -> List[str]:
        matches = re.findall(r"[\w./-]+\.(?:py|md|txt|json|yaml|yml)", text)
        seen = []
        for match in matches:
            if match not in seen:
                seen.append(match)
        return seen[:5]

    async def _dispatch_step(
        self,
        agent_key: str,
        agent: Any,
        step: Step,
        context: List[Dict[str, Any]],
    ) -> Any:
        if agent_key == "critic":
            last_non_critic = next(
                (entry for entry in reversed(context) if entry["agent"] != "critic"),
                None,
            )
            if last_non_critic:
                prior = last_non_critic["result"].raw_output or last_non_critic["summary"]
            else:
                prior = "NO PRIOR RESULT AVAILABLE"
            if isinstance(prior, dict):
                prior_payload = json.dumps(prior, ensure_ascii=False)
            else:
                prior_payload = str(prior)
            return await agent.review(step, prior_payload)

        if hasattr(agent, "run_step"):
            return await agent.run_step(step, budget_calls=self.cfg.max_tool_calls_per_step)

        if hasattr(agent, "review"):
            return await agent.review(step, "")

        raise RuntimeError(f"Agent {agent_key} does not support run_step or review")

    def _normalize_result(
        self,
        agent_key: str,
        agent: Any,
        step: Step,
        raw_result: Any,
        step_index: int,
    ) -> StepResult:
        scratch = getattr(getattr(agent, "sp", None), "location", None)

        if isinstance(raw_result, StepResult):
            summary = raw_result.summary or self.summarize_result(raw_result.raw_output)
            metadata = dict(raw_result.metadata)
            raw_output = raw_result.raw_output
            scratch_path = raw_result.scratch_path or scratch
            return StepResult(
                agent=agent_key,
                step_index=step_index,
                step_description=step.description,
                summary=summary,
                scratch_path=scratch_path,
                raw_output=raw_output,
                metadata=metadata,
            )

        if isinstance(raw_result, dict):
            summary = self.summarize_result(raw_result)
            metadata = dict(raw_result)
            raw_output = raw_result
        else:
            summary = self.summarize_result(raw_result)
            metadata = {}
            raw_output = raw_result

        return StepResult(
            agent=agent_key,
            step_index=step_index,
            step_description=step.description,
            summary=summary,
            scratch_path=scratch,
            raw_output=raw_output,
            metadata=metadata,
        )

    def choose_agent(self, step: Step) -> str:
        """Heuristic fallback for agent assignment."""
        text = step.description.lower()
        if any(
            k in text
            for k in ["list", "read", "write", "refactor", "implement", "run", "execute", "test"]
        ):
            return "executor"
        if any(
            k in text
            for k in ["analyz", "hypoth", "design", "plan", "refactor plan", "strategy"]
        ):
            return "theorist"
        if any(k in text for k in ["review", "crit", "check", "verify"]):
            return "critic"
        return "executor"
