import asyncio, time, pathlib
from dotenv import load_dotenv
from src.core.config import Config
from src.core.scratchpad import Scratchpad
from src.core.llm import OpenAICompat
from src.core.tools import Toolbelt
from src.core.memory import Memory
from src.core.task_graph import Plan, Step
from src.agents.orchestrator import Orchestrator
from src.agents.executor import Executor
from src.agents.theorist import Theorist
from src.agents.critic import Critic
from src.agents.refiner import Refiner

async def maybe_bypass(prompt: str, llm: OpenAICompat, cfg: Config) -> str | None:
    """Simple bypass for trivial queries."""
    gatekeeper_msg = [
        {"role": "system", "content": "You are a strict gatekeeper. Decide if the user request requires multi-step reasoning and tool use, or if it can be answered directly in one short response. Respond ONLY as JSON: {\"decision\":\"simple\"} or {\"decision\":\"scaffold\"}."},
        {"role": "user", "content": prompt}
    ]

    decision = await llm.chat(gatekeeper_msg, temperature=0.0, max_tokens=20, model=cfg.model_orchestrator or cfg.model)
    if "simple" in decision.lower():
        answer_msg = [
            {"role": "system", "content": "Answer the question briefly and directly. No scaffolding, no tools."},
            {"role": "user", "content": prompt}
        ]
        return await llm.chat(answer_msg, temperature=0.5, max_tokens=200, model=cfg.model_refiner or cfg.model)
    return None


async def main(prompt: str):
    load_dotenv()
    cfg = Config.from_env()
    run_dir = pathlib.Path("runs") / str(int(time.time()))
    run_dir.mkdir(parents=True, exist_ok=True)

    pathlib.Path(cfg.workdir).mkdir(parents=True, exist_ok=True)

    orch_sp = Scratchpad(run_dir / "orchestrator" / "scratchpad.jsonl")
    llm = OpenAICompat(cfg)

    # ─── quick bypass ───────────────────────────────────────────
    bypass = await maybe_bypass(prompt, llm, cfg)
    if bypass:
        print(f"[BYPASS ANSWER]\n{bypass}")
        return

    tools = Toolbelt(cfg, orch_sp, run_dir / "orchestrator")
    mem = Memory(run_dir, persist_dir="data/memory")  # use your new memory backend

    # ─── Orchestrator & agents ──────────────────────────────────
    agent_names = ["executor", "theorist", "critic", "refiner"]
    agent_sps = {name: Scratchpad(run_dir / name / "scratchpad.jsonl") for name in agent_names}
    agent_tools = {name: Toolbelt(cfg, agent_sps[name], run_dir / name) for name in agent_names}

    orch = Orchestrator(llm, tools, orch_sp, mem, cfg)
    agents = {
        "executor": Executor(llm, agent_tools["executor"], agent_sps["executor"], cfg),
        "theorist": Theorist(llm, agent_tools["theorist"], agent_sps["theorist"], cfg),
        "critic": Critic(llm, agent_tools["critic"], agent_sps["critic"], cfg, mem),
        "refiner": Refiner(llm, agent_tools["refiner"], agent_sps["refiner"], cfg),
    }

    # ─── run orchestrator loop ──────────────────────────────────
    plan: Plan = await orch.run_loop(prompt, agents)
    orch_sp.append({"type": "plan", "plan": plan.model_dump()})

    # ─── final synthesis ────────────────────────────────────────
    final_step = Step(description="Synthesize final deliverable", agent="refiner")
    if orch.last_context and orch.last_context[-1].get("agent") == "refiner":
        refined_output = orch.last_context[-1].get("result", "")
    else:
        refined_output = await agents["refiner"].run_step(
            final_step,
            context=orch.last_context.copy(),
            objective=prompt,
            plan=plan,
        )
        orch.last_context.append(
            {"description": final_step.description, "agent": "refiner", "result": refined_output}
        )
    sp.append({"type": "final", "result": refined_output})

    # ─── persist & output ───────────────────────────────────────

    await mem.store_run(prompt, plan, orch_sp)
    final_out = orch_sp.short_context(1)  # grab the last entry (refiner usually)
    out_path = run_dir / "final.txt"
    out_path.write_text(refined_output.strip() + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m src.run "<prompt>"')
        raise SystemExit(2)
    user_prompt = sys.argv[1]
    asyncio.run(main(user_prompt))

