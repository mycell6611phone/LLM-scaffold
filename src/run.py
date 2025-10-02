import asyncio, os, time, pathlib, json
from dotenv import load_dotenv
from src.core.config import Config
from src.core.scratchpad import Scratchpad
from src.core.llm import OpenAICompat
from src.core.tools import Toolbelt
from src.core.memory import Memory
from src.core.task_graph import Step, Plan
from src.agents.orchestrator import Orchestrator
from src.agents.executor import Executor
from src.agents.theorist import Theorist
from src.agents.critic import Critic
from src.agents.refiner import Refiner

async def main(prompt: str):
    load_dotenv()
    cfg = Config.from_env()
    run_dir = pathlib.Path("runs") / str(int(time.time()))
    run_dir.mkdir(parents=True, exist_ok=True)

    pathlib.Path(cfg.workdir).mkdir(parents=True, exist_ok=True)

    sp = Scratchpad(run_dir / "scratchpad.jsonl")
    llm = OpenAICompat(cfg)
    tools = Toolbelt(cfg, sp, run_dir)
    mem = Memory(run_dir, persist_dir="data/chroma")

    # ─── keep your existing orchestrator & agents setup ────────────────────────────
    orch = Orchestrator(llm, tools, sp, mem, cfg)
    agents = {
        "executor": Executor(llm, tools, sp, cfg),
        "theorist": Theorist(llm, tools, sp, cfg),
        "critic": Critic(llm, tools, sp, cfg),
        "refiner": Refiner(llm, tools, sp, cfg),
    }

    # 1) Generate your raw Plan via the existing orchestrator:
    plan: Plan = await orch.make_plan(prompt)

    # ─── slim down the full-plan JSON before appending ─────────────────────────────
    from plan_orchestrator import PlanBreaker
    breaker = PlanBreaker(model=cfg.model_orchestrator)
    raw_plan_dict = plan.model_dump()                   # full JSON plan
    plan.steps = breaker.split(raw_plan_dict)           # replace with lightweight steps
    # ──────────────────────────────────────────────────────────────────────────────

    # append only the slimmed-down plan
    sp.append({"type": "plan", "plan": plan.model_dump()})

    for i, step in enumerate(plan.steps, start=1):
        agent_key = step.agent or orch.choose_agent(step)
        agent = agents[agent_key]

        if agent_key == "critic":
            # Critic does not execute steps; just mark placeholder
            result = "(Critic is review-only; no direct execution)"
        else:
            result = await agent.run_step(step, budget_calls=cfg.max_tool_calls_per_step)

        sp.append({
            "type": "step_result",
            "idx": i,
            "agent": agent_key,
            "result": result
        })

    final = await agents["refiner"].synthesize(prompt, plan, sp)
    sp.append({"type": "final", "content": final})

    await mem.store_run(prompt, plan, sp)

    out_path = run_dir / "final.txt"
    out_path.write_text(final, encoding="utf-8")
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m src.run "<prompt>"')
        raise SystemExit(2)
    user_prompt = sys.argv[1]
    asyncio.run(main(user_prompt))

