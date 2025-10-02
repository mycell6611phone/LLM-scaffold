# Agent Layer Notes

## Purpose
- `orchestrator.py` drives the step-by-step coordination loop, delegating tasks to specialized agents based on LLM guidance or heuristics.
- `executor.py`, `theorist.py`, `critic.py`, and `refiner.py` implement the concrete agent behaviors, while `prompts.py` houses their system prompts and `util.py` provides JSON action parsing.

## Findings & Suggested Follow-ups
- In `Orchestrator.run_loop`, the fallback path for agents without `run_step` calls `agent.review(step, context)`, but `Critic.review` expects the second argument to be the latest output string. Passing the full context list will produce confusing critiques. Feed the critic the actual artifact it should assess (e.g., the most recent execution result).
- The refiner’s `run_step` is a stub that returns an empty string, so even when the orchestrator hands off to the refiner the pipeline never produces a synthesized deliverable. Connect the orchestrator to `Refiner.synthesize` (or move the logic into `run_step`) so the final agent can generate output.
- Consider enriching the step bookkeeping: store each agent’s produced text inside the `Step.outputs` structure so downstream components (memory, UI) can replay the execution trace without scraping the scratchpad.
