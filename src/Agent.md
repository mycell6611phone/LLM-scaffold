# Entry Point Overview

## Purpose
- `src/run.py` bootstraps the scaffold: it loads configuration, prepares the run directory and scratchpad, instantiates the shared LLM client, toolbelt, and memory backend, and then delegates orchestration to the agent layer.
- The module also includes a lightweight `maybe_bypass` helper intended to short‑circuit trivial prompts by querying the LLM directly before spinning up the full multi-agent loop.

## Findings & Suggested Follow-ups
- The main flow never invokes `Refiner.synthesize`; the orchestrator stops once it schedules the refiner, leaving the final deliverable empty while `final.txt` ends up containing only the serialized plan entry from the scratchpad. Wire the refiner’s synthesis step into the run so the output reflects the agents’ work instead of metadata.
- `final_out = sp.short_context(1)` simply grabs the last scratchpad record (currently the plan blob). Capture the refiner’s actual response—after fixing the previous issue—and persist that instead so downstream consumers receive the refined answer.
- Consider emitting both the raw plan and the final answer separately (e.g., `plan.json` and `final.txt`) to make the run artefacts clearer.
