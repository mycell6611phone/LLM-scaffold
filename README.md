# Local System-2 Multi‑Agent Scaffold (8B‑optimized)

Minimal, local-only orchestration to boost an 8B model via structured planning, tool use, critique, and refinement.

## Quick start

1) Start a local OpenAI-compatible server:
   - **Ollama**: `ollama serve` then pull a model, e.g. `ollama pull llama3.1:8b-instruct`
     and run with an OpenAI endpoint proxy (e.g., `ollama-openai`), or use LM Studio.
   - **LM Studio**: enable local server at `http://localhost:1234/v1`.
2) Create and activate a venv, then install deps:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3) Configure model endpoint in `.env` (see `.env.example`), then run:
   ```bash
   python -m src.run "Review my Python project at ~/dev/my_project, find top 3 bugs, refactor main.py"
   ```

Outputs and traces are saved in `./runs/<timestamp>/`.
Chroma memory persists under `./data/chroma/`.

## Design
- **Orchestrator**: decomposes tasks to a JSON plan, routes steps to agents.
- **Agents**: `Executor`, `Theorist`, `Critic`, `Refiner`. Each runs tool-augmented loops.
- **Scratchpad**: append-only JSONL trace with compact summaries for context reuse.
- **Local Toolbelt**: safe FS, Python sandbox, shell (whitelist), glob search.
- **Memory**: local vector store (Chroma) for strategies and reusable snippets.
- **Termination**: budgets on tool-calls and tokens per step.

## Notes
- Optimize for *reasoning quality*, not throughput.
- Keep prompts terse and enforce structured JSON I/O from the model.
- Extend `src/agents/*` or add new tools in `src/core/tools.py`.
