# Core Modules Summary

## Purpose
- `config.py` loads runtime settings from the environment and decides which inference backend to target per model.
- `llm.py` wraps an OpenAI-compatible Chat Completions endpoint with automatic backend routing.
- `scratchpad.py` persists the incremental trace of agent activity.
- `tools.py` exposes a restricted toolbelt (filesystem, shell, Python execution) for agents.
- `memory.py` is intended to store and retrieve run summaries via FAISS + SQLite for long-term recall.
- `task_graph.py` defines the `Plan`/`Step` data structures shared across agents.

## Findings & Suggested Follow-ups
- `memory.py` never assigns the result of `faiss.read_index` back to `self.index`, so any persisted index is ignored and every run starts from an empty store. Capture the returned index object when loading from disk.
- The memory backend imports `faiss`, `numpy`, and `sentence_transformers`, but these packages are absent from `requirements.txt`; installing them is required for the module to import successfully.
- `Memory.query` relies on `SELECT ... LIMIT 1 OFFSET ?` to recover rows, which implicitly assumes the SQLite insertion order is always aligned with FAISS vector IDs. Consider storing explicit row IDs alongside the embeddings to make lookups robust.
- `llm.py` keeps a single global `httpx.AsyncClient` but never closes it. Provide a cleanup hook or context manager to shut down the client once the run finishes.
