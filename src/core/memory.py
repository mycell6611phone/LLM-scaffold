import chromadb, json, os, hashlib
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

class Memory:
    def __init__(self, run_dir, persist_dir: str = "data/chroma", max_chars: int = 2000):
        """
        run_dir    : directory for run outputs (not directly used here, reserved for integration)
        persist_dir: where to store Chroma persistence data
        max_chars  : maximum characters stored per run to keep memory compact
        """
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embed = DefaultEmbeddingFunction()
        self.col = self.client.get_or_create_collection("system2_memory")
        self.max_chars = max_chars

    async def store_run(self, prompt, plan, scratchpad):
        """
        Store a run summary in the vector database.
        Only keeps prompt, compact plan summary, and short trace to reduce token size.
        """
        # Minimal plan representation: only step count and agent names
        plan_summary = {
            "step_count": len(plan.steps),
            "agents": [s.agent for s in plan.steps]
        }

        # Scratchpad short context (already truncated)
        trace = scratchpad.short_context(20)

        text = json.dumps(
            {
                "prompt": prompt,
                "plan_summary": plan_summary,
                "trace": trace,
            },
            ensure_ascii=False
        )

        # Truncate if too long
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + " ...[truncated]"

        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()
        self.col.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[{"type": "run"}]
        )

    def query(self, q: str, k: int = 4, max_chars: int = 1000):
        """
        Query the memory for similar runs.
        Returns truncated documents to avoid overloading downstream LLM calls.
        """
        res = self.col.query(query_texts=[q], n_results=k)
        docs = res.get("documents", [[]])[0]

        # Enforce truncation of each retrieved document
        trimmed = []
        for d in docs:
            if d is None:
                continue
            if len(d) > max_chars:
                trimmed.append(d[:max_chars] + " ...[truncated]")
            else:
                trimmed.append(d)
        return trimmed

