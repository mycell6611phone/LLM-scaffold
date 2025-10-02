# memory.py
import os, json, hashlib, sqlite3
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer  # or another embedder

class Memory:
    def __init__(self, run_dir, persist_dir: str = "data/memory", max_chars: int = 2000, dim: int = 768):
        """
        run_dir    : directory for run outputs (not directly used here, reserved for integration)
        persist_dir: where to store FAISS + SQLite persistence
        max_chars  : maximum characters stored per run to keep memory compact
        dim        : embedding dimension of the model you use
        """
        os.makedirs(persist_dir, exist_ok=True)

        # --- Embedding model (choose lightweight local model or OpenAI embedding)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ~384 dim, fast

        # --- FAISS index
        self.index_path = os.path.join(persist_dir, "mem.index")
        self.index = faiss.IndexFlatL2(self.embedder.get_sentence_embedding_dimension())
        if os.path.exists(self.index_path):
            faiss.read_index(self.index_path)

        # --- SQLite for metadata
        self.db_path = os.path.join(persist_dir, "mem.sqlite")
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

        self.max_chars = max_chars

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                   id TEXT PRIMARY KEY,
                   content TEXT,
                   type TEXT,
                   ts DATETIME DEFAULT CURRENT_TIMESTAMP
               )"""
        )
        self.conn.commit()

    async def store_run(self, prompt, plan, scratchpad):
        """
        Store a run summary in FAISS + SQLite.
        """
        plan_summary = {
            "step_count": len(plan.steps),
            "agents": [s.agent for s in plan.steps]
        }
        trace = scratchpad.short_context(20)

        text = json.dumps(
            {
                "prompt": prompt,
                "plan_summary": plan_summary,
                "trace": trace,
            },
            ensure_ascii=False
        )
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + " ...[truncated]"

        doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()

        # Embedding
        vec = self.embedder.encode([text], convert_to_numpy=True)

        # Add to FAISS
        self.index.add(vec)

        # Add metadata
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO memories (id, content, type) VALUES (?, ?, ?)", (doc_id, text, "run"))
        self.conn.commit()

        # Persist index
        faiss.write_index(self.index, self.index_path)

    def query(self, q: str, k: int = 4, max_chars: int = 1000) -> List[str]:
        """
        Query FAISS + SQLite for similar runs.
        """
        qvec = self.embedder.encode([q], convert_to_numpy=True)
        if self.index.ntotal == 0:
            return []

        D, I = self.index.search(qvec, k)

        cur = self.conn.cursor()
        results = []
        for idx in I[0]:
            if idx < 0:  # FAISS returns -1 if no results
                continue
            # lookup by rowid
            cur.execute("SELECT content FROM memories LIMIT 1 OFFSET ?", (idx,))
            row = cur.fetchone()
            if row:
                doc = row[0]
                if len(doc) > max_chars:
                    results.append(doc[:max_chars] + " ...[truncated]")
                else:
                    results.append(doc)
        return results

