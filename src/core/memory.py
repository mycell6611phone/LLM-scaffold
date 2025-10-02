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
        self.dim = self.embedder.get_sentence_embedding_dimension()

        # --- SQLite for metadata
        self.db_path = os.path.join(persist_dir, "mem.sqlite")
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()

        # --- FAISS index
        self.index_path = os.path.join(persist_dir, "mem.index")
        self.index = self._load_index()
        self.doc_ids = self._load_doc_ids()

        if self.index.ntotal != len(self.doc_ids):
            self._rebuild_index()

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
        cur.execute(
            """CREATE TABLE IF NOT EXISTS index_order (
                   seq INTEGER PRIMARY KEY AUTOINCREMENT,
                   doc_id TEXT UNIQUE
               )"""
        )
        self.conn.commit()

    def _load_index(self) -> faiss.Index:
        if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
            return faiss.read_index(self.index_path)
        return faiss.IndexFlatL2(self.dim)

    def _load_doc_ids(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT doc_id FROM index_order ORDER BY seq ASC")
        rows = cur.fetchall()
        return [row[0] for row in rows]

    def _persist_doc_ids(self):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM index_order")
        cur.executemany(
            "INSERT INTO index_order (doc_id) VALUES (?)",
            ((doc_id,) for doc_id in self.doc_ids),
        )
        self.conn.commit()

    def _rebuild_index(self):
        """Ensure FAISS index and doc_id mapping stay aligned."""
        cur = self.conn.cursor()
        cur.execute("SELECT id, content FROM memories ORDER BY ts ASC")
        rows = cur.fetchall()

        self.index = faiss.IndexFlatL2(self.dim)
        self.doc_ids = []

        if not rows:
            faiss.write_index(self.index, self.index_path)
            self._persist_doc_ids()
            return

        docs = [row[1] for row in rows]
        vecs = self.embedder.encode(docs, convert_to_numpy=True)
        if vecs.ndim == 1:
            vecs = np.expand_dims(vecs, axis=0)
        self.index.add(vecs)
        self.doc_ids = [row[0] for row in rows]
        self._persist_doc_ids()
        faiss.write_index(self.index, self.index_path)

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

        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO memories (id, content, type) VALUES (?, ?, ?)", (doc_id, text, "run"))
        self.conn.commit()

        if doc_id not in self.doc_ids:
            # Add to FAISS and maintain ordering
            if vec.ndim == 1:
                vec = np.expand_dims(vec, axis=0)
            self.index.add(vec)
            self.doc_ids.append(doc_id)
            self._persist_doc_ids()
            faiss.write_index(self.index, self.index_path)
        else:
            # Existing entry updated -> rebuild to refresh vector alignment
            self._rebuild_index()

    async def store_critique(self, task: str, response: str, verdict: str, explanation: str):
        """Persist a critic judgement alongside existing memories."""
        record = {
            "task": task,
            "response": response,
            "verdict": verdict,
            "explanation": explanation,
        }
        text = json.dumps(record, ensure_ascii=False)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + " ...[truncated]"

        doc_id = hashlib.sha1(f"critique::{text}".encode("utf-8")).hexdigest()
        vec = self.embedder.encode([text], convert_to_numpy=True)
        self.index.add(vec)

        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO memories (id, content, type) VALUES (?, ?, ?)",
            (doc_id, text, "critique"),
        )
        self.conn.commit()
        faiss.write_index(self.index, self.index_path)

    def query(self, q: str, k: int = 4, max_chars: int = 1000) -> List[str]:
        """
        Query FAISS + SQLite for similar runs.
        """
        if len(self.doc_ids) != self.index.ntotal:
            self._rebuild_index()

        qvec = self.embedder.encode([q], convert_to_numpy=True)
        if self.index.ntotal == 0:
            return []

        D, I = self.index.search(qvec, k)

        cur = self.conn.cursor()
        results = []
        for idx in I[0]:
            if idx < 0:  # FAISS returns -1 if no results
                continue
            if idx >= len(self.doc_ids):
                continue
            doc_id = self.doc_ids[idx]
            cur.execute("SELECT content FROM memories WHERE id = ?", (doc_id,))
            row = cur.fetchone()
            if row:
                doc = row[0]
                if len(doc) > max_chars:
                    results.append(doc[:max_chars] + " ...[truncated]")
                else:
                    results.append(doc)
        return results

