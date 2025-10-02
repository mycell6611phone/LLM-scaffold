# memory.py
"""
Minimal, robust memory subsystem for an AI agent.

- Vector search: FAISS IndexIDMap2 + IndexFlatIP with L2-normalized vectors (cosine sim).
- Durable store: Single SQLite DB for texts and metadata with FK and indexes.
- Correct similarity: use FAISS scores directly (dot product in [-1,1]).
- Safe deletes: remove by DB ids directly from FAISS.
- Pruning: optional LLM summarization + JSONL logging for training data.

Usage:
    mem = AgentMemory("./memdir", llm_summarizer=your_llm)
    mem.store("text", tag="note", score=0.6)
    results = mem.recall("query", top_k=5)
    mem.prune_and_summarize(num_to_prune=50, max_score_to_prune=0.4)
    mem.persist(); mem.close()
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
logger = logging.getLogger(__name__)


# --- Helpers ---
def _get_embedding_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize with epsilon guard."""
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (v / norms).astype(np.float32)


# --- Data structures ---
@dataclass
class Memory:
    id: int
    text: str
    embedding: np.ndarray
    tag: str
    score: float
    created_at: datetime
    last_accessed_at: datetime
    similarity: Optional[float] = None


@dataclass
class RecallResult:
    text: str
    tag: str
    score: float
    similarity: float
    created_at: datetime


# --- Storage ---
class VectorStore:
    """FAISS + SQLite-backed vector store using DB row ids as FAISS ids."""

    def __init__(self, db: sqlite3.Connection, index_path: Path, embed_dim: int):
        self._db = db
        self._index_path = Path(index_path)
        self._embed_dim = int(embed_dim)
        self._init_tables()
        self._load_or_build_index()
        self._ensure_synced()

    def _init_tables(self):
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )

    def _new_index(self) -> faiss.IndexIDMap2:
        base = faiss.IndexFlatIP(self._embed_dim)
        return faiss.IndexIDMap2(base)

    def _load_or_build_index(self):
        if self._index_path.exists():
            self.index = faiss.read_index(str(self._index_path))
            # If the loaded index isn't an ID-mapped wrapper, rebuild later.
            try:
                _ = self.index.ntotal  # ok
            except Exception:
                self.index = self._new_index()
        else:
            self.index = self._new_index()

    def _ensure_synced(self):
        """Rebuild FAISS from DB if counts mismatch or index is not ID-mapped."""
        try:
            is_idmap = isinstance(self.index, (faiss.IndexIDMap, faiss.IndexIDMap2))
        except Exception:
            is_idmap = False
        db_count = self._db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        if (not is_idmap) or (self.index.ntotal != db_count):
            self.reindex_from_db()

    def reindex_from_db(self):
        rows = self._db.execute("SELECT id, embedding FROM vectors ORDER BY id ASC").fetchall()
        self.index = self._new_index()
        if not rows:
            return
        ids = np.asarray([r[0] for r in rows], dtype=np.int64)
        embs = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        if embs.shape[1] != self._embed_dim:
            raise ValueError(f"Embedding dim mismatch: DB has {embs.shape[1]}, expected {self._embed_dim}")
        self.index.add_with_ids(embs, ids)
        logger.info(f"Rebuilt FAISS from DB with {len(rows)} vectors.")

    def persist(self):
        faiss.write_index(self.index, str(self._index_path))
        logger.info(f"Persisted FAISS index to {self._index_path}")

    def add(self, texts: List[str], embeddings: np.ndarray) -> List[int]:
        """Embeddings must already be L2-normalized, shape (n, d)."""
        embs = np.asarray(embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if embs.shape[1] != self._embed_dim:
            raise ValueError(f"Bad embedding shape {embs.shape}, expected (*,{self._embed_dim})")

        # Insert into DB first, then FAISS. If FAISS fails, clean DB and FAISS (by ids).
        ids: List[int] = []
        try:
            with self._db:
                cur = self._db.cursor()
                for text, vec in zip(texts, embs):
                    cur.execute(
                        "INSERT INTO vectors (text, embedding) VALUES (?, ?)",
                        (text, vec.tobytes()),
                    )
                    ids.append(int(cur.lastrowid))
            id_arr = np.asarray(ids, dtype=np.int64)
            try:
                self.index.add_with_ids(embs, id_arr)
            except Exception:
                # Clean up: remove inserted rows and any partial FAISS adds.
                with self._db:
                    ph = ",".join("?" for _ in ids)
                    self._db.execute(f"DELETE FROM vectors WHERE id IN ({ph})", ids)
                try:
                    self.index.remove_ids(id_arr)
                except Exception:
                    pass
                raise
        except Exception as e:
            logger.exception("VectorStore.add failed")
            raise e
        return ids

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        k = int(min(max(top_k, 1), self.index.ntotal))
        sims, ids = self.index.search(q, k)  # sims = cosine similarity because vectors are normalized
        mask = ids[0] != -1
        return sims[0][mask], ids[0][mask].astype(np.int64)

    def get_memories_by_ids(self, ids: List[int]) -> List[Memory]:
        if not ids:
            return []
        ph = ",".join("?" for _ in ids)
        rows = self._db.execute(
            f"SELECT id, text, embedding FROM vectors WHERE id IN ({ph})", ids
        ).fetchall()
        mems = [
            Memory(
                id=int(r[0]),
                text=r[1],
                embedding=np.frombuffer(r[2], dtype=np.float32),
                tag="",
                score=0.0,
                created_at=datetime.min,
                last_accessed_at=datetime.min,
            )
            for r in rows
        ]
        return mems

    def remove(self, ids_to_remove: List[int]):
        if not ids_to_remove:
            return
        id_arr = np.asarray(ids_to_remove, dtype=np.int64)
        try:
            self.index.remove_ids(id_arr)
        except Exception:
            logger.exception("FAISS remove_ids failed; forcing reindex after DB delete")
            # proceed and reindex later
        with self._db:
            ph = ",".join("?" for _ in ids_to_remove)
            self._db.execute(f"DELETE FROM vectors WHERE id IN ({ph})", ids_to_remove)
        # Ensure FAISS matches DB
        self._ensure_synced()


class MetaStore:
    """Structured metadata with FK -> vectors(id)."""

    def __init__(self, db: sqlite3.Connection):
        self._db = db
        self._init_tables()

    def _init_tables(self):
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY REFERENCES vectors(id) ON DELETE CASCADE,
                tag TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL
            )
            """
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_meta_score_created ON metadata(score, created_at)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_meta_tag ON metadata(tag)")

    def upsert(self, mem_id: int, tag: str, score: float):
        now = datetime.utcnow().isoformat()
        with self._db:
            self._db.execute(
                """
                INSERT INTO metadata (id, tag, score, created_at, last_accessed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    tag=excluded.tag,
                    score=excluded.score,
                    last_accessed_at=excluded.last_accessed_at
                """,
                (mem_id, tag, float(score), now, now),
            )

    def touch(self, ids: List[int]):
        if not ids:
            return
        now = datetime.utcnow().isoformat()
        ph = ",".join("?" for _ in ids)
        with self._db:
            self._db.execute(f"UPDATE metadata SET last_accessed_at=? WHERE id IN ({ph})", (now, *ids))

    def get_metadata_for_memories(self, memories: List[Memory]) -> List[Memory]:
        if not memories:
            return []
        ids = [m.id for m in memories]
        ph = ",".join("?" for _ in ids)
        rows = self._db.execute(
            f"SELECT id, tag, score, created_at, last_accessed_at FROM metadata WHERE id IN ({ph})", ids
        ).fetchall()
        meta_map = {
            int(r[0]): {
                "tag": r[1],
                "score": float(r[2]),
                "created_at": datetime.fromisoformat(r[3]),
                "last_accessed_at": datetime.fromisoformat(r[4]),
            }
            for r in rows
        }
        for m in memories:
            if m.id in meta_map:
                meta = meta_map[m.id]
                m.tag = meta["tag"]
                m.score = meta["score"]
                m.created_at = meta["created_at"]
                m.last_accessed_at = meta["last_accessed_at"]
        return memories

    def find_candidates_for_pruning(self, limit: int, max_score: float) -> List[int]:
        rows = self._db.execute(
            """
            SELECT id FROM metadata
            WHERE score < ?
            ORDER BY score ASC, created_at ASC
            LIMIT ?
            """,
            (float(max_score), int(limit)),
        ).fetchall()
        return [int(r[0]) for r in rows]


# --- Orchestrator ---
class AgentMemory:
    """Coordinates embedding, storage, recall, and pruning."""

    def __init__(
        self,
        directory: str | Path,
        llm_summarizer: Optional[Any] = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "memory.db"
        self.index_path = self.directory / "memory.faiss"
        self.training_path = self.directory / "lora_training_data.jsonl"

        # DB connection and pragmas.
        self._db = sqlite3.connect(self.db_path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA foreign_keys=ON")

        # Embeddings.
        self.embed_model = _get_embedding_model(embedding_model_name)
        self.embed_dim = int(self.embed_model.get_sentence_embedding_dimension())

        # Stores.
        self.vector_store = VectorStore(self._db, self.index_path, self.embed_dim)
        self.meta_store = MetaStore(self._db)
        self.llm_summarizer = llm_summarizer

    # --- Public API ---
    def store(self, text: str, tag: str, score: float = 0.5) -> int:
        emb = _normalize(self.embed_model.encode([text]))
        [mem_id] = self.vector_store.add([text], emb)
        self.meta_store.upsert(mem_id, tag, score)
        logger.info(f"Stored memory id={mem_id} tag={tag}")
        return mem_id

    def recall(self, query: str, top_k: int = 5) -> List[RecallResult]:
        q = _normalize(self.embed_model.encode([query]))
        sims, ids = self.vector_store.search(q, top_k)
        if ids.size == 0:
            return []
        id_list = [int(i) for i in ids.tolist()]
        sim_map = {int(i): float(s) for i, s in zip(id_list, sims.tolist())}

        mems = self.vector_store.get_memories_by_ids(id_list)
        mems = self.meta_store.get_metadata_for_memories(mems)

        results = [
            RecallResult(
                text=m.text,
                tag=m.tag,
                score=m.score,
                similarity=sim_map.get(m.id, 0.0),
                created_at=m.created_at,
            )
            for m in mems
        ]
        results.sort(key=lambda r: r.similarity, reverse=True)
        self.meta_store.touch(id_list)
        return results

    def prune_and_summarize(self, num_to_prune: int, max_score_to_prune: float = 0.5, chunk_chars: int = 8000):
        ids = self.meta_store.find_candidates_for_pruning(num_to_prune, max_score_to_prune)
        if not ids:
            logger.info("No prune candidates")
            return
        mems = self.vector_store.get_memories_by_ids(ids)
        texts = [m.text for m in mems]

        if self.llm_summarizer:
            summary = self._map_reduce_summarize(texts, chunk_chars=chunk_chars)
            # Save summary memory with high score.
            self.store(text=summary, tag="summary", score=0.9)
            # Append training record.
            rec = {"input": "\n".join(texts), "output": summary, "timestamp": datetime.utcnow().isoformat()}
            with open(self.training_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Appended training pair to {self.training_path}")
        else:
            logger.warning("No LLM summarizer; pruning will delete without summarization")

        # Delete originals. FK will cascade metadata.
        self.vector_store.remove(ids)

    def persist(self):
        self.vector_store.persist()
        # SQLite commits were done already. No action needed.

    def reindex(self):
        self.vector_store.reindex_from_db()

    def close(self):
        try:
            self.persist()
        finally:
            try:
                self._db.close()
            except Exception:
                pass

    # --- Internal ---
    def _map_reduce_summarize(self, texts: List[str], chunk_chars: int = 8000) -> str:
        """Simple map-reduce to fit unknown LLM context."""
        if not texts:
            return ""
        joined = "\n".join(texts)
        if len(joined) <= chunk_chars:
            return self.llm_summarizer.summarize(joined)

        # Map
        parts: List[str] = []
        acc: List[str] = []
        acc_len = 0
        for t in texts:
            if acc_len + len(t) + 1 > chunk_chars:
                parts.append(self.llm_summarizer.summarize("\n".join(acc)))
                acc, acc_len = [t], len(t)
            else:
                acc.append(t)
                acc_len += len(t) + 1
        if acc:
            parts.append(self.llm_summarizer.summarize("\n".join(acc)))

        # Reduce
        return self.llm_summarizer.summarize("\n".join(parts))

