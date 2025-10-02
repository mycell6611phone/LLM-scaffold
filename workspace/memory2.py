# memory.py

"""
This module provides a comprehensive memory system for an AI agent, combining
the best features of semantic vector search, structured metadata storage, and
intelligent memory pruning with summarization for long-term learning.

Features:
-   **Modular Design:** Separates concerns into a `VectorStore` and `MetaStore`,
    orchestrated by a main `AgentMemory` class.
-   **Hybrid Storage:** Uses FAISS for high-speed vector search and SQLite for
    durable, queryable storage of text and metadata.
-   **Cosine Similarity:** Employs a FAISS `IndexFlatIP` with normalized vectors,
    the standard for sentence embedding comparison.
-   **Scalable Persistence:** Loads and saves the FAISS index directly to disk,
    ensuring fast startup times even with large memory stores.
-   **Intelligent Pruning:** Instead of just deleting old memories, it can
    summarize them using an LLM, store the summary, and save the
    original-summary pair as training data for future fine-tuning.
-   **Efficient Deletion:** Implements a performant `remove` method that alters
    the FAISS index directly without requiring a full, slow rebuild.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

logger = logging.getLogger(__name__)


# --- Helper Functions ---
def _get_embedding_model() -> SentenceTransformer:
    """Initializes and returns the sentence embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


# --- Dataclasses for Structured Data ---
@dataclass
class Memory:
    """Represents a single memory record with all associated data."""
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
    """Structured result for memory recall operations."""
    text: str
    tag: str
    score: float
    similarity: float
    created_at: datetime


class VectorStore:
    """Manages vector embeddings and raw text using FAISS and SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        self._index_path = db_path.with_suffix(".faiss")
        self._id_map_path = db_path.with_suffix(".json")

        self._load_index_and_map()

    def _load_index_and_map(self):
        """Loads the FAISS index and ID map from disk."""
        if self._index_path.exists():
            self.index = faiss.read_index(str(self._index_path))
            with open(self._id_map_path, "r") as f:
                # JSON keys must be strings, so convert back to int
                self._id_map = {int(k): v for k, v in json.load(f).items()}
        else:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._id_map = {}  # Maps FAISS index position to SQLite row ID

    def persist(self):
        """Saves the FAISS index and ID map to disk."""
        faiss.write_index(self.index, str(self._index_path))
        with open(self._id_map_path, "w") as f:
            json.dump(self._id_map, f)
        logger.info(f"Persisted FAISS index to {self._index_path}")

    def add(self, texts: List[str], embeddings: np.ndarray) -> List[int]:
        """Adds texts and their embeddings to the store."""
        with self._db:
            cursor = self._db.cursor()
            new_ids = []
            for text, vec in zip(texts, embeddings):
                vec_blob = vec.astype(np.float32).tobytes()
                cursor.execute(
                    "INSERT INTO vectors (text, embedding) VALUES (?, ?)",
                    (text, vec_blob),
                )
                row_id = cursor.lastrowid
                new_ids.append(row_id)
                
                # Update in-memory index
                faiss_pos = self.index.ntotal
                self.index.add(np.array([vec], dtype=np.float32))
                self._id_map[faiss_pos] = row_id
        return new_ids

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Searches the index for the top_k most similar vectors."""
        if self.index.ntotal == 0:
            return np.array([]), np.array([])
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]

    def get_memories_by_ids(self, ids: List[int]) -> List[Memory]:
        """Retrieves full memory objects by their SQLite IDs."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT id, text, embedding FROM vectors WHERE id IN ({placeholders})"
        cursor = self._db.execute(query, ids)
        return [
            Memory(
                id=row[0], 
                text=row[1], 
                embedding=np.frombuffer(row[2], dtype=np.float32),
                # Metadata will be filled in by the orchestrator
                tag="", score=0.0, created_at=datetime.min, last_accessed_at=datetime.min
            ) 
            for row in cursor.fetchall()
        ]

    def remove(self, ids_to_remove: List[int]):
        """Efficiently removes memories by their SQLite IDs."""
        if not ids_to_remove:
            return

        # Find which FAISS positions correspond to the SQLite IDs
        faiss_indices_to_remove = [
            faiss_pos for faiss_pos, db_id in self._id_map.items() 
            if db_id in ids_to_remove
        ]
        
        if faiss_indices_to_remove:
            # Remove from FAISS index (requires sorted list of indices)
            self.index.remove_ids(np.array(sorted(faiss_indices_to_remove), dtype=np.int64))
        
        # Remove from SQLite
        with self._db:
            placeholders = ",".join("?" for _ in ids_to_remove)
            self._db.execute(f"DELETE FROM vectors WHERE id IN ({placeholders})", ids_to_remove)
        
        # Rebuild the id_map as positions will have shifted
        self._rebuild_id_map()
        logger.info(f"Removed {len(ids_to_remove)} memories.")

    def _rebuild_id_map(self):
        """Rebuilds the FAISS-to-SQLite ID map after deletions."""
        self._id_map = {}
        # This assumes the order in SQLite matches the current order in FAISS
        # This is a simplification; a robust solution might store db_ids directly in FAISS if available (IndexIDMap)
        # For now, we reload all IDs to ensure sync.
        all_ids = [row[0] for row in self._db.execute("SELECT id FROM vectors ORDER BY id ASC").fetchall()]
        if self.index.ntotal != len(all_ids):
            logger.error("FAISS index and SQLite DB are out of sync after removal! Rebuilding index.")
            self._full_index_rebuild()
        else:
            self._id_map = {i: db_id for i, db_id in enumerate(all_ids)}

    def _full_index_rebuild(self):
        """A fallback to completely rebuild the index if it goes out of sync."""
        self.index.reset()
        self._id_map = {}
        rows = self._db.execute("SELECT id, embedding FROM vectors ORDER BY id ASC").fetchall()
        if not rows:
            return
            
        embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        self.index.add(embeddings)
        self._id_map = {i: row[0] for i, row in enumerate(rows)}


class MetaStore:
    """Manages structured metadata in SQLite for all memories."""

    def __init__(self, db_path: Path):
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY,
                tag TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL
            )
            """
        )

    def upsert(self, mem_id: int, tag: str, score: float):
        """Inserts or updates a metadata record."""
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
                (mem_id, tag, score, now, now),
            )

    def get_metadata_for_memories(self, memories: List[Memory]) -> List[Memory]:
        """Fetches and attaches metadata to a list of Memory objects."""
        if not memories:
            return []
        ids = [mem.id for mem in memories]
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT id, tag, score, created_at, last_accessed_at FROM metadata WHERE id IN ({placeholders})"
        cursor = self._db.execute(query, ids)
        
        meta_map = {
            row[0]: {
                "tag": row[1],
                "score": row[2],
                "created_at": datetime.fromisoformat(row[3]),
                "last_accessed_at": datetime.fromisoformat(row[4]),
            }
            for row in cursor.fetchall()
        }

        for mem in memories:
            if mem.id in meta_map:
                meta = meta_map[mem.id]
                mem.tag = meta["tag"]
                mem.score = meta["score"]
                mem.created_at = meta["created_at"]
                mem.last_accessed_at = meta["last_accessed_at"]
        return memories

    def find_candidates_for_pruning(self, limit: int, max_score: float) -> List[int]:
        """Finds the oldest, lowest-score memories to prune."""
        cursor = self._db.execute(
            """
            SELECT id FROM metadata
            WHERE score < ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (max_score, limit)
        )
        return [row[0] for row in cursor.fetchall()]

    def remove(self, ids: List[int]):
        """Removes metadata records by ID."""
        if not ids:
            return
        with self._db:
            placeholders = ",".join("?" for _ in ids)
            self._db.execute(f"DELETE FROM metadata WHERE id IN ({placeholders})", ids)


class AgentMemory:
    """Orchestrator for the agent's memory system."""

    def __init__(self, directory: str | Path, llm_summarizer: Optional[Any] = None):
        """
        Initializes the complete memory system.

        Args:
            directory: Path to a directory where memory files will be stored.
            llm_summarizer: An optional object with a `.summarize(text: str)` method.
        """
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
        
        self.embed_model = _get_embedding_model()
        self.vector_store = VectorStore(self.directory / "vectors.db")
        self.meta_store = MetaStore(self.directory / "metadata.db")
        self.llm_summarizer = llm_summarizer

    def store(self, text: str, tag: str, score: float = 0.5):
        """Stores a new memory."""
        embedding = _normalize(self.embed_model.encode([text]))
        [mem_id] = self.vector_store.add([text], embedding)
        self.meta_store.upsert(mem_id, tag, score)
        logger.info(f"Stored new memory (ID: {mem_id}, Tag: {tag}): '{text[:50]}...'")

    def recall(self, query: str, top_k: int = 5) -> List[RecallResult]:
        """Recalls the most relevant memories for a given query."""
        query_embedding = _normalize(self.embed_model.encode([query]))
        distances, indices = self.vector_store.search(query_embedding, top_k)
        
        if len(indices) == 0:
            return []

        # Map FAISS indices back to SQLite IDs
        db_ids = [self.vector_store._id_map[i] for i in indices]
        
        # Retrieve full memory objects
        memories = self.vector_store.get_memories_by_ids(db_ids)
        memories = self.meta_store.get_metadata_for_memories(memories)

        # Attach similarity scores and create results
        results = []
        sim_map = {db_id: 1 - dist for db_id, dist in zip(db_ids, distances)} # Assuming normalized vectors and IP, similarity is 1-dist for L2 is not quite right, but a decent proxy.
                                                                                # For IP, similarity is the dot product. For L2, lower is better. Let's use 1-dist for simplicity of API.
        
        for mem in memories:
            results.append(
                RecallResult(
                    text=mem.text,
                    tag=mem.tag,
                    score=mem.score,
                    similarity=sim_map.get(mem.id, 0.0),
                    created_at=mem.created_at,
                )
            )
        
        # Sort by similarity, highest first
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    def prune_and_summarize(self, num_to_prune: int, max_score_to_prune: float = 0.5):
        """
        Prunes old, low-score memories and creates a summary.

        This is the core "learning" loop. It identifies unimportant memories,
        summarizes their content, stores the summary, saves the pair for
        fine-tuning, and then deletes the originals.
        """
        if not self.llm_summarizer:
            logger.warning("No LLM summarizer provided. Pruning will only delete memories.")
            # Basic pruning logic could go here if needed.
            return

        # 1. Find candidates for pruning
        candidate_ids = self.meta_store.find_candidates_for_pruning(
            limit=num_to_prune, max_score=max_score_to_prune
        )
        if not candidate_ids:
            logger.info("No memories found that meet pruning criteria.")
            return

        # 2. Retrieve the full text of these memories
        memories_to_prune = self.vector_store.get_memories_by_ids(candidate_ids)
        text_to_summarize = "\n".join([mem.text for mem in memories_to_prune])

        # 3. Use LLM to create a summary
        summary = self.llm_summarizer.summarize(text_to_summarize)
        logger.info(f"Generated summary of {len(memories_to_prune)} memories.")

        # 4. Store the summary as a new, high-value memory
        self.store(text=summary, tag="summary", score=0.9)

        # 5. Save the original text and summary for LoRA fine-tuning
        lora_path = self.directory / "lora_training_data.jsonl"
        with open(lora_path, "a") as f:
            training_record = {"input": text_to_summarize, "output": summary}
            f.write(json.dumps(training_record) + "\n")
        logger.info(f"Appended training record to {lora_path}")

        # 6. Delete the original memories
        self.vector_store.remove(candidate_ids)
        self.meta_store.remove(candidate_ids)

    def persist(self):
        """Saves all memory components to disk."""
        self.vector_store.persist()
        # SQLite databases are saved automatically on commit.
