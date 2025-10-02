memory.py

Handles the AGI's long-term memory.

This module provides a hybrid memory system that combines the semantic search
capabilities of a vector database (FAISS) with the structured metadata storage
of a relational database (SQLite). This allows the AGI to retrieve memories
based on conceptual similarity while also tracking important metadata like
recall counts and importance.
"""

import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from datetime import datetime

# --- Constants ---
DB_PATH = "memory.db"
FAISS_INDEX_PATH = "memory.index"
# Using a lightweight, high-performance model that runs locally.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
MAX_TOKENS = 512 # Max tokens for content to avoid overflow

class MemoryManager:
    """Manages the storage and retrieval of memories."""

    def __init__(self):
        """
        Initializes the MemoryManager, loading the embedding model,
        tokenizer, database, and FAISS index.
        """
        print("MemoryManager: Initializing...")
        
        # Ensure you have installed the necessary libraries:
        # pip install sentence-transformers faiss-cpu transformers torch
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self._init_db()
        self._load_or_init_faiss()
        
        print("MemoryManager: Initialization complete.")

    def _init_db(self):
        """Initializes the SQLite database and creates the memories table if it doesn't exist."""
        try:
            self.conn = sqlite3.connect(DB_PATH)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance_weight REAL DEFAULT 0.5,
                    recall_count INTEGER DEFAULT 0,
                    tags TEXT
                )
            ''')
            self.conn.commit()
        finally:
            if self.conn:
                self.conn.close()

    def _load_or_init_faiss(self):
        """Loads the FAISS index from disk or creates a new one if it doesn't exist."""
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"MemoryManager: Loading existing FAISS index from {FAISS_INDEX_PATH}")
            self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            print("MemoryManager: No FAISS index found. Creating a new one.")
            # We use IndexIDMap to map the FAISS internal index to our SQLite IDs
            index = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_index = faiss.IndexIDMap(index)

    def _save_faiss_index(self):
        """Saves the current FAISS index to disk."""
        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)

    def add(self, content: str, importance: float = 0.5, tags: list = None):
        """
        Adds a new memory to the system.

        Args:
            content (str): The text content of the memory.
            importance (float): A score from 0.0 to 1.0 indicating the memory's importance.
            tags (list, optional): A list of string tags associated with the memory.
        """
        # 1. Tokenizer count check
        token_count = len(self.tokenizer.encode(content))
        if token_count > MAX_TOKENS:
            print(f"Error: Memory content exceeds max token limit of {MAX_TOKENS}. Memory not added.")
            return

        # 2. Generate embedding
        embedding = self.embedding_model.encode([content])
        
        # 3. Store metadata in SQLite
        tag_str = ",".join(tags) if tags else ""
        timestamp = datetime.now().isoformat()
        
        try:
            self.conn = sqlite3.connect(DB_PATH)
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO memories (content, timestamp, importance_weight, tags) VALUES (?, ?, ?, ?)",
                (content, timestamp, importance, tag_str)
            )
            memory_id = cursor.lastrowid
            self.conn.commit()
        finally:
            if self.conn:
                self.conn.close()

        # 4. Add embedding to FAISS index with the SQLite ID
        self.faiss_index.add_with_ids(np.array(embedding, dtype=np.float32), np.array([memory_id]))
        self._save_faiss_index()
        
        print(f"MemoryManager: Added new memory with ID: {memory_id}")

    def retrieve(self, query_text: str, top_k: int = 5) -> list:
        """
        Retrieves the most relevant memories based on a query text.

        Args:
            query_text (str): The text to search for.
            top_k (int): The number of memories to return.

        Returns:
            list: A list of dictionaries, where each dictionary represents a memory.
        """
        query_embedding = self.embedding_model.encode([query_text])
        
        # Search the FAISS index
        distances, ids = self.faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        retrieved_memories = []
        if ids.size > 0:
            # Filter out invalid IDs (-1)
            valid_ids = [int(id) for id in ids[0] if id != -1]
            if not valid_ids:
                return []

            try:
                self.conn = sqlite3.connect(DB_PATH)
                cursor = self.conn.cursor()
                # Use a placeholder string for the IN clause
                placeholders = ",".join("?" * len(valid_ids))
                cursor.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", valid_ids)
                rows = cursor.fetchall()
                
                for row in rows:
                    # Increment recall count for each retrieved memory
                    self.update_recall_count(row[0])
                    
                    retrieved_memories.append({
                        "id": row[0],
                        "content": row[1],
                        "timestamp": row[2],
                        "importance": row[3],
                        "recall_count": row[4],
                        "tags": row[5].split(",") if row[5] else []
                    })
            finally:
                if self.conn:
                    self.conn.close()

        return retrieved_memories

    def update_recall_count(self, memory_id: int):
        """Increments the recall count for a specific memory."""
        try:
            self.conn = sqlite3.connect(DB_PATH)
            cursor = self.conn.cursor()
            cursor.execute("UPDATE memories SET recall_count = recall_count + 1 WHERE id = ?", (memory_id,))
            self.conn.commit()
        finally:
            if self.conn:
                self.conn.close()

    def update_importance(self, memory_id: int, new_importance: float):
        """Updates the importance score for a specific memory."""
        try:
            self.conn = sqlite3.connect(DB_PATH)
            cursor = self.conn.cursor()
            cursor.execute("UPDATE memories SET importance_weight = ? WHERE id = ?", (new_importance, memory_id))
            self.conn.commit()
        finally:
            if self.conn:
                self.conn.close()

# --- Example Usage ---
if __name__ == "__main__":
    # Clean up previous runs for a fresh start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)

    mem_manager = MemoryManager()
    
    print("\n--- Adding Memories ---")
    mem_manager.add("The AGI project started on a Tuesday.", importance=0.8, tags=["project_management", "history"])
    mem_manager.add("The core loop orchestrates all modules.", importance=0.9, tags=["architecture", "core_loop"])
    mem_manager.add("The weather in California is often sunny.", importance=0.3, tags=["geography", "weather"])
    mem_manager.add("Memory is managed by a hybrid FAISS and SQLite system.", importance=0.9, tags=["architecture", "memory"])
    
    print("\n--- Retrieving Memories for 'AGI architecture' ---")
    results = mem_manager.retrieve("Information about the AGI's architecture")
    
    for res in results:
        print(f"  ID: {res['id']}, Importance: {res['importance']}, Recalls: {res['recall_count']}")
        print(f"  Content: {res['content']}\n")
