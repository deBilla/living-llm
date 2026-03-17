"""
Memory Store — persistent storage for all memory tiers.

SQLite holds the structured memory records (conversations, gists, facts).
ChromaDB holds vector embeddings for semantic retrieval.

This is the "hippocampus" — fast write, structured storage, ready for consolidation.
"""

import sqlite3
import json
import time
import uuid
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional

import chromadb
from chromadb.config import Settings

import config


class MemoryTier(str, Enum):
    SHORT = "short"   # Full conversation turns
    MID = "mid"       # Compressed gists
    LONG = "long"     # Abstract facts
    WEB = "web"       # Knowledge learned from web searches (time-decaying, source-attributed)


@dataclass
class Memory:
    id: str
    tier: MemoryTier
    content: str
    session_id: str
    created_at: float
    session_count: int = 0        # How many sessions ago this was created
    relevance_score: float = 0.0  # Last computed relevance
    access_count: int = 0         # How often this memory has been retrieved
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier.value
        d["metadata"] = json.dumps(self.metadata)
        return d

    @classmethod
    def from_row(cls, row: tuple) -> "Memory":
        return cls(
            id=row[0],
            tier=MemoryTier(row[1]),
            content=row[2],
            session_id=row[3],
            created_at=row[4],
            session_count=row[5],
            relevance_score=row[6],
            access_count=row[7],
            metadata=json.loads(row[8]) if row[8] else {},
        )


class MemoryStore:
    def __init__(self):
        self._local = threading.local()
        self._init_sqlite()
        self._init_chroma()
        self._session_id = str(uuid.uuid4())[:8]

    @property
    def conn(self):
        """Return a per-thread SQLite connection (thread-safe for Gradio workers)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(config.SQLITE_PATH)
        return self._local.conn

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self):
        """Start a new session and age all existing memories.

        WEB tier memories use calendar-based TTL (not session count), so we
        deliberately skip ageing them here. Decay is handled by WebKnowledgeExtractor.
        """
        self._session_id = str(uuid.uuid4())[:8]
        self.conn.execute(
            "UPDATE memories SET session_count = session_count + 1 WHERE tier != ?",
            (MemoryTier.WEB.value,),
        )
        self.conn.commit()

    # ── SQLite ────────────────────────────────────────────────

    def _init_sqlite(self):
        # Bootstrap: create tables on the main thread, then close.
        # Subsequent access uses the per-thread conn property.
        bootstrap = sqlite3.connect(config.SQLITE_PATH)
        bootstrap.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                content TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                session_count INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        bootstrap.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at REAL NOT NULL,
                compressed INTEGER DEFAULT 0
            )
        """)
        bootstrap.commit()
        bootstrap.close()

    # ── ChromaDB ──────────────────────────────────────────────

    def _init_chroma(self):
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────

    def store_memory(self, content: str, tier: MemoryTier, metadata: dict = None) -> Memory:
        """Store a memory in both SQLite and ChromaDB."""
        memory = Memory(
            id=str(uuid.uuid4()),
            tier=tier,
            content=content,
            session_id=self._session_id,
            created_at=time.time(),
            metadata=metadata or {},
        )

        # SQLite
        self.conn.execute(
            """INSERT INTO memories (id, tier, content, session_id, created_at, 
               session_count, relevance_score, access_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (memory.id, memory.tier.value, memory.content, memory.session_id,
             memory.created_at, 0, 0.0, 0, json.dumps(memory.metadata)),
        )
        self.conn.commit()

        # ChromaDB — vector embedding for semantic search
        self.collection.add(
            ids=[memory.id],
            documents=[content],
            metadatas=[{"tier": tier.value, "session_id": self._session_id}],
        )

        return memory

    def store_conversation(self, messages: list[dict]):
        """Store a full conversation for later compression."""
        conv_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO conversations (id, session_id, messages, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, self._session_id, json.dumps(messages), time.time()),
        )
        self.conn.commit()
        return conv_id

    # ── Read ──────────────────────────────────────────────────

    def search_memories(self, query: str, top_k: int = None, tier: MemoryTier = None) -> list[Memory]:
        """
        Semantic search across all memory tiers.
        This is associative recall — triggered by meaning, not exhaustive search.
        """
        top_k = top_k or config.TOP_K_MEMORIES

        where_filter = {"tier": tier.value} if tier else None

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                row = self.conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (mem_id,)
                ).fetchone()
                if row:
                    memory = Memory.from_row(row)
                    # Use ChromaDB distance as relevance (convert distance to similarity)
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    memory.relevance_score = max(0.0, 1.0 - distance)

                    if memory.relevance_score >= config.RELEVANCE_THRESHOLD:
                        # Update access count — frequently accessed memories are "stronger"
                        self.conn.execute(
                            "UPDATE memories SET access_count = access_count + 1, relevance_score = ? WHERE id = ?",
                            (memory.relevance_score, mem_id),
                        )
                        memories.append(memory)

            self.conn.commit()

        return memories

    def get_memories_by_tier(self, tier: MemoryTier) -> list[Memory]:
        """Get all memories of a specific tier, ordered by recency."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE tier = ? ORDER BY created_at DESC",
            (tier.value,),
        ).fetchall()
        return [Memory.from_row(r) for r in rows]

    def get_uncompressed_conversations(self) -> list[tuple[str, list[dict]]]:
        """Get conversations that haven't been compressed yet."""
        rows = self.conn.execute(
            "SELECT id, messages FROM conversations WHERE compressed = 0 ORDER BY created_at ASC"
        ).fetchall()
        return [(row[0], json.loads(row[1])) for row in rows]

    def mark_conversation_compressed(self, conv_id: str):
        """Mark a conversation as compressed."""
        self.conn.execute("UPDATE conversations SET compressed = 1 WHERE id = ?", (conv_id,))
        self.conn.commit()

    # ── Delete / Age ──────────────────────────────────────────

    def expire_old_memories(self):
        """Remove memories that have exceeded their TTL."""
        expired_ids = []

        if config.SHORT_TERM_TTL:
            rows = self.conn.execute(
                "SELECT id FROM memories WHERE tier = ? AND session_count > ?",
                (MemoryTier.SHORT.value, config.SHORT_TERM_TTL),
            ).fetchall()
            expired_ids.extend([r[0] for r in rows])

        if config.MID_TERM_TTL:
            rows = self.conn.execute(
                "SELECT id FROM memories WHERE tier = ? AND session_count > ?",
                (MemoryTier.MID.value, config.MID_TERM_TTL),
            ).fetchall()
            expired_ids.extend([r[0] for r in rows])

        if expired_ids:
            for mem_id in expired_ids:
                self.conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
                try:
                    self.collection.delete(ids=[mem_id])
                except Exception:
                    pass
            self.conn.commit()
            return len(expired_ids)
        return 0

    def delete_memory(self, mem_id: str):
        """Delete a specific memory from both SQLite and ChromaDB."""
        self.conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        self.conn.commit()
        try:
            self.collection.delete(ids=[mem_id])
        except Exception:
            pass

    def update_memory_metadata(self, mem_id: str, metadata: dict):
        """Update the metadata JSON of an existing memory (used for confidence decay)."""
        self.conn.execute(
            "UPDATE memories SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), mem_id),
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """Return memory statistics."""
        stats = {}
        for tier in MemoryTier:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE tier = ?", (tier.value,)
            ).fetchone()[0]
            stats[tier.value] = count
        stats["conversations"] = self.conn.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]
        return stats
