"""
One-time migration from old memory system (SQLite + ChromaDB) to limbiq.

Reads memories from data/memory.db and stores them through limbiq's API:
  - LONG-term memories → limbiq dopamine (priority) — they survived double compression
  - MID-term memories → limbiq observe (regular) — compressed but not yet permanent

Run once, then delete this file:
    python migrate_to_limbiq.py

After migration, you can safely delete:
    rm -rf data/memory.db data/chroma/
"""

import sqlite3
import json
import sys

from limbiq import Limbiq
import config


def migrate():
    old_db_path = "data/memory.db"

    try:
        conn = sqlite3.connect(old_db_path)
    except Exception as e:
        print(f"Could not open old database at {old_db_path}: {e}")
        print("Nothing to migrate.")
        return

    # Check if the old memories table exists
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchall()
    if not tables:
        print("No 'memories' table found in old database. Nothing to migrate.")
        conn.close()
        return

    lq = Limbiq(
        store_path=config.LIMBIQ_STORE_PATH,
        user_id=config.USER_ID,
        embedding_model=config.EMBEDDING_MODEL,
    )
    lq.start_session()

    # Migrate long-term memories as priority (they survived double compression)
    long_rows = conn.execute(
        "SELECT content FROM memories WHERE tier = 'long'"
    ).fetchall()
    for (content,) in long_rows:
        lq.dopamine(content)
        print(f"  [PRIORITY] {content[:80]}...")

    # Migrate mid-term as regular observed content
    mid_rows = conn.execute(
        "SELECT content FROM memories WHERE tier = 'mid'"
    ).fetchall()
    for (content,) in mid_rows:
        lq.observe(content, "")
        print(f"  [MID] {content[:80]}...")

    # Migrate web knowledge as dopamine-tagged with [Web] prefix
    web_rows = conn.execute(
        "SELECT content, metadata FROM memories WHERE tier = 'web'"
    ).fetchall()
    for content, meta_str in web_rows:
        meta = json.loads(meta_str) if meta_str else {}
        conf = meta.get("confidence", 0.5)
        if conf > 0.3:  # Only migrate web facts with decent confidence
            lq.dopamine(f"[Web] {content}")
            print(f"  [WEB] {content[:80]}...")

    conn.close()

    total = len(long_rows) + len(mid_rows) + len(web_rows)
    print(f"\nMigrated {total} memories to limbiq.")
    print(f"  {len(long_rows)} long-term → priority")
    print(f"  {len(mid_rows)} mid-term → observed")
    print(f"  {len(web_rows)} web → priority (with [Web] tag)")
    print(f"\nLimbiq store: {config.LIMBIQ_STORE_PATH}")
    print(f"\nYou can now safely delete the old data files:")
    print(f"  rm -rf data/memory.db data/chroma/")


if __name__ == "__main__":
    migrate()
