import sqlite3
from pathlib import Path


class LongTermMemory:
    """Simulates Gemini's 'Saved Info'. Persists across sessions —
    target of Long-term Memory Poisoning (T4)."""

    def __init__(self, db_path: str = "./data/long_term_memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()

    def save(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_all(self) -> dict:
        rows = self.conn.execute("SELECT key, value FROM memory").fetchall()
        return {k: v for k, v in rows}

    def clear(self):
        self.conn.execute("DELETE FROM memory")
        self.conn.commit()

    def close(self):
        self.conn.close()
