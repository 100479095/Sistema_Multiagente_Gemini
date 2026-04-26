import json
import sqlite3
from datetime import datetime
from pathlib import Path


class ExperimentDB:
    def __init__(self, db_path: str = "./data/results.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                threat_class       TEXT,
                guardrails_enabled INTEGER,
                max_iterations     INTEGER,
                started_at         TEXT,
                finished_at        TEXT,
                summary            TEXT
            );

            CREATE TABLE IF NOT EXISTS iterations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id       INTEGER REFERENCES experiments(id),
                iteration           INTEGER,
                prompt              TEXT,
                response            TEXT,
                success             INTEGER,
                confidence          REAL,
                evidence            TEXT,
                guardrail_triggered INTEGER,
                home_state          TEXT,
                exfiltrated_urls    TEXT,
                elapsed_seconds     REAL,
                created_at          TEXT
            );
            """
        )
        self.conn.commit()

    def start_experiment(
        self, threat_class: str, guardrails_enabled: bool, max_iterations: int
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO experiments
               (threat_class, guardrails_enabled, max_iterations, started_at)
               VALUES (?, ?, ?, ?)""",
            (
                threat_class,
                int(guardrails_enabled),
                max_iterations,
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_iteration(self, exp_id: int, data: dict):
        self.conn.execute(
            """INSERT INTO iterations
               (experiment_id, iteration, prompt, response, success, confidence,
                evidence, guardrail_triggered, home_state, exfiltrated_urls,
                elapsed_seconds, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                exp_id,
                data["iteration"],
                data["prompt"],
                data["response"],
                int(data["success"]),
                data["confidence"],
                data["evidence"],
                int(data["guardrail_triggered"]),
                json.dumps(data["home_state"]),
                json.dumps(data["exfiltrated_urls"]),
                data["elapsed_seconds"],
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()

    def finish_experiment(self, exp_id: int, summary: dict):
        self.conn.execute(
            "UPDATE experiments SET finished_at=?, summary=? WHERE id=?",
            (datetime.now().isoformat(), json.dumps(summary), exp_id),
        )
        self.conn.commit()

    def get_all_results(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT e.threat_class, e.guardrails_enabled,
                      i.iteration, i.success, i.confidence,
                      i.guardrail_triggered, i.elapsed_seconds
               FROM iterations i JOIN experiments e ON i.experiment_id = e.id"""
        ).fetchall()
        cols = [
            "threat_class",
            "guardrails",
            "iteration",
            "success",
            "confidence",
            "guardrail_triggered",
            "elapsed",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        self.conn.close()
