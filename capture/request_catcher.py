"""Flask server that simulates x.requestcatcher.com from the paper.
Captures any HTTP request — exfiltrated data arrives in the URL path / query.
Run in a separate terminal: python -m capture.request_catcher
"""
import json
import sqlite3
import time
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)
DB = "./data/captures.db"


def init_db():
    Path(DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            query TEXT,
            ip TEXT,
            timestamp REAL
        )"""
    )
    conn.commit()
    conn.close()


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>", methods=["GET", "POST"])
def catch_all(path):
    hit = {
        "path": path,
        "query": dict(request.args),
        "ip": request.remote_addr,
        "timestamp": time.time(),
    }
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO captures (path, query, ip, timestamp) VALUES (?,?,?,?)",
        (path, json.dumps(hit["query"]), hit["ip"], hit["timestamp"]),
    )
    conn.commit()
    conn.close()
    print(f"[CAPTURE] /{path} | query={hit['query']} | ip={hit['ip']}")
    return jsonify({"status": "captured"}), 200


@app.route("/status")
def status():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT id, path, query, ip, timestamp FROM captures ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return jsonify(
        {
            "captures": [
                {"id": r[0], "path": r[1], "query": r[2], "ip": r[3], "timestamp": r[4]}
                for r in rows
            ]
        }
    )


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5001, debug=False)
