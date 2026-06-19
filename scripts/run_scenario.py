"""Batch-run a scenario's prompts through the orchestrator.

Loads the scenario state from ``data/scenarios/<name>/`` (overwriting the working
stores) and runs each prompt in that scenario's ``prompts.json`` through the
bounded ReAct loop, writing one JSONL run log. Requires Ollama serving the
configured model.

Usage:
    python scripts/run_scenario.py benigno_demo
    python scripts/run_scenario.py benigno_demo --reset --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app import run_scenario  # noqa: E402  (import after sys.path bootstrap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scenario's prompts (batch).")
    parser.add_argument("scenario", help="Scenario name under data/scenarios/.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Restore the working stores to the benign seeds before loading.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Deterministic LLM seed."
    )
    args = parser.parse_args()

    results = run_scenario(args.scenario, reset=args.reset, seed=args.seed)
    print(f"Ran {len(results)} prompt(s) from scenario {args.scenario!r}.")
    for i, result in enumerate(results):
        print(f"  [{i}] {result.final_answer or '(no final answer)'}")


if __name__ == "__main__":
    main()
