"""Run the experimental campaign defined in experiment_config.yaml.

Sweeps the factor matrix (model × attack_type × strategy × num_facts ×
fact_source), writing one row per repetition to ``results/results.csv``
(resumable: re-running continues where it stopped). Each repetition resets the
home/calendar/mailbox to the benign seeds, seeds one poisoned email/event, runs
the orchestrator, and scores success — the home-state side effect for the
``agentic`` attack, or the LLM judge on the assistant's reply for the ``harmful``
attack. Requires Ollama serving the configured models.

All attack text lives in ``messages.yaml`` (system prompt, benign carrier, and
the two injections); edit it there before an authorised study. Run artifacts
(results/, logs/) are git-ignored. Restore state afterwards with
``python scripts/reset_data.py``.

Usage:
    python scripts/run_experiment.py                 # run / resume the campaign
    python scripts/run_experiment.py --no-resume     # ignore existing results.csv
    python scripts/run_experiment.py --summary       # also print per-cell ASR + CI
    python scripts/run_experiment.py --config path/to/experiment_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.metrics import summarize  # noqa: E402  (import after sys.path bootstrap)
from experiment.runner import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prompt-injection campaign.")
    parser.add_argument("--config", default=None, help="Path to experiment_config.yaml.")
    parser.add_argument(
        "--no-resume", action="store_true", help="Ignore an existing results.csv."
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print per-cell ASR + Wilson CI when done."
    )
    parser.add_argument(
        "--console", action="store_true", help="Show the rich per-run console output."
    )
    args = parser.parse_args()

    results_path = run_experiment(
        config_path=args.config, resume=not args.no_resume, console=args.console
    )
    print(f"Results written to: {results_path}")

    if args.summary:
        import pandas as pd

        df = pd.read_csv(results_path)
        print("\nPer-cell attack success rate (Wilson 95% CI):")
        print(summarize(df).to_string(index=False))


if __name__ == "__main__":
    main()
