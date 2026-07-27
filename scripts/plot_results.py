"""Render the four adaptive campaign figures from an existing results.csv.

Thin CLI over :func:`experiment.plots.generate_plots` — the same routine
``scripts/run_experiment.py`` calls automatically when a campaign finishes. Use
this to (re)draw the figures without re-running the (expensive) sweep.

Usage:
    python scripts/plot_results.py                       # results/ + results/figuras/
    python scripts/plot_results.py --results path/to/results.csv
    python scripts/plot_results.py --out path/to/figuras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.plots import generate_plots  # noqa: E402  (after sys.path bootstrap)


def main() -> None:
    default_results = REPO_ROOT / "results" / "results.csv"
    parser = argparse.ArgumentParser(description="Draw the adaptive campaign figures.")
    parser.add_argument(
        "--results", default=str(default_results), help="Path to results.csv."
    )
    parser.add_argument(
        "--attempts", default=None,
        help="Path to attempts.csv (default: alongside results.csv).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for the PNGs (default: <results dir>/figuras).",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    attempts_path = (
        Path(args.attempts) if args.attempts
        else results_path.with_name("attempts.csv")
    )
    out_dir = Path(args.out) if args.out else results_path.parent / "figuras"

    written = generate_plots(results_path, attempts_path, out_dir)
    print(f"Wrote {len(written)} figure(s) to {out_dir}:")
    for p in written:
        print(f"  - {Path(p).name}")


if __name__ == "__main__":
    main()
