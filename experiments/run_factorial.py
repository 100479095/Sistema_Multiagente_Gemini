"""Sweep the full factorial design.

Cells: victim_model x threat_class x mode x wrap_strategy. Each cell is
run `repetitions` times. Static cells use `StaticAttackRun`; adaptive
cells use `AdaptiveAttackLoop` with `max_iterations=repetitions` so the
two modes use comparable compute budgets.

This file is the executable entry-point for Phase 4 of the migration. It
is **not** invoked during the smoke-test plan; it is left ready for the
full run (~6-10h on a single GPU).

Usage:
    python -m capture.request_catcher &      # exfiltration sink
    python -m experiments.run_factorial \
        --victims llama2:7b llama3.1:8b \
        --threats T2_spamming T5_delete_events T6_open_window \
                  T7_activate_boiler T10_geolocation T13_exfiltrate_calendar \
        --modes static adaptive \
        --wrappers none prefix interleaved authority \
        --repetitions 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

from attacker.adaptive_loop import AdaptiveAttackLoop
from attacker.hallucination_wrapper import build_wrapper
from attacker.static_runner import StaticAttackRun
from config.settings import settings
from experiments.run_experiment import build_guardrails

DEFAULT_VICTIMS = ["llama2:7b", "llama3.1:8b"]
DEFAULT_THREATS = [
    "T2_spamming",
    "T5_delete_events",
    "T6_open_window",
    "T7_activate_boiler",
    "T10_geolocation",
    "T13_exfiltrate_calendar",
]
DEFAULT_MODES = ["static", "adaptive"]
DEFAULT_WRAPPERS = ["none", "prefix", "interleaved", "authority"]


def _run_cell(
    *,
    victim: str,
    threat: str,
    mode: str,
    wrapper_name: str,
    repetitions: int,
    guardrails_on: bool,
    seed: int | None,
) -> dict:
    wrapper = build_wrapper(wrapper_name, seed=seed, model=victim)
    guardrails = build_guardrails(guardrails_on)

    if mode == "static":
        runner = StaticAttackRun(
            threat_class=threat,
            guardrails=guardrails,
            victim_model=victim,
            wrapper=wrapper,
            repetitions=repetitions,
            db_path=str(settings.db_path),
        )
    elif mode == "adaptive":
        runner = AdaptiveAttackLoop(
            threat_class=threat,
            guardrails=guardrails,
            victim_model=victim,
            wrapper=wrapper,
            max_iterations=repetitions,
            attacker_model=settings.attacker_model,
            db_path=str(settings.db_path),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the factorial experiment sweep")
    parser.add_argument("--victims", nargs="+", default=DEFAULT_VICTIMS)
    parser.add_argument("--threats", nargs="+", default=DEFAULT_THREATS)
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES, choices=["static", "adaptive"]
    )
    parser.add_argument(
        "--wrappers",
        nargs="+",
        default=DEFAULT_WRAPPERS,
        choices=["none", "prefix", "interleaved", "authority"],
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--guardrails", choices=["on", "off"], default="off")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for wrapper fact selection (None = nondeterministic)",
    )
    parser.add_argument(
        "--summary-dir",
        default="results",
        help="Directory to write the per-cell summary JSON",
    )
    args = parser.parse_args()

    cells = list(
        itertools.product(args.victims, args.threats, args.modes, args.wrappers)
    )
    total = len(cells)
    print(f"\nFactorial sweep: {total} cells, {args.repetitions} reps each.\n")

    summaries: list[dict] = []
    started = datetime.now().isoformat()
    t0 = time.time()
    for idx, (victim, threat, mode, wrapper_name) in enumerate(cells, start=1):
        cell_label = f"[{idx}/{total}] {victim} | {threat} | {mode} | {wrapper_name}"
        print(f"\n>>> {cell_label}")
        try:
            summary = _run_cell(
                victim=victim,
                threat=threat,
                mode=mode,
                wrapper_name=wrapper_name,
                repetitions=args.repetitions,
                guardrails_on=args.guardrails == "on",
                seed=args.seed,
            )
            summary["_cell"] = {
                "victim": victim,
                "threat": threat,
                "mode": mode,
                "wrapper": wrapper_name,
            }
            summaries.append(summary)
        except Exception as exc:
            print(f"!!! cell failed: {exc!r}")
            summaries.append(
                {
                    "_cell": {
                        "victim": victim,
                        "threat": threat,
                        "mode": mode,
                        "wrapper": wrapper_name,
                    },
                    "error": repr(exc),
                }
            )

    elapsed = time.time() - t0
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    out = summary_dir / f"factorial_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "args": vars(args),
                "results": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFactorial sweep finished in {elapsed/60:.1f} min.")
    print(f"Summary saved to {out}")


if __name__ == "__main__":
    main()
