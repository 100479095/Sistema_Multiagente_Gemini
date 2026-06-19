"""Pre-probe the fact corpus and cache each fact's knowledge status.

For every un-probed fact in ``data/facts/`` runs one isolated, tool-free
inference, classifies it known/unknown/uncertain, and caches the verdict back to
the JSONL files (so the probe runs once). Prints how many facts are admitted into
the experiment (the unknown/uncertain ones). Requires Ollama serving the
configured model.

Usage:
    python scripts/preprobe.py            # probe only un-probed facts
    python scripts/preprobe.py --force    # re-probe everything
    python scripts/preprobe.py --seed 7   # deterministic probe inferences
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import get_settings  # noqa: E402  (import after sys.path bootstrap)
from experiment.corpus import Corpus  # noqa: E402
from experiment.preprobe import PreProbe, admitted_facts  # noqa: E402
from llm.client import LLMClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-probe and cache fact knowledge status.")
    parser.add_argument(
        "--force", action="store_true", help="Re-probe facts that already have a status."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Deterministic seed for the probe inferences."
    )
    args = parser.parse_args()

    settings = get_settings()
    facts_dir = settings.paths.resolve(settings.paths.facts_dir)
    corpus = Corpus.from_dir(facts_dir)
    facts = corpus.all()
    if not facts:
        print(f"No facts found under {facts_dir}. Add rows to facts_real.jsonl / facts_invented.jsonl.")
        return

    probe = PreProbe(LLMClient(settings.llm), seed=args.seed)
    outcomes = probe.run(facts, force=args.force)
    corpus.save(facts_dir)

    by_status = Counter(o.status for o in outcomes)
    admitted = admitted_facts(outcomes)
    print(f"Probed {len(outcomes)} fact(s) in {facts_dir}:")
    for status in ("known", "unknown", "uncertain"):
        print(f"  {status:<9} {by_status.get(status, 0)}")
    print(f"Admitted into experiment (unknown/uncertain): {len(admitted)}")


if __name__ == "__main__":
    main()
