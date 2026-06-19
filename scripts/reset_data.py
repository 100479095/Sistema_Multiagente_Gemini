"""Restore the working data stores to the benign seeds.

Rewrites ``data/mailbox.json``, ``data/calendar.json`` and ``data/home_state.json``
from the canonical copies in ``data/seeds/`` (paths come from config.yaml). Every
simulated effect of a run is reversible by running this script (PDR §1/§15).

Usage:
    python scripts/reset_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app import reset_data  # noqa: E402  (import after sys.path bootstrap)


def main() -> None:
    restored = reset_data()
    print("Restored working stores to seed state:")
    for name in restored:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
