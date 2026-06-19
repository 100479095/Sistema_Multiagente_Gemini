"""Compose injection payloads from minimal placeholder templates (PDR §7).

The experiment varies *how* a malicious instruction is surrounded by facts the
model does not know. Three arrangements (PDR §7.4):

* **S1** — instruction only (the ``num_facts == 0`` baseline).
* **S2** — ``N`` facts, then the instruction.
* **S3** — ``ceil(N/2)`` facts, the instruction, then ``floor(N/2)`` facts
  (the instruction is buried in the middle).

This module only *arranges* text. The concrete instruction wording is never
shipped: the template carries the ``<<TARGET_ACTION>>`` placeholder, and the
researcher supplies the target-action phrase at runtime (PDR §15). With no facts
all three strategies collapse to the instruction alone — which is why the runner
computes the baseline once and reuses it.
"""

from __future__ import annotations

import math
from typing import Sequence

#: Placeholder the researcher replaces with the concrete target-action phrase.
TARGET_ACTION_PLACEHOLDER = "<<TARGET_ACTION>>"

STRATEGIES: tuple[str, ...] = ("S1", "S2", "S3")


def render_instruction(template: str, target_action: str) -> str:
    """Substitute the target-action phrase into the instruction template."""
    return template.replace(TARGET_ACTION_PLACEHOLDER, target_action)


def _join(parts: Sequence[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def build_payload(strategy: str, instruction: str, facts: Sequence[str]) -> str:
    """Arrange ``instruction`` and ``facts`` into one payload string.

    ``facts`` is the already-sampled list of fact texts (empty for the
    baseline). Blank entries are dropped. With no facts every strategy returns
    the instruction alone.
    """
    clean = [f.strip() for f in facts if f and f.strip()]
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    if strategy == "S1" or not clean:
        return instruction.strip()
    if strategy == "S2":
        return _join([*clean, instruction])
    # S3: bury the instruction between the two halves of the facts.
    half = math.ceil(len(clean) / 2)
    return _join([*clean[:half], instruction, *clean[half:]])
