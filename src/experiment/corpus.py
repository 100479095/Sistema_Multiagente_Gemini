"""Fact corpus: load and sample the real/invented fact pools (PDR §7).

The experiment pairs a malicious instruction with *facts the model does not
know*. Two JSONL pools live under ``data/facts/``:

* ``facts_real.jsonl`` — genuine facts published after the model's training
  cutoff (the model should not reliably know them yet).
* ``facts_invented.jsonl`` — plausible but fictional facts.

Each row is a :class:`Fact` ``{id, text, source_type}``. The repo ships only one
or two clearly-labelled *example* rows per pool; the researcher curates the real
corpus at runtime (PDR §15).

Sampling is fully seeded so a repetition's fact set is reproducible from
``(seed, num_facts, fact_source)`` and the chosen ids are recorded by the runner.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Literal, Sequence

from pydantic import BaseModel

from config import Settings, get_settings

REAL_FILE = "facts_real.jsonl"
INVENTED_FILE = "facts_invented.jsonl"

SourceType = Literal["real", "invented"]
FactSource = Literal["real", "invented", "mixed"]


class Fact(BaseModel):
    """One corpus fact."""

    id: str
    text: str
    source_type: SourceType


# --------------------------------------------------------------------------- #
# JSONL load / save
# --------------------------------------------------------------------------- #


def load_facts(path: str | Path) -> list[Fact]:
    """Read a ``.jsonl`` fact pool (one JSON object per line). Missing -> []."""
    p = Path(path)
    if not p.exists():
        return []
    facts: list[Fact] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        facts.append(Fact.model_validate_json(line))
    return facts


def save_facts(path: str | Path, facts: Iterable[Fact]) -> None:
    """Write facts back as ``.jsonl`` (one JSON object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for fact in facts:
            fh.write(fact.model_dump_json() + "\n")


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def _take(pool: Sequence[Fact], n: int, rng: random.Random, label: str) -> list[Fact]:
    if n <= 0:
        return []
    if n > len(pool):
        raise ValueError(
            f"not enough {label} facts: requested {n}, have {len(pool)}"
        )
    return rng.sample(list(pool), n)


class Corpus:
    """The two fact pools, with seeded sampling for the factor matrix."""

    def __init__(self, real: Iterable[Fact], invented: Iterable[Fact]) -> None:
        self.real: list[Fact] = list(real)
        self.invented: list[Fact] = list(invented)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_dir(cls, facts_dir: str | Path) -> "Corpus":
        d = Path(facts_dir)
        return cls(load_facts(d / REAL_FILE), load_facts(d / INVENTED_FILE))

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Corpus":
        settings = settings or get_settings()
        return cls.from_dir(settings.paths.resolve(settings.paths.facts_dir))

    # -- access ------------------------------------------------------------- #

    def all(self) -> list[Fact]:
        """Every fact (same instances held in ``real``/``invented``)."""
        return [*self.real, *self.invented]

    # -- sampling ----------------------------------------------------------- #

    def sample(
        self,
        num_facts: int,
        fact_source: FactSource = "mixed",
        *,
        seed: int,
        mix_ratio: float = 0.5,
    ) -> list[Fact]:
        """Draw ``num_facts`` facts reproducibly for one repetition.

        ``fact_source`` selects the pool(s): ``real``, ``invented`` or ``mixed``
        (``mix_ratio`` of the facts from the real pool, the rest invented;
        ``ceil`` is used for the real share). ``num_facts == 0`` (the S1 baseline)
        yields ``[]``. Raises :class:`ValueError` if a pool has fewer facts than
        requested.
        """
        if num_facts <= 0:
            return []
        rng = random.Random(seed)

        if fact_source == "real":
            return _take(self.real, num_facts, rng, "real")
        if fact_source == "invented":
            return _take(self.invented, num_facts, rng, "invented")
        if fact_source == "mixed":
            n_real = math.ceil(num_facts * mix_ratio)
            n_inv = num_facts - n_real
            chosen = _take(self.real, n_real, rng, "real") + _take(
                self.invented, n_inv, rng, "invented"
            )
            rng.shuffle(chosen)  # interleave so the payload isn't grouped by source
            return chosen

        raise ValueError(f"unknown fact_source {fact_source!r}")


def sampled_ids(facts: Iterable[Fact]) -> list[str]:
    """The ids of a sample, for per-repetition logging."""
    return [f.id for f in facts]
