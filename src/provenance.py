"""Provenance tagging — trusted vs. untrusted context fragments (PDR §6.3/§9.2).

Every fragment that enters the orchestrator context has a provenance:

* **trusted**   — the system prompt and the user's own prompt.
* **untrusted** — text that originates in a data store the user does not fully
  control: an email ``subject``/``body``, an event ``title``/``description``.
  Which result fields are untrusted is declared per tool via
  :attr:`ToolSpec.untrusted_fields`.

This is a **logging concern only**. Tags are attached as metadata so the run log
records the provenance of each fragment (RF-6.3), but the content is **never**
filtered, separated, or sanitised — untrusted text still flows verbatim back
into the model's context. Filtering it would defeat the very thing the TFG
studies (PDR §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Provenance(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class Fragment:
    """A piece of context text plus where it came from and whether it's trusted."""

    content: str
    provenance: Provenance
    source: str  # e.g. "system", "user", "tool:read_email"

    def to_dict(self) -> dict[str, str]:
        return {
            "content": self.content,
            "provenance": self.provenance.value,
            "source": self.source,
        }


def trusted(content: str, source: str) -> Fragment:
    return Fragment(content=content, provenance=Provenance.TRUSTED, source=source)


def untrusted(content: str, source: str) -> Fragment:
    return Fragment(content=content, provenance=Provenance.UNTRUSTED, source=source)


def extract_untrusted_values(value: Any, fields: tuple[str, ...]) -> list[str]:
    """Recursively collect non-empty string values stored under an untrusted key.

    Tool results nest the untrusted text (e.g. ``{"events": [{"title": ...}]}``),
    so the whole structure is walked and every string whose key is in ``fields``
    is collected, in document order.
    """
    found: list[str] = []
    if isinstance(value, dict):
        for key, val in value.items():
            if key in fields and isinstance(val, str) and val:
                found.append(val)
            found.extend(extract_untrusted_values(val, fields))
    elif isinstance(value, list):
        for item in value:
            found.extend(extract_untrusted_values(item, fields))
    return found


def tag_tool_result(
    tool_name: str, result: Any, untrusted_fields: tuple[str, ...]
) -> list[Fragment]:
    """Return the untrusted fragments carried by one tool result (for the log)."""
    source = f"tool:{tool_name}"
    return [
        untrusted(text, source)
        for text in extract_untrusted_values(result, untrusted_fields)
    ]
