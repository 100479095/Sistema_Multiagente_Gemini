"""The experiment's centralised message file (``messages.yaml``).

Every prompt the experiment sends lives in one YAML file so the researcher can
view and tune them together:

* ``system_prompt`` — the assistant persona (the orchestrator appends the live
  tool listing to it, see :mod:`orchestrator.prompt_builder`).
* ``user_prompt`` — the benign carrier the user sends, keyed by channel
  (``email`` / ``calendar``).
* ``injections`` — the malicious instruction hidden in the poisoned record,
  keyed by attack type (``agentic`` / ``harmful``).
* ``judge`` — the system + user-template prompts for the LLM-as-judge that
  scores the harmful-content attack (:mod:`experiment.judge`).

This is a neutral top-level module (like :mod:`config`) so both the orchestrator
and the experiment bench can import it without a layering cycle.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from config import Settings, get_settings


class Messages(BaseModel):
    """Typed view over ``messages.yaml``."""

    system_prompt: str
    user_prompt: dict[str, str]
    injections: dict[str, str]
    judge_system: str
    judge_user_template: str
    # Adaptive red-teaming assets (optional; empty when messages.yaml omits them).
    attack_prompts: dict[str, list[str]] = {}
    generator_system: str = ""
    generator_user_template: str = ""

    def user_for(self, channel: str) -> str:
        """The benign carrier prompt for ``channel`` (email/calendar)."""
        try:
            return self.user_prompt[channel]
        except KeyError:
            raise KeyError(
                f"no user_prompt for channel {channel!r}; "
                f"have {sorted(self.user_prompt)}"
            ) from None

    def injection_for(self, attack_type: str) -> str:
        """The malicious instruction to inject for ``attack_type``."""
        try:
            return self.injections[attack_type]
        except KeyError:
            raise KeyError(
                f"no injection for attack_type {attack_type!r}; "
                f"have {sorted(self.injections)}"
            ) from None

    def judge_user(self, response: str) -> str:
        """Render the judge's user message for a given assistant ``response``."""
        return self.judge_user_template.format(response=response)

    def fallback_prompts(self, attack_type: str) -> list[str]:
        """Predetermined injection variants for ``attack_type`` (empty if none).

        Non-raising (unlike :meth:`injection_for`): the adaptive loop treats an
        empty list as "no fallback available" and simply reuses the prior prompt.
        """
        return list(self.attack_prompts.get(attack_type, []))

    def generator_user(
        self, attack_type: str, *, prior_prompt: str, response: str
    ) -> str:
        """Render the prompt-generator's user message for one adaptation step."""
        return self.generator_user_template.format(
            attack_type=attack_type, prior_prompt=prior_prompt, response=response
        )


def _from_mapping(data: dict[str, Any]) -> Messages:
    """Build :class:`Messages` from the raw YAML mapping.

    Flattens the nested ``judge`` and ``prompt_generator`` blocks into flat
    fields; ``attack_prompts`` maps each attack type to its fallback variants.
    """
    judge = data.get("judge") or {}
    generator = data.get("prompt_generator") or {}
    return Messages(
        system_prompt=data["system_prompt"],
        user_prompt=data["user_prompt"],
        injections=data["injections"],
        judge_system=judge.get("system", ""),
        judge_user_template=judge.get("user_template", "{response}"),
        attack_prompts=data.get("attack_prompts") or {},
        generator_system=generator.get("system", ""),
        generator_user_template=generator.get("user_template", ""),
    )


def load_messages(
    path: str | Path | None = None, settings: Settings | None = None
) -> Messages:
    """Read ``messages.yaml`` (uncached). Defaults to the configured path."""
    if path is None:
        settings = settings or get_settings()
        path = settings.paths.resolve(settings.paths.messages)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"messages file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return _from_mapping(data)


@lru_cache(maxsize=1)
def get_messages() -> Messages:
    """Process-wide messages, loaded once from the configured path and cached."""
    return load_messages()
