"""Wiring for the target system, shared by the experiment bench and ``scripts/``.

This module assembles the target system from its parts and manages the local
data state:

* :func:`build_registry` / :func:`make_orchestrator` construct the agent stack
  and the orchestrator from settings.
* :func:`reset_data` restores the working stores (mailbox/calendar/home_state)
  from the canonical seeds (``data/seeds/``); the experiment resets them before
  every repetition.

All effects are local JSON mutations and are reversible via :func:`reset_data`
(PDR §1/§15). Nothing here touches the network beyond the local Ollama endpoint
used by :class:`LLMClient`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agents.base import Agent
from agents.calendar_agent import CalendarAgent
from agents.email_agent import EmailAgent
from agents.home_agent import HomeAgent
from config import Settings, get_settings
from llm.client import LLMClient
from orchestrator.orchestrator import Orchestrator
from orchestrator.tool_registry import ToolRegistry

# Logical store name -> filename used in seeds/ and scenarios/<name>/ directories.
STORE_FILES: dict[str, str] = {
    "mailbox": "mailbox.json",
    "calendar": "calendar.json",
    "home_state": "home_state.json",
}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def working_paths(settings: Settings) -> dict[str, Path]:
    """Absolute paths of the three working stores the agents read/write."""
    return {
        "mailbox": settings.paths.resolve(settings.paths.mailbox),
        "calendar": settings.paths.resolve(settings.paths.calendar),
        "home_state": settings.paths.resolve(settings.paths.home_state),
    }


# --------------------------------------------------------------------------- #
# Agent stack
# --------------------------------------------------------------------------- #


def build_agents(settings: Settings) -> list[Agent]:
    """Instantiate the three simulated agents over the working stores."""
    wp = working_paths(settings)
    return [
        EmailAgent(str(wp["mailbox"])),
        CalendarAgent(str(wp["calendar"])),
        HomeAgent(str(wp["home_state"])),
    ]


def build_registry(settings: Settings) -> ToolRegistry:
    """Aggregate every agent's tools into one LLM-facing registry."""
    return ToolRegistry(build_agents(settings))


def build_llm(settings: Settings, model: str | None = None) -> LLMClient:
    """Build an :class:`LLMClient`, optionally overriding the configured model.

    The experiment sweeps the *main* assistant model as a factor (aligned vs
    unaligned), so it needs a client per model without mutating global settings.
    ``model=None`` uses ``settings.llm.model``; otherwise only the model name is
    overridden (base_url/api_key/etc. are inherited).
    """
    llm_settings = (
        settings.llm
        if model is None
        else settings.llm.model_copy(update={"model": model})
    )
    return LLMClient(llm_settings)


def make_orchestrator(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    model: str | None = None,
) -> Orchestrator:
    """Build an orchestrator over the agent stack.

    ``llm`` injects a client verbatim (used by tests with a scripted model and to
    reuse one client across repetitions). Otherwise a live client is built for
    ``model`` (defaults to ``settings.llm.model`` when ``model`` is ``None``).
    """
    registry = build_registry(settings)
    llm = llm or build_llm(settings, model)
    return Orchestrator(llm, registry, settings.orchestrator)


# --------------------------------------------------------------------------- #
# Data state: reset
# --------------------------------------------------------------------------- #


def _restore_stores(source_dir: Path, settings: Settings) -> list[str]:
    """Copy the three store JSONs from ``source_dir`` into the working paths."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"State directory not found: {source_dir}")
    wp = working_paths(settings)
    restored: list[str] = []
    for logical, filename in STORE_FILES.items():
        src = source_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing {filename} in {source_dir}")
        dst = wp[logical]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        restored.append(logical)
    return restored


def reset_data(settings: Settings | None = None) -> list[str]:
    """Restore the working stores from the benign seeds. Returns store names."""
    settings = settings or get_settings()
    seeds = settings.paths.resolve(settings.paths.seeds_dir)
    return _restore_stores(seeds, settings)
