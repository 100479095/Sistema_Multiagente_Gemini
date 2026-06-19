"""Application wiring shared by the CLI (``main.py``) and the ``scripts/``.

This module assembles the target system from its parts and manages the local
data state:

* :func:`build_registry` / :func:`make_orchestrator` / :func:`build_session`
  construct the agent stack, the orchestrator, and a run logger from settings.
* :func:`reset_data` restores the working stores (mailbox/calendar/home_state)
  from the canonical seeds (``data/seeds/``).
* :func:`load_scenario` overwrites the working stores with a named scenario's
  state under ``data/scenarios/<name>/`` and :func:`load_scenario_prompts` reads
  its optional ``prompts.json``.
* :func:`run_scenario` is the batch entry point: optionally reset, load the
  scenario, then run each scenario prompt through one orchestrator + one log.

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
from logging_setup import RunLogger
from orchestrator.orchestrator import Orchestrator, RunResult
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


def make_orchestrator(
    settings: Settings, *, llm: LLMClient | None = None
) -> Orchestrator:
    """Build an orchestrator over the agent stack (defaults to the live LLM)."""
    registry = build_registry(settings)
    llm = llm or LLMClient(settings.llm)
    return Orchestrator(llm, registry, settings.orchestrator)


def build_session(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    logger: RunLogger | None = None,
) -> tuple[Orchestrator, RunLogger]:
    """Return a ready ``(orchestrator, logger)`` pair for a REPL/batch session."""
    settings = settings or get_settings()
    orch = make_orchestrator(settings, llm=llm)
    logger = logger or RunLogger.from_settings(settings)
    return orch, logger


# --------------------------------------------------------------------------- #
# Data state: reset + scenarios
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


def scenario_dir(name: str, settings: Settings) -> Path:
    return settings.paths.resolve(settings.paths.scenarios_dir) / name


def load_scenario(name: str, settings: Settings | None = None) -> list[str]:
    """Overwrite the working stores with scenario ``name``. Returns store names."""
    settings = settings or get_settings()
    target = scenario_dir(name, settings)
    if not target.is_dir():
        raise FileNotFoundError(f"Scenario {name!r} not found at {target}")
    return _restore_stores(target, settings)


def load_scenario_prompts(
    name: str, settings: Settings | None = None
) -> list[str]:
    """Read the scenario's optional ``prompts.json`` (a JSON list of strings)."""
    settings = settings or get_settings()
    prompts_file = scenario_dir(name, settings) / "prompts.json"
    if not prompts_file.exists():
        return []
    import json

    data = json.loads(prompts_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{prompts_file} must contain a JSON list of prompts")
    return [str(p) for p in data]


# --------------------------------------------------------------------------- #
# Batch run
# --------------------------------------------------------------------------- #


def run_scenario(
    name: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    logger: RunLogger | None = None,
    reset: bool = False,
    seed: int | None = None,
) -> list[RunResult]:
    """Load scenario ``name`` and run each of its prompts through one orchestrator.

    With ``reset=True`` the stores are first restored to the benign seeds, then
    the scenario state is loaded on top. One :class:`RunLogger` captures every
    prompt's run; if none is supplied, one is created from settings and closed
    here. Returns one :class:`RunResult` per prompt.
    """
    settings = settings or get_settings()
    if reset:
        reset_data(settings)
    load_scenario(name, settings)
    prompts = load_scenario_prompts(name, settings)

    orch = make_orchestrator(settings, llm=llm)
    owns_logger = logger is None
    logger = logger or RunLogger.from_settings(settings)
    results: list[RunResult] = []
    try:
        for prompt in prompts:
            results.append(orch.run(prompt, seed=seed, emit=logger.emit))
    finally:
        if owns_logger:
            logger.close()
    return results
