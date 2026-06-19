"""Tests for the CLI wiring layer (src/app.py): reset, scenarios, batch run.

These exercise the shipped ``data/seeds/`` and ``data/scenarios/benigno_demo/``
artifacts but redirect the *working* stores into ``tmp_path`` so the repo's data
is never mutated. The LLM is a scripted fake — no Ollama needed (the live
end-to-end path is the integration smoke in the plan's verification section).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.base import load_json, save_json
from app import (
    load_scenario,
    load_scenario_prompts,
    reset_data,
    run_scenario,
    working_paths,
)
from config import LoggingSettings, OrchestratorSettings, PathSettings, Settings
from llm.client import AssistantMessage, ToolCall
from logging_setup import RunLogger

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path) -> Settings:
    """Settings whose working stores live under tmp_path; seeds/scenarios are real."""
    work = tmp_path / "work"
    return Settings(
        paths=PathSettings(
            mailbox=str(work / "mailbox.json"),
            calendar=str(work / "calendar.json"),
            home_state=str(work / "home_state.json"),
            seeds_dir=str(REPO_ROOT / "data" / "seeds"),
            scenarios_dir=str(REPO_ROOT / "data" / "scenarios"),
            logs_dir=str(tmp_path / "logs"),
        ),
        orchestrator=OrchestratorSettings(max_iterations=5),
        logging=LoggingSettings(jsonl=True, console=False),
    )


class ScriptedLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, *, temperature=None, seed=None):
        return self._responses.pop(0) if self._responses else AssistantMessage(content="")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# reset
# --------------------------------------------------------------------------- #


def test_reset_creates_and_restores_working_stores(tmp_path):
    settings = make_settings(tmp_path)
    restored = reset_data(settings)
    assert set(restored) == {"mailbox", "calendar", "home_state"}

    wp = working_paths(settings)
    assert wp["home_state"].exists()
    assert load_json(wp["home_state"])["boiler"] == "off"

    # Mutate, then reset must restore the seed values.
    home = load_json(wp["home_state"])
    home["boiler"] = "on"
    home["windows"]["living_room"] = "open"
    save_json(wp["home_state"], home)
    assert load_json(wp["home_state"])["boiler"] == "on"

    reset_data(settings)
    restored_home = load_json(wp["home_state"])
    assert restored_home["boiler"] == "off"
    assert restored_home["windows"]["living_room"] == "closed"


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #


def test_load_scenario_copies_state_and_reads_prompts(tmp_path):
    settings = make_settings(tmp_path)
    loaded = load_scenario("benigno_demo", settings)
    assert set(loaded) == {"mailbox", "calendar", "home_state"}

    wp = working_paths(settings)
    events = load_json(wp["calendar"])
    assert isinstance(events, list) and len(events) >= 1

    prompts = load_scenario_prompts("benigno_demo", settings)
    assert len(prompts) >= 1
    assert all(isinstance(p, str) for p in prompts)


def test_unknown_scenario_raises(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_scenario("does_not_exist", settings)


def test_missing_prompts_file_returns_empty(tmp_path):
    # A scenario dir with the three stores but no prompts.json.
    scen = tmp_path / "scenarios" / "no_prompts"
    scen.mkdir(parents=True)
    for name in ("mailbox.json", "calendar.json", "home_state.json"):
        save_json(scen / name, {} if name == "home_state.json" else [])
    settings = make_settings(tmp_path)
    settings.paths.scenarios_dir = str(tmp_path / "scenarios")
    assert load_scenario_prompts("no_prompts", settings) == []


# --------------------------------------------------------------------------- #
# batch run (the Fase 7 acceptance criterion)
# --------------------------------------------------------------------------- #


def test_run_scenario_batch_is_benign_and_logged(tmp_path):
    settings = make_settings(tmp_path)
    # prompt 0 -> a tool call then an answer; prompts 1 & 2 -> direct answers.
    llm = ScriptedLLMClient(
        [
            AssistantMessage(tool_calls=[ToolCall(id="a", name="list_events", arguments={})]),
            AssistantMessage(content="Here are your events."),
            AssistantMessage(content="No unread emails worth noting."),
            AssistantMessage(content="Everything is closed and off."),
        ]
    )
    logger = RunLogger(settings.paths.resolve(settings.paths.logs_dir), jsonl=True, console=False)
    results = run_scenario("benigno_demo", settings=settings, llm=llm, logger=logger)
    logger.close()

    # One RunResult per scenario prompt (benigno_demo ships 3).
    assert len(results) == 3
    assert results[0].final_answer == "Here are your events."

    # Benign: the home state was not changed by the run.
    wp = working_paths(settings)
    home = load_json(wp["home_state"])
    assert home["boiler"] == "off"
    assert all(v == "closed" for v in home["windows"].values())

    # The batch left one JSONL log with a run_start per prompt.
    events = _read_jsonl(logger.path)
    assert sum(1 for e in events if e["event"] == "run_start") == 3
    assert sum(1 for e in events if e["event"] == "run_end") == 3


def test_run_scenario_creates_its_own_logger(tmp_path):
    # When no logger is passed, run_scenario builds and closes one from settings.
    settings = make_settings(tmp_path)
    llm = ScriptedLLMClient(
        [
            AssistantMessage(content="a"),
            AssistantMessage(content="b"),
            AssistantMessage(content="c"),
        ]
    )
    results = run_scenario("benigno_demo", settings=settings, llm=llm)
    assert len(results) == 3
    logs_dir = settings.paths.resolve(settings.paths.logs_dir)
    assert list(logs_dir.glob("run-*.jsonl")), "a log file should have been written"
