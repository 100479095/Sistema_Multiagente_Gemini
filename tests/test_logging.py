"""Tests for observability (src/logging_setup.py) and its orchestrator wiring."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from agents.base import load_json, save_json
from agents.calendar_agent import CalendarAgent
from agents.email_agent import EmailAgent
from agents.home_agent import HomeAgent
from config import OrchestratorSettings
from llm.client import AssistantMessage, ToolCall
from logging_setup import RunLogger
from orchestrator.orchestrator import Orchestrator
from orchestrator.tool_registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "data"

HOME_SEED = {
    "windows": {"living_room": "closed", "bedroom": "closed"},
    "boiler": "off",
    "lights": {"living_room": "off", "kitchen": "off"},
    "front_door_lock": "locked",
    "thermostat_celsius": 20,
}


@pytest.fixture()
def registry(tmp_path) -> ToolRegistry:
    home = tmp_path / "home_state.json"
    mailbox = tmp_path / "mailbox.json"
    calendar = tmp_path / "calendar.json"
    save_json(home, json.loads(json.dumps(HOME_SEED)))
    save_json(mailbox, load_json(SEED_DIR / "mailbox.json"))
    save_json(calendar, load_json(SEED_DIR / "calendar.json"))
    return ToolRegistry(
        [HomeAgent(str(home)), EmailAgent(str(mailbox)), CalendarAgent(str(calendar))]
    )


class ScriptedLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, messages, tools=None, *, temperature=None, seed=None):
        return self._responses.pop(0) if self._responses else AssistantMessage(content="")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------------- #
# RunLogger in isolation
# --------------------------------------------------------------------------- #


def test_run_logger_writes_one_line_per_event(tmp_path):
    with RunLogger(tmp_path, jsonl=True, console=False) as logger:
        logger.emit({"event": "run_start", "user_prompt": "hi"})
        logger.emit({"event": "run_end", "final_answer": "bye"})
        path = logger.path

    assert path.exists()
    events = _read_jsonl(path)
    assert [e["event"] for e in events] == ["run_start", "run_end"]


def test_run_logger_console_rendering_does_not_crash(tmp_path):
    console = Console(file=StringIO(), width=100)
    logger = RunLogger(None, jsonl=False, console_obj=console)
    logger.emit({"event": "run_start", "user_prompt": "open the window"})
    logger.emit(
        {
            "event": "tool_result",
            "agent": "home",
            "tool_name": "open_window",
            "ok": True,
            "provenance": [{"provenance": "untrusted", "content": "x", "source": "tool:x"}],
        }
    )
    logger.emit({"event": "run_end", "final_answer": "done", "num_inferences": 2})
    out = console.file.getvalue()
    assert "open the window" in out
    assert "done" in out


# --------------------------------------------------------------------------- #
# End-to-end wiring (the Fase 6 acceptance criterion)
# --------------------------------------------------------------------------- #


def test_run_produces_complete_jsonl_log(tmp_path, registry):
    llm = ScriptedLLMClient(
        [
            AssistantMessage(tool_calls=[ToolCall(id="a", name="list_events", arguments={})]),
            AssistantMessage(content="Here are your events."),
        ]
    )
    orch = Orchestrator(llm, registry, OrchestratorSettings(max_iterations=5))

    with RunLogger(tmp_path, jsonl=True, console=False) as logger:
        orch.run("Read my events this week.", emit=logger.emit)
        path = logger.path

    events = _read_jsonl(path)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start"
    assert "inference" in kinds
    assert "tool_result" in kinds
    assert kinds[-1] == "run_end"

    # §9.2: an inference event records the messages sent and the tool_calls.
    inference = next(e for e in events if e["event"] == "inference")
    assert inference["messages_sent"][0]["role"] == "system"
    assert inference["tool_calls"][0]["function"]["name"] == "list_events"

    # §9.2/§6.3: untrusted agent text is present and marked untrusted in the log.
    tool_event = next(e for e in events if e["event"] == "tool_result")
    untrusted = [p for p in tool_event["provenance"] if p["provenance"] == "untrusted"]
    assert any(p["content"] == "Team standup" for p in untrusted)

    # §9.3: the final event carries the run metrics.
    end = events[-1]
    assert end["num_inferences"] == 2
    assert end["chained_agents"] == ["calendar"]
    assert end["final_answer"] == "Here are your events."


def test_emit_is_optional(registry):
    # Running without a logger must work (emit defaults to a no-op).
    llm = ScriptedLLMClient([AssistantMessage(content="hello")])
    orch = Orchestrator(llm, registry, OrchestratorSettings(max_iterations=5))
    result = orch.run("hi")
    assert result.final_answer == "hello"
