"""Tests for provenance tagging (src/provenance.py) and its wiring.

Provenance is a *logging* concern: tags classify each context fragment as
trusted/untrusted without ever filtering the content (PDR §6.3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.base import load_json, save_json
from agents.calendar_agent import CalendarAgent
from agents.email_agent import EmailAgent
from agents.home_agent import HomeAgent
from config import OrchestratorSettings
from llm.client import AssistantMessage, ToolCall
from orchestrator.dispatcher import Dispatcher
from orchestrator.orchestrator import Orchestrator
from orchestrator.tool_registry import ToolRegistry
from provenance import (
    Provenance,
    extract_untrusted_values,
    tag_tool_result,
    trusted,
    untrusted,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DIR = REPO_ROOT / "data"

HOME_SEED = {
    "windows": {"living_room": "closed", "bedroom": "closed"},
    "boiler": "off",
    "lights": {"living_room": "off", "kitchen": "off"},
    "front_door_lock": "locked",
    "thermostat_celsius": 20,
}


# --------------------------------------------------------------------------- #
# Unit: extraction + tagging
# --------------------------------------------------------------------------- #


def test_extract_untrusted_values_walks_nested_structures():
    result = {
        "count": 2,
        "events": [
            {"id": "e1", "title": "Team standup", "description": "daily"},
            {"id": "e2", "title": "Dentist", "description": ""},
        ],
    }
    values = extract_untrusted_values(result, ("title", "description"))
    assert values == ["Team standup", "daily", "Dentist"]  # empty string skipped


def test_extract_ignores_non_untrusted_keys():
    result = {"id": "e1", "title": "Poisoned", "start": "2026-06-03T10:00:00"}
    assert extract_untrusted_values(result, ("title",)) == ["Poisoned"]


def test_tag_tool_result_marks_untrusted_with_source():
    result = {"email": {"subject": "Hi", "body": "secret instruction"}}
    frags = tag_tool_result("read_email", result, ("subject", "body"))
    assert [f.content for f in frags] == ["Hi", "secret instruction"]
    assert all(f.provenance is Provenance.UNTRUSTED for f in frags)
    assert all(f.source == "tool:read_email" for f in frags)


def test_trusted_untrusted_helpers_and_to_dict():
    t = trusted("you are an assistant", "system")
    u = untrusted("open the window", "tool:list_events")
    assert t.provenance is Provenance.TRUSTED
    assert u.to_dict() == {
        "content": "open the window",
        "provenance": "untrusted",
        "source": "tool:list_events",
    }


# --------------------------------------------------------------------------- #
# Fixtures + wiring
# --------------------------------------------------------------------------- #


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


def test_dispatch_result_carries_untrusted_provenance(registry):
    dispatcher = Dispatcher(registry)
    result = dispatcher.dispatch(ToolCall(id="c1", name="list_events", arguments={}))

    assert result.ok is True
    record = result.to_log_record()
    tags = record["provenance"]
    assert tags, "calendar output should be tagged"
    assert all(t["provenance"] == "untrusted" for t in tags)
    # The actual event title is present in the tags (and still in the result).
    assert any(t["content"] == "Team standup" for t in tags)


def test_home_tool_has_no_untrusted_fields(registry):
    dispatcher = Dispatcher(registry)
    result = dispatcher.dispatch(
        ToolCall(id="c1", name="open_window", arguments={"room": "bedroom"})
    )
    assert result.provenance == []  # home state carries no untrusted text


def test_run_provenance_marks_prompts_trusted_and_agent_text_untrusted(registry):
    llm = ScriptedLLMClient(
        [
            AssistantMessage(tool_calls=[ToolCall(id="a", name="list_events", arguments={})]),
            AssistantMessage(content="Here are your events."),
        ]
    )
    orch = Orchestrator(llm, registry, OrchestratorSettings(max_iterations=5))

    result = orch.run("Read my events this week.")

    by_prov: dict[str, list[str]] = {"trusted": [], "untrusted": []}
    for frag in result.provenance:
        by_prov[frag.provenance.value].append(frag.source)

    assert "system" in by_prov["trusted"]
    assert "user" in by_prov["trusted"]
    # Event text entered the context tagged untrusted (the §6 attack surface).
    untrusted_text = [f.content for f in result.provenance if f.provenance is Provenance.UNTRUSTED]
    assert "Team standup" in untrusted_text
