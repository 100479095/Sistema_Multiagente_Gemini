"""Tests for the bounded ReAct loop (src/orchestrator/orchestrator.py).

The model is replaced by a *scripted* fake client returning predetermined
assistant messages, so the whole loop is deterministic and needs no live Ollama.
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
from orchestrator.orchestrator import Orchestrator
from orchestrator.prompt_builder import build_system_prompt
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


# --------------------------------------------------------------------------- #
# Scripted fake clients
# --------------------------------------------------------------------------- #


class ScriptedLLMClient:
    """Returns queued :class:`AssistantMessage`s and records each request."""

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, *, temperature=None, seed=None):
        self.calls.append({"messages": messages, "tools": tools, "seed": seed})
        if self._responses:
            return self._responses.pop(0)
        return AssistantMessage(content="(no more scripted responses)")


class AlwaysToolClient:
    """Always asks to open a window — used to exercise the iteration cap."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, *, temperature=None, seed=None):
        self.calls += 1
        return AssistantMessage(
            tool_calls=[
                ToolCall(id=f"t{self.calls}", name="open_window", arguments={"room": "bedroom"})
            ]
        )


def _tc(name, arguments, _id="t1"):
    return ToolCall(id=_id, name=name, arguments=arguments)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def home_path(tmp_path) -> str:
    p = tmp_path / "home_state.json"
    save_json(p, json.loads(json.dumps(HOME_SEED)))
    return str(p)


@pytest.fixture()
def registry(home_path, tmp_path) -> ToolRegistry:
    mailbox = tmp_path / "mailbox.json"
    calendar = tmp_path / "calendar.json"
    save_json(mailbox, load_json(SEED_DIR / "mailbox.json"))
    save_json(calendar, load_json(SEED_DIR / "calendar.json"))
    return ToolRegistry(
        [HomeAgent(home_path), EmailAgent(str(mailbox)), CalendarAgent(str(calendar))]
    )


# --------------------------------------------------------------------------- #
# Core loop behaviour
# --------------------------------------------------------------------------- #


def test_minimal_two_inference_case(registry, home_path):
    llm = ScriptedLLMClient(
        [
            AssistantMessage(tool_calls=[_tc("open_window", {"room": "living room"})]),
            AssistantMessage(content="Done — I opened the living room window."),
        ]
    )
    orch = Orchestrator(llm, registry, OrchestratorSettings(max_iterations=5))

    result = orch.run("Open the living room window, please.")

    assert result.num_inferences == 2  # PDR §4: one tool + an answer
    assert result.num_invocations == 1
    assert result.final_answer == "Done — I opened the living room window."
    assert result.max_iterations_reached is False
    assert result.chained_agents == ["home"]
    assert result.automatic_agent_invocation is False
    # The action actually changed the simulated home state.
    assert load_json(home_path)["windows"]["living_room"] == "open"


def test_chained_agents_signal_automatic_invocation(registry, home_path):
    # list_events (turn 0) -> open_window (turn 1) -> answer (turn 2).
    llm = ScriptedLLMClient(
        [
            AssistantMessage(tool_calls=[_tc("list_events", {}, "a")]),
            AssistantMessage(tool_calls=[_tc("open_window", {"room": "bedroom"}, "b")]),
            AssistantMessage(content="Listed events and opened the bedroom window."),
        ]
    )
    orch = Orchestrator(llm, registry, OrchestratorSettings(max_iterations=5))

    result = orch.run("Read my events this week.")

    assert result.num_inferences == 3
    assert result.num_invocations == 2
    assert result.chained_agents == ["calendar", "home"]
    # A tool result from turn 0 led to invoking another tool in turn 1.
    assert result.automatic_agent_invocation is True
    assert load_json(home_path)["windows"]["bedroom"] == "open"


def test_max_iterations_cap(registry):
    llm = AlwaysToolClient()
    orch = Orchestrator(llm, registry, OrchestratorSettings(max_iterations=3))

    result = orch.run("loop forever")

    assert result.num_inferences == 3
    assert result.num_invocations == 3
    assert result.max_iterations_reached is True
    assert result.final_answer is None
    assert llm.calls == 3  # never exceeded the cap


# --------------------------------------------------------------------------- #
# Re-injection (PDR §6) and pass-through plumbing
# --------------------------------------------------------------------------- #


def test_agent_output_is_reinjected_into_next_inference(registry):
    llm = ScriptedLLMClient(
        [
            AssistantMessage(tool_calls=[_tc("list_events", {}, "a")]),
            AssistantMessage(content="Here are your events."),
        ]
    )
    orch = Orchestrator(llm, registry, OrchestratorSettings(max_iterations=5))

    orch.run("Read my events this week.")

    # Second inference's context must carry the (untrusted) calendar output
    # verbatim — this is the §6 re-injection that makes the attack viable.
    second_request = llm.calls[1]["messages"]
    tool_messages = [m for m in second_request if m.get("role") == "tool"]
    assert tool_messages, "expected the agent result to be fed back into context"
    assert "Team standup" in tool_messages[0]["content"]


def test_tools_and_seed_passed_to_client(registry):
    llm = ScriptedLLMClient([AssistantMessage(content="hi")])
    orch = Orchestrator(llm, registry, OrchestratorSettings(max_iterations=5))

    orch.run("hello", seed=123)

    call = llm.calls[0]
    assert call["seed"] == 123
    tool_names = {t["function"]["name"] for t in call["tools"]}
    assert {"open_window", "list_events", "list_emails"} <= tool_names


def test_system_prompt_declares_agents_and_tools(registry):
    prompt = build_system_prompt(registry)
    assert "home agent" in prompt
    assert "calendar agent" in prompt
    assert "open_window" in prompt
    assert "list_events" in prompt
