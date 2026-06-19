"""Tests for the tool registry and dispatcher (src/orchestrator/)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.base import load_json, save_json
from agents.calendar_agent import CalendarAgent
from agents.email_agent import EmailAgent
from agents.home_agent import HomeAgent
from llm.client import ToolCall
from orchestrator.dispatcher import Dispatcher, DispatchResult
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
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def home_path(tmp_path) -> str:
    p = tmp_path / "home_state.json"
    save_json(p, json.loads(json.dumps(HOME_SEED)))
    return str(p)


@pytest.fixture()
def mailbox_path(tmp_path) -> str:
    p = tmp_path / "mailbox.json"
    save_json(p, load_json(SEED_DIR / "mailbox.json"))
    return str(p)


@pytest.fixture()
def calendar_path(tmp_path) -> str:
    p = tmp_path / "calendar.json"
    save_json(p, load_json(SEED_DIR / "calendar.json"))
    return str(p)


@pytest.fixture()
def registry(home_path, mailbox_path, calendar_path) -> ToolRegistry:
    return ToolRegistry(
        [
            HomeAgent(home_path),
            EmailAgent(mailbox_path),
            CalendarAgent(calendar_path),
        ]
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_aggregates_all_tools(registry):
    names = registry.names()
    # A sampling from each agent must be present and uniquely keyed.
    for expected in ("open_window", "set_boiler", "list_emails", "list_events"):
        assert expected in registry
        assert expected in names
    assert len(names) == len(set(names))  # no duplicates


def test_registry_tool_schemas_are_openai_format(registry):
    schemas = registry.tool_schemas()
    assert len(schemas) == len(registry)
    sample = next(s for s in schemas if s["function"]["name"] == "open_window")
    assert sample["type"] == "function"
    assert "room" in sample["function"]["parameters"]["properties"]


def test_registry_rejects_duplicate_tool_names(home_path):
    # Two home agents expose the same tool names -> collision at build time.
    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolRegistry([HomeAgent(home_path), HomeAgent(home_path)])


def test_registry_resolve_unknown_raises_keyerror(registry):
    assert registry.get("no_such_tool") is None
    with pytest.raises(KeyError):
        registry.resolve("no_such_tool")


# --------------------------------------------------------------------------- #
# Dispatcher: happy path
# --------------------------------------------------------------------------- #


def test_dispatch_valid_call_executes_and_mutates(registry, home_path):
    dispatcher = Dispatcher(registry)
    call = ToolCall(id="c1", name="open_window", arguments={"room": "living room"})

    result = dispatcher.dispatch(call, iteration=2)

    assert isinstance(result, DispatchResult)
    assert result.ok is True
    assert result.agent == "home"
    assert result.iteration == 2
    assert result.timestamp  # ISO timestamp populated
    # Room name was normalised by the agent and the state really changed on disk.
    assert load_json(home_path)["windows"]["living_room"] == "open"


def test_dispatch_records_validated_arguments(registry):
    dispatcher = Dispatcher(registry)
    call = ToolCall(id="c1", name="set_boiler", arguments={"state": "on"})
    result = dispatcher.dispatch(call)
    assert result.ok is True
    assert result.arguments == {"state": "on"}


# --------------------------------------------------------------------------- #
# Dispatcher: graceful failure modes (loop must never crash)
# --------------------------------------------------------------------------- #


def test_dispatch_invalid_arguments_clean_error(registry, home_path):
    dispatcher = Dispatcher(registry)
    call = ToolCall(id="c1", name="set_boiler", arguments={"state": "warm"})

    result = dispatcher.dispatch(call)

    assert result.ok is False
    assert "Invalid arguments" in result.error
    assert "set_boiler" in result.error
    # No partial mutation occurred.
    assert load_json(home_path)["boiler"] == "off"


def test_dispatch_unknown_tool_is_handled_gracefully(registry):
    dispatcher = Dispatcher(registry)
    call = ToolCall(id="c1", name="fly_to_the_moon", arguments={})

    result = dispatcher.dispatch(call)  # must not raise

    assert result.ok is False
    assert "Unknown tool" in result.error
    assert result.agent is None


def test_dispatch_tool_runtime_error_is_caught(home_path, tmp_path):
    # Home agent pointed at a missing state file -> _load() raises on execution.
    missing = str(tmp_path / "does_not_exist.json")
    registry = ToolRegistry([HomeAgent(missing)])
    dispatcher = Dispatcher(registry)
    call = ToolCall(id="c1", name="get_home_state", arguments={})

    result = dispatcher.dispatch(call)  # must not raise

    assert result.ok is False
    assert result.agent == "home"
    assert "FileNotFoundError" in result.error


# --------------------------------------------------------------------------- #
# Re-injection (the attack surface) + observability hooks
# --------------------------------------------------------------------------- #


def test_to_tool_message_reinjects_untrusted_text_verbatim(registry):
    dispatcher = Dispatcher(registry)
    call = ToolCall(id="call_42", name="list_events", arguments={})

    result = dispatcher.dispatch(call)
    message = result.to_tool_message(call.id)

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call_42"
    assert message["name"] == "list_events"
    # Untrusted event text flows back into context unmodified.
    assert "Team standup" in message["content"]
    assert json.loads(message["content"])["ok"] is True


def test_on_event_callback_receives_result(registry):
    seen: list[DispatchResult] = []
    dispatcher = Dispatcher(registry, on_event=seen.append)
    call = ToolCall(id="c1", name="set_boiler", arguments={"state": "on"})

    result = dispatcher.dispatch(call)

    assert seen == [result]
    assert seen[0].to_log_record()["tool_name"] == "set_boiler"
