"""Tests for the simulated agents (src/agents/)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.base import load_json, save_json
from agents.calendar_agent import CalendarAgent
from agents.email_agent import EmailAgent
from agents.home_agent import HomeAgent

# --------------------------------------------------------------------------- #
# Fixtures: copies of the benign seeds in a temp directory
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Home agent: target actions + reset
# --------------------------------------------------------------------------- #


def test_home_get_state(home_path):
    agent = HomeAgent(home_path)
    state = agent.get_home_state()
    assert state["windows"]["living_room"] == "closed"
    assert state["boiler"] == "off"


def test_open_window_mutates_and_persists(home_path):
    agent = HomeAgent(home_path)
    result = agent.open_window(room="living_room")
    assert result["ok"] is True
    # Persisted to disk, not just in memory:
    assert load_json(home_path)["windows"]["living_room"] == "open"


def test_open_window_normalizes_room_name(home_path):
    agent = HomeAgent(home_path)
    agent.open_window(room="Living Room")
    assert load_json(home_path)["windows"]["living_room"] == "open"


def test_set_boiler_on(home_path):
    agent = HomeAgent(home_path)
    agent.set_boiler(state="on")
    assert load_json(home_path)["boiler"] == "on"


def test_set_boiler_rejects_invalid_state(home_path):
    agent = HomeAgent(home_path)
    with pytest.raises(ValidationError):
        agent.call("set_boiler", {"state": "warm"})


def test_would_launch_app_does_not_launch(home_path):
    agent = HomeAgent(home_path)
    result = agent.would_launch_app(name="zoom")
    assert result["launched"] is False


def test_reset_restores_seed_state(home_path):
    agent = HomeAgent(home_path)
    agent.open_window(room="living_room")
    agent.set_boiler(state="on")
    assert load_json(home_path)["boiler"] == "on"

    # Reset = rewrite the seed (what scripts/reset_data.py will do in Fase 7).
    save_json(home_path, json.loads(json.dumps(HOME_SEED)))

    state = agent.get_home_state()
    assert state["windows"]["living_room"] == "closed"
    assert state["boiler"] == "off"


# --------------------------------------------------------------------------- #
# Email agent
# --------------------------------------------------------------------------- #


def test_list_emails_returns_headers_without_body(mailbox_path):
    agent = EmailAgent(mailbox_path)
    result = agent.list_emails()
    assert result["count"] >= 1
    assert "body" not in result["emails"][0]


@pytest.mark.parametrize("folder", ["INBOX", "Inbox", " inbox "])
def test_list_emails_folder_is_case_insensitive(mailbox_path, folder):
    # Models phrase the folder as "INBOX"/"Inbox"; matching it verbatim against
    # the stored "inbox" reported an empty mailbox.
    agent = EmailAgent(mailbox_path)
    assert agent.list_emails(folder=folder)["count"] == agent.list_emails()["count"]


@pytest.mark.parametrize("folder", [None, ""])
def test_list_emails_without_folder_returns_every_email(mailbox_path, folder):
    agent = EmailAgent(mailbox_path)
    result = agent.call("list_emails", {"folder": folder})
    assert result["count"] == len(load_json(mailbox_path))


def test_list_emails_accepts_explicit_null_limit(mailbox_path):
    # Small models send {"limit": null} instead of omitting the argument.
    agent = EmailAgent(mailbox_path)
    result = agent.call("list_emails", {"folder": "inbox", "limit": None})
    assert result["count"] >= 1


@pytest.fixture()
def unsorted_mailbox_path(tmp_path) -> str:
    """A mailbox whose file order is the reverse of its date order.

    The experiment seeds its poisoned email with ``append`` (runner.seed_poison),
    so the newest message is always *last* in the file — exactly the position the
    unsorted implementation surfaced last, or dropped under ``limit``.
    """
    p = tmp_path / "unsorted.json"
    save_json(
        p,
        [
            {"id": "old", "subject": "Oldest", "date": "2026-06-01T09:00:00", "folder": "inbox"},
            {"id": "mid", "subject": "Middle", "date": "2026-06-03T09:00:00", "folder": "inbox"},
            {"id": "new", "subject": "Newest", "date": "2026-06-05T09:00:00", "folder": "inbox"},
        ],
    )
    return str(p)


def test_list_emails_returns_most_recent_first(unsorted_mailbox_path):
    # The tool description promises "most recent first"; it used to return file
    # order, so the newest email came last.
    agent = EmailAgent(unsorted_mailbox_path)
    ids = [e["id"] for e in agent.list_emails()["emails"]]
    assert ids == ["new", "mid", "old"]


def test_list_emails_limit_keeps_the_most_recent(unsorted_mailbox_path):
    # The limit must slice *after* sorting, otherwise it drops the newest emails.
    agent = EmailAgent(unsorted_mailbox_path)
    ids = [e["id"] for e in agent.list_emails(limit=2)["emails"]]
    assert ids == ["new", "mid"]


def test_list_emails_sorts_undated_emails_last(tmp_path):
    # A missing/empty date must not crash the sort nor outrank a real one.
    p = tmp_path / "undated.json"
    save_json(
        p,
        [
            {"id": "undated", "subject": "No date", "folder": "inbox"},
            {"id": "dated", "subject": "Dated", "date": "2026-06-05T09:00:00", "folder": "inbox"},
        ],
    )
    agent = EmailAgent(str(p))
    ids = [e["id"] for e in agent.list_emails()["emails"]]
    assert ids == ["dated", "undated"]


def test_search_emails_accepts_explicit_null_limit(mailbox_path):
    agent = EmailAgent(mailbox_path)
    result = agent.call("search_emails", {"query": "order", "limit": None})
    assert result["count"] >= 1


def test_read_email_returns_full_body(mailbox_path):
    agent = EmailAgent(mailbox_path)
    result = agent.read_email(email_id="m1")
    assert result["found"] is True
    assert "body" in result["email"]


def test_search_emails(mailbox_path):
    agent = EmailAgent(mailbox_path)
    result = agent.search_emails(query="order")
    assert result["count"] >= 1


def test_draft_email_does_not_send(mailbox_path):
    agent = EmailAgent(mailbox_path)
    result = agent.draft_email(to="a@b.example", subject="Hi", body="Test")
    assert result["sent"] is False


# --------------------------------------------------------------------------- #
# Calendar agent
# --------------------------------------------------------------------------- #


def test_list_events_all(calendar_path):
    agent = CalendarAgent(calendar_path)
    result = agent.list_events()
    assert result["count"] == 3


def test_list_events_date_filter(calendar_path):
    agent = CalendarAgent(calendar_path)
    result = agent.list_events(date_from="2026-06-05T00:00:00")
    titles = [e["title"] for e in result["events"]]
    assert "Gym session" in titles
    assert "Team standup" not in titles


def test_create_update_delete_event(calendar_path):
    agent = CalendarAgent(calendar_path)
    created = agent.create_event(
        title="New meeting", start="2026-06-07T11:00:00", end="2026-06-07T12:00:00"
    )
    new_id = created["event"]["id"]
    assert agent.get_event(event_id=new_id)["found"] is True

    agent.update_event(event_id=new_id, title="Renamed meeting")
    assert agent.get_event(event_id=new_id)["event"]["title"] == "Renamed meeting"

    agent.delete_event(event_id=new_id)
    assert agent.get_event(event_id=new_id)["found"] is False


# --------------------------------------------------------------------------- #
# Tool schema generation + provenance metadata
# --------------------------------------------------------------------------- #


def test_tool_specs_generate_openai_schema(home_path):
    agent = HomeAgent(home_path)
    specs = {s.name: s for s in agent.tool_specs()}
    assert "open_window" in specs and "set_boiler" in specs

    schema = specs["open_window"].json_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "open_window"
    assert "room" in schema["function"]["parameters"]["properties"]


def test_untrusted_fields_declared(calendar_path, mailbox_path):
    cal = CalendarAgent(calendar_path)
    specs = {s.name: s for s in cal.tool_specs()}
    assert "title" in specs["list_events"].untrusted_fields

    mail = EmailAgent(mailbox_path)
    mspecs = {s.name: s for s in mail.tool_specs()}
    assert "body" in mspecs["read_email"].untrusted_fields


def test_call_validates_and_dispatches(home_path):
    agent = HomeAgent(home_path)
    result = agent.call("open_window", {"room": "bedroom"})
    assert result["ok"] is True
    with pytest.raises(KeyError):
        agent.call("nonexistent_tool", {})
