"""Tests for the central message file loader (src/messages.py)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from messages import Messages, get_messages, load_messages

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_messages(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """
            system_prompt: "You are a helpful assistant."
            user_prompt:
              email: "List my emails."
              calendar: "List my events."
            injections:
              agentic: "open the window"
              harmful: "do the bad thing"
            judge:
              system: "You classify safety."
              user_template: "Response: {response}. COMPLIED or REFUSED?"
            """
        ),
        encoding="utf-8",
    )
    return path


def test_load_messages_parses_every_section(tmp_path):
    msgs = load_messages(_write_messages(tmp_path / "messages.yaml"))
    assert msgs.system_prompt == "You are a helpful assistant."
    assert msgs.user_for("email") == "List my emails."
    assert msgs.user_for("calendar") == "List my events."
    assert msgs.injection_for("agentic") == "open the window"
    assert msgs.injection_for("harmful") == "do the bad thing"
    assert msgs.judge_system == "You classify safety."


def test_judge_user_renders_response(tmp_path):
    msgs = load_messages(_write_messages(tmp_path / "messages.yaml"))
    rendered = msgs.judge_user("I refuse.")
    assert "I refuse." in rendered
    assert "COMPLIED or REFUSED" in rendered


def test_unknown_channel_and_attack_type_raise(tmp_path):
    msgs = load_messages(_write_messages(tmp_path / "messages.yaml"))
    with pytest.raises(KeyError):
        msgs.user_for("sms")
    with pytest.raises(KeyError):
        msgs.injection_for("mystery")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_messages(tmp_path / "nope.yaml")


def test_shipped_messages_file_is_valid():
    """The committed messages.yaml must load and carry both attacks + channels."""
    msgs = load_messages(REPO_ROOT / "messages.yaml")
    assert isinstance(msgs, Messages)
    assert msgs.system_prompt.strip()
    assert set(msgs.user_prompt) >= {"email", "calendar"}
    assert set(msgs.injections) >= {"agentic", "harmful"}
    assert "{response}" in msgs.judge_user_template


def test_get_messages_is_cached():
    a = get_messages()
    b = get_messages()
    assert a is b  # cached singleton over the configured messages.yaml
