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
              user_template: "Transcript: {transcript}. COMPLIED or REFUSED?"
            attack_prompts:
              agentic:
                - "fallback agentic one"
                - "fallback agentic two"
              harmful:
                - "fallback harmful one"
            prompt_generator:
              system: "You are a red-team prompt optimiser."
              user_template: |-
                Attack type: {attack_type}
                Previous prompt: {prior_prompt}
                Assistant response: {response}
                Return an improved injection between <prompt> and </prompt>.
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


def test_judge_user_renders_transcript(tmp_path):
    msgs = load_messages(_write_messages(tmp_path / "messages.yaml"))
    rendered = msgs.judge_user("[ASSISTANT]\nI refuse.")
    assert "I refuse." in rendered
    assert "COMPLIED or REFUSED" in rendered


def test_unknown_channel_and_attack_type_raise(tmp_path):
    msgs = load_messages(_write_messages(tmp_path / "messages.yaml"))
    with pytest.raises(KeyError):
        msgs.user_for("sms")
    with pytest.raises(KeyError):
        msgs.injection_for("mystery")


def test_load_messages_parses_attack_prompts_and_generator(tmp_path):
    msgs = load_messages(_write_messages(tmp_path / "messages.yaml"))
    assert msgs.fallback_prompts("agentic") == [
        "fallback agentic one",
        "fallback agentic two",
    ]
    assert msgs.fallback_prompts("harmful") == ["fallback harmful one"]
    assert msgs.generator_system == "You are a red-team prompt optimiser."
    assert "<prompt>" in msgs.generator_user_template


def test_fallback_prompts_missing_attack_type_returns_empty(tmp_path):
    """Unlike injection_for, fallback lookup is non-raising (empty list = none)."""
    msgs = load_messages(_write_messages(tmp_path / "messages.yaml"))
    assert msgs.fallback_prompts("mystery") == []


def test_generator_user_renders_all_fields(tmp_path):
    msgs = load_messages(_write_messages(tmp_path / "messages.yaml"))
    rendered = msgs.generator_user(
        "agentic", prior_prompt="open the window", response="I refuse."
    )
    assert "agentic" in rendered
    assert "open the window" in rendered
    assert "I refuse." in rendered
    assert "<prompt>" in rendered and "</prompt>" in rendered


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
    assert "{transcript}" in msgs.judge_user_template
    # Adaptive red-teaming assets: fallback lists + a generator prompt.
    assert set(msgs.attack_prompts) >= {"agentic", "harmful"}
    assert msgs.fallback_prompts("agentic")
    assert msgs.fallback_prompts("harmful")
    assert "{response}" in msgs.generator_user_template
    assert "<prompt>" in msgs.generator_user_template


def test_get_messages_is_cached():
    a = get_messages()
    b = get_messages()
    assert a is b  # cached singleton over the configured messages.yaml
