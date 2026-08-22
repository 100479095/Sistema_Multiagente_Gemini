"""Tests for the LLM access layer (src/llm/client.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from config import LLMSettings
from llm.client import (
    AssistantMessage,
    LLMClient,
    ToolCall,
    extract_text_tool_calls,
)


class RecordingClient:
    """Fake OpenAI client that records the request and returns a canned reply."""

    def __init__(self, response):
        self._response = response
        self.last_kwargs = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


def _response(content=None, tool_calls=None, finish_reason="stop"):
    """Build an object shaped like an OpenAI ChatCompletion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_chat_builds_expected_request_payload():
    settings = LLMSettings(model="qwen2.5-tools:7b", temperature=0.7, top_p=0.9, seed=None)
    fake = RecordingClient(_response(content="hi"))
    client = LLMClient(settings=settings, openai_client=fake)

    tools = [{"type": "function", "function": {"name": "list_events"}}]
    client.chat([{"role": "user", "content": "events?"}], tools=tools, seed=123)

    kwargs = fake.last_kwargs
    assert kwargs["model"] == "qwen2.5-tools:7b"
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["seed"] == 123  # per-call seed overrides config
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


def test_chat_omits_tools_and_seed_when_not_provided():
    settings = LLMSettings(seed=None)
    fake = RecordingClient(_response(content="plain answer"))
    client = LLMClient(settings=settings, openai_client=fake)

    client.chat([{"role": "user", "content": "hello"}])

    kwargs = fake.last_kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "seed" not in kwargs


def test_chat_sends_the_configured_generation_cap():
    settings = LLMSettings(max_tokens=1024)
    fake = RecordingClient(_response(content="hi"))
    client = LLMClient(settings=settings, openai_client=fake)

    client.chat([{"role": "user", "content": "hello"}])

    assert fake.last_kwargs["max_tokens"] == 1024


def test_chat_omits_max_tokens_when_uncapped():
    settings = LLMSettings(max_tokens=None)
    fake = RecordingClient(_response(content="hi"))
    client = LLMClient(settings=settings, openai_client=fake)

    client.chat([{"role": "user", "content": "hello"}])

    assert "max_tokens" not in fake.last_kwargs


def test_chat_per_call_max_tokens_overrides_the_configured_cap():
    settings = LLMSettings(max_tokens=1024)
    fake = RecordingClient(_response(content="hi"))
    client = LLMClient(settings=settings, openai_client=fake)

    client.chat([{"role": "user", "content": "hello"}], max_tokens=64)

    assert fake.last_kwargs["max_tokens"] == 64


def test_openai_client_gets_timeout_and_retry_budget_from_settings(monkeypatch):
    """The wall-clock budget of an inference lives in the SDK client.

    ``max_retries`` matters as much as ``timeout``: the SDK retries timeouts by
    default (2 extra attempts), so one slow inference burnt ``3 x timeout_s``
    before surfacing the error.
    """
    import openai

    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    client = LLMClient(settings=LLMSettings(timeout_s=1200, max_retries=0))

    client._ensure_client()

    assert captured["timeout"] == 1200
    assert captured["max_retries"] == 0


def test_default_generation_budget_is_the_campaign_budget():
    """The class defaults must match the sizing derived in config.yaml.

    ``config.yaml`` is what the campaign actually reads, but anything building
    ``LLMSettings`` directly (probes, scripts, ad-hoc analysis) falls back here,
    and a stale default would silently run under a different budget.
    """
    settings = LLMSettings()

    assert settings.max_tokens == 2048  # covers the longest generation observed
    assert settings.timeout_s == 1800   # worst case ~1100 s + a cold model load
    assert settings.max_retries == 0    # a timeout is "too slow", not "transient"


def test_chat_records_finish_reason_stop_on_a_complete_reply():
    fake = RecordingClient(_response(content="a complete answer", finish_reason="stop"))
    client = LLMClient(settings=LLMSettings(), openai_client=fake)

    reply = client.chat([{"role": "user", "content": "hello"}])

    assert reply.finish_reason == "stop"
    assert reply.truncated is False


def test_chat_flags_a_reply_cut_off_by_the_generation_cap():
    """A capped answer must be distinguishable from a complete one.

    Without this the truncation is invisible: a reply cut at ``max_tokens``
    looks exactly like a finished one in results.csv, and the harmful-attack
    judge would score an answer whose tail never arrived.
    """
    fake = RecordingClient(_response(content="an answer cut mid-", finish_reason="length"))
    client = LLMClient(settings=LLMSettings(), openai_client=fake)

    reply = client.chat([{"role": "user", "content": "hello"}])

    assert reply.finish_reason == "length"
    assert reply.truncated is True


def test_finish_reason_is_not_sent_back_on_the_wire():
    """``finish_reason`` is instrumentation, not part of the assistant message."""
    reply = AssistantMessage(content="hi", finish_reason="length")

    assert "finish_reason" not in reply.to_openai()


def test_parses_tool_calls_with_json_string_arguments():
    args = json.dumps({"room": "living_room"})
    fake = RecordingClient(
        _response(content=None, tool_calls=[_tool_call("c1", "open_window", args)])
    )
    client = LLMClient(settings=LLMSettings(), openai_client=fake)

    result = client.chat([{"role": "user", "content": "x"}])

    assert isinstance(result, AssistantMessage)
    assert result.has_tool_calls
    assert result.tool_calls[0] == ToolCall(
        id="c1", name="open_window", arguments={"room": "living_room"}
    )


def test_parse_response_tolerates_malformed_arguments():
    fake = RecordingClient(
        _response(tool_calls=[_tool_call("c1", "set_boiler", "not-json")])
    )
    client = LLMClient(settings=LLMSettings(), openai_client=fake)

    result = client.chat([{"role": "user", "content": "x"}])

    assert result.tool_calls[0].arguments == {}


def test_assistant_message_roundtrips_to_openai_format():
    msg = AssistantMessage(
        content="",
        tool_calls=[ToolCall("c1", "open_window", {"room": "bedroom"})],
    )
    wire = msg.to_openai()
    assert wire["role"] == "assistant"
    assert wire["tool_calls"][0]["function"]["name"] == "open_window"
    assert json.loads(wire["tool_calls"][0]["function"]["arguments"]) == {
        "room": "bedroom"
    }


def test_text_fallback_recovers_tool_call_with_mangled_opening_tag():
    # Reproduces the observed live failure: opening tag mangled, JSON intact.
    content = (
        'portun\n{"name": "open_window", "arguments": {"room": "living room"}}\n'
        "</tool_call>"
    )
    fake = RecordingClient(_response(content=content))
    client = LLMClient(settings=LLMSettings(), openai_client=fake)

    result = client.chat([{"role": "user", "content": "x"}])

    assert result.has_tool_calls
    assert result.tool_calls[0].name == "open_window"
    assert result.tool_calls[0].arguments == {"room": "living room"}


def test_text_fallback_ignores_plain_json_without_name():
    calls, cleaned = extract_text_tool_calls('here is data {"room": "kitchen"}')
    assert calls == []
    assert "kitchen" in cleaned


def test_text_fallback_strips_tool_response_tags():
    # Observed live with dolphin3-tools:8b, which wraps its call in the *result*
    # tag instead of <tool_call>. Recovering the JSON but leaving the tags behind
    # put "<tool_response></tool_response>" into the assistant's own history.
    content = (
        '<tool_response>\n{"name": "read_email", "arguments": {"email_id": "m1"}}\n'
        "</tool_response>"
    )
    calls, cleaned = extract_text_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "read_email"
    assert calls[0].arguments == {"email_id": "m1"}
    assert cleaned == ""


def test_text_fallback_handles_clean_tool_call_tags():
    content = '<tool_call>{"name": "set_boiler", "arguments": {"state": "on"}}</tool_call>'
    calls, cleaned = extract_text_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "set_boiler"
    assert calls[0].arguments == {"state": "on"}
    assert cleaned == ""


@pytest.mark.integration
def test_live_qwen_returns_tool_call():
    """Smoke test against a live Ollama server (requires qwen2.5-tools:7b)."""
    client = LLMClient()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "open_window",
                "description": "Open a window in a given room of the home.",
                "parameters": {
                    "type": "object",
                    "properties": {"room": {"type": "string"}},
                    "required": ["room"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You control a smart home. Use the provided tools when asked.",
        },
        {"role": "user", "content": "Please open the window in the living room."},
    ]
    result = client.chat(messages, tools=tools, seed=1)
    assert result.has_tool_calls
    assert result.tool_calls[0].name == "open_window"
