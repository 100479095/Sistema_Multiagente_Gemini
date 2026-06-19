"""LLM access layer.

Wraps the orchestrator model (Qwen 2.5 7B served by Ollama) behind a small,
backend-agnostic interface so the rest of the system never imports the OpenAI
SDK directly. Swapping Ollama for vLLM or transformers means changing only this
module (PDR §5.2).

Uses Ollama's OpenAI-compatible endpoint and **native structured tool-calling**:
``chat(messages, tools)`` returns an :class:`AssistantMessage` carrying free text
and/or parsed :class:`ToolCall` objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from config import LLMSettings, get_settings


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        """Serialize back to the OpenAI ``tool_calls`` wire format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class AssistantMessage:
    """The assistant's reply for one inference: text and/or tool calls."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_openai(self) -> dict[str, Any]:
        """Serialize as an OpenAI ``assistant`` message for the next inference."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Parse a tool-call ``arguments`` payload into a dict.

    The model returns arguments as a JSON string; be defensive about empty or
    malformed payloads so a single bad call doesn't crash the orchestrator.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _iter_json_objects(text: str):
    """Yield top-level ``{...}`` substrings from ``text`` via brace matching.

    Tolerates nested objects (e.g. an ``arguments`` sub-object) and surrounding
    prose / tags.
    """
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def extract_text_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Recover tool calls that the model emitted as plain text.

    Qwen models render tool calls as ``<tool_call>{"name": ..., "arguments":
    {...}}</tool_call>``. Ollama's OpenAI-compatible endpoint usually parses
    these into structured ``tool_calls``, but occasionally (e.g. a mangled
    opening tag) leaves them in the message content. This fallback scans the
    content for JSON objects carrying a ``name`` key and turns them into
    :class:`ToolCall` objects, returning the cleaned-up residual text.
    """
    calls: list[ToolCall] = []
    cleaned = content
    for idx, obj_str in enumerate(_iter_json_objects(content)):
        try:
            obj = json.loads(obj_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            args = _parse_arguments(args)
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(id=f"text_call_{idx}", name=obj["name"], arguments=args))
        cleaned = cleaned.replace(obj_str, "", 1)
    cleaned = cleaned.replace("<tool_call>", "").replace("</tool_call>", "").strip()
    return calls, cleaned


class LLMClient:
    """Thin client over the Ollama OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        openai_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings().llm
        self._client = openai_client  # lazily created if not injected (testable)

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key,
                timeout=self.settings.timeout_s,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> AssistantMessage:
        """Run one inference and return the parsed assistant message.

        ``messages`` follow the OpenAI chat format. ``tools`` is the OpenAI tool
        schema list (or ``None`` to disable tool-calling for this call). Per-call
        ``temperature``/``seed`` override the configured defaults (used by the
        experiment runner to vary the seed per repetition).
        """
        client = self._ensure_client()

        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.settings.temperature
            ),
            "top_p": self.settings.top_p,
        }
        effective_seed = seed if seed is not None else self.settings.seed
        if effective_seed is not None:
            kwargs["seed"] = effective_seed
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: Any) -> AssistantMessage:
        """Convert an OpenAI-style response object into an AssistantMessage."""
        message = response.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_arguments(tc.function.arguments),
                )
            )
        content = getattr(message, "content", None)

        # Fallback: the model emitted the tool call as text instead of structured
        # tool_calls (a known Qwen/Ollama quirk). Recover it so the orchestrator
        # still sees the model's decision to act.
        if not tool_calls and content:
            recovered, cleaned = extract_text_tool_calls(content)
            if recovered:
                tool_calls = recovered
                content = cleaned or None

        return AssistantMessage(content=content, tool_calls=tool_calls)
