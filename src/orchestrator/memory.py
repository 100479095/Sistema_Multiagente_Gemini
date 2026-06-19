"""Short-term memory for one orchestrator run (PDR §4.3).

A thin wrapper around the running OpenAI-format message list that the ReAct loop
sends to the model each inference. It grows as the loop progresses: the system
prompt and user prompt first, then each assistant reply and each tool result.

Crucially, tool results (untrusted agent output) are appended **as-is** and fed
back into the next inference verbatim — this module performs no filtering. That
re-injection is the attack surface under study (PDR §6); provenance tagging
(Fase 5) annotates fragments for the log but never removes them here.
"""

from __future__ import annotations

from typing import Any

from llm.client import AssistantMessage


class ShortTermMemory:
    """The conversation state passed to the LLM during a single run."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_system(self, content: str) -> None:
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, message: AssistantMessage) -> None:
        """Record the model's reply (text and/or tool calls) for the next turn."""
        self.messages.append(message.to_openai())

    def add_tool_result(self, message: dict[str, Any]) -> None:
        """Append a ``tool``-role message (an agent result re-injected verbatim)."""
        self.messages.append(message)

    def snapshot(self) -> list[dict[str, Any]]:
        """A copy of the current message list (what gets sent to the LLM now)."""
        return [dict(m) for m in self.messages]
