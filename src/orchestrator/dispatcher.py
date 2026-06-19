"""Dispatcher: execute a model-requested :class:`ToolCall` against the registry.

For one tool call the dispatcher (1) resolves the name to the owning agent,
(2) validates the arguments with the tool's pydantic model, (3) runs the
callable, and (4) returns a :class:`DispatchResult` carrying everything the
orchestrator and the logger need: tool name, agent, validated args, the result
(or a clean error), the iteration index and a UTC timestamp (PDR §9.2).

**Robustness is deliberate.** An unknown tool name, invalid arguments, or an
exception raised *inside* a tool are all turned into an ``ok=False`` result with
a readable error message instead of propagating. The bounded ReAct loop (Fase 4)
and the experiment runner (Fase 9) must survive a malformed or hallucinated tool
call mid-run rather than crashing — the model is the untrusted decision-maker
here, so its mistakes are data, not fatal errors (PDR §4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import ValidationError

from llm.client import ToolCall
from orchestrator.tool_registry import ToolRegistry
from provenance import Fragment, tag_tool_result


@dataclass
class DispatchResult:
    """Structured outcome of executing one tool call."""

    tool_name: str
    arguments: dict[str, Any]
    iteration: int
    timestamp: str
    ok: bool
    agent: str | None = None
    result: Any | None = None
    error: str | None = None
    provenance: list[Fragment] = field(default_factory=list)

    @property
    def payload(self) -> dict[str, Any]:
        """The compact result surfaced back to the model for the next inference."""
        if self.ok:
            return {"ok": True, "result": self.result}
        return {"ok": False, "error": self.error}

    def to_tool_message(self, tool_call_id: str) -> dict[str, Any]:
        """OpenAI ``tool``-role message re-injecting this result into context.

        The result is serialised and inserted **verbatim**, with no
        sanitisation: untrusted agent output (event titles, email bodies) flows
        straight back into the orchestrator context. This is precisely the attack
        surface under study (PDR §6). Provenance tagging (Fase 5) annotates the
        fragment for the log but never filters it.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": self.tool_name,
            "content": json.dumps(self.payload, ensure_ascii=False),
        }

    def to_log_record(self) -> dict[str, Any]:
        """Flat dict for the JSONL run log (Fase 6)."""
        return {
            "tool_name": self.tool_name,
            "agent": self.agent,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "provenance": [f.to_dict() for f in self.provenance],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Dispatcher:
    """Resolve, validate and execute tool calls against a :class:`ToolRegistry`."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        on_event: Callable[[DispatchResult], None] | None = None,
    ) -> None:
        self.registry = registry
        self._on_event = on_event

    def dispatch(self, tool_call: ToolCall, *, iteration: int = 0) -> DispatchResult:
        """Execute one tool call, never raising on bad input."""
        ts = _utc_now()
        registered = self.registry.get(tool_call.name)

        if registered is None:
            return self._finish(
                DispatchResult(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    iteration=iteration,
                    timestamp=ts,
                    ok=False,
                    error=f"Unknown tool {tool_call.name!r}",
                )
            )

        spec = registered.spec
        agent = registered.agent

        try:
            validated = spec.validate_arguments(tool_call.arguments)
        except ValidationError as exc:
            return self._finish(
                DispatchResult(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    iteration=iteration,
                    timestamp=ts,
                    ok=False,
                    agent=agent.name,
                    error=f"Invalid arguments for {tool_call.name!r}: {exc}",
                )
            )

        try:
            method = getattr(agent, spec.func_name)
            result = method(**validated)
        except Exception as exc:  # noqa: BLE001 - keep the ReAct loop alive
            return self._finish(
                DispatchResult(
                    tool_name=tool_call.name,
                    arguments=validated,
                    iteration=iteration,
                    timestamp=ts,
                    ok=False,
                    agent=agent.name,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

        # Tag the untrusted text this result carries (for the log only; the
        # content is still re-injected verbatim — PDR §6.3).
        provenance = tag_tool_result(tool_call.name, result, spec.untrusted_fields)
        return self._finish(
            DispatchResult(
                tool_name=tool_call.name,
                arguments=validated,
                iteration=iteration,
                timestamp=ts,
                ok=True,
                agent=agent.name,
                result=result,
                provenance=provenance,
            )
        )

    def _finish(self, result: DispatchResult) -> DispatchResult:
        if self._on_event is not None:
            self._on_event(result)
        return result
