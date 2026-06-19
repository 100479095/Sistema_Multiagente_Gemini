"""Tool registry: aggregate every agent's ``@tool`` into one LLM-facing catalog.

The registry walks the supplied agents, collects the :class:`ToolSpec` of each
tool they expose, and provides two things the orchestrator needs:

* :meth:`ToolRegistry.tool_schemas` — the OpenAI tool-schema list passed to
  :meth:`LLMClient.chat` so the model knows which tools exist.
* :meth:`ToolRegistry.resolve` — map a tool name the model asked for back to the
  owning agent instance + spec, so the dispatcher can validate and execute it.

Tool names must be unique across all agents (the LLM addresses a tool by name
only); a collision is a wiring bug and raises at construction time (PDR §8.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from agents.base import Agent, ToolSpec


@dataclass(frozen=True)
class RegisteredTool:
    """One tool, bound to the agent instance that implements it."""

    agent: Agent
    spec: ToolSpec


class ToolRegistry:
    """Catalog of every tool exposed by a set of agents, keyed by tool name."""

    def __init__(self, agents: Iterable[Agent]) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        for agent in agents:
            for spec in agent.tool_specs():
                existing = self._tools.get(spec.name)
                if existing is not None:
                    raise ValueError(
                        f"Duplicate tool name {spec.name!r}: defined by both "
                        f"{existing.agent.name!r} and {agent.name!r}"
                    )
                self._tools[spec.name] = RegisteredTool(agent=agent, spec=spec)

    # -- introspection ------------------------------------------------------ #

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        """Every registered tool name (in registration order)."""
        return list(self._tools)

    # -- lookup ------------------------------------------------------------- #

    def get(self, name: str) -> RegisteredTool | None:
        """Return the bound tool, or ``None`` if no tool by that name exists."""
        return self._tools.get(name)

    def resolve(self, name: str) -> RegisteredTool:
        """Return the bound tool, raising :class:`KeyError` if unknown."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool {name!r}") from None

    # -- LLM-facing schema -------------------------------------------------- #

    def tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI tool-schema list for :meth:`LLMClient.chat`."""
        return [rt.spec.json_schema() for rt in self._tools.values()]
