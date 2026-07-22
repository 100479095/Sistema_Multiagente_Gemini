"""Build the orchestrator's system prompt (PDR §4.1).

The system prompt sets a neutral, realistic personal-assistant persona and
declares the agents and their tools, mirroring a Gemini-style assistant. The
tools are *also* passed structurally via the OpenAI ``tools=`` field; listing
them here too is faithful to the PDR's inference model.

The persona text (the preamble) lives in the central ``messages.yaml`` so the
researcher edits every prompt from one place (:mod:`messages`); this module only
renders it and appends the *live* tool listing derived from the registry.

Design note (experimental validity): the prompt deliberately contains **no
hardened anti-injection instructions**. The research question is whether the
model obeys injected instructions arriving through agent output (PDR §7.1);
baking in defenses would confound that measurement. It is a plain helpful
assistant, nothing more, nothing less.
"""

from __future__ import annotations

from messages import get_messages
from orchestrator.tool_registry import ToolRegistry


def _tools_by_agent(registry: ToolRegistry) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for name in registry.names():
        rt = registry.resolve(name)
        grouped.setdefault(rt.agent.name, []).append((name, rt.spec.description))
    return grouped


def build_system_prompt(registry: ToolRegistry) -> str:
    """Render the system prompt declaring every agent and its tools."""
    preamble = get_messages().system_prompt.strip()
    lines = [preamble, "", "Available tools:"]
    for agent_name, tools in _tools_by_agent(registry).items():
        lines.append(f"\n[{agent_name} agent]")
        for tool_name, description in tools:
            lines.append(f"- {tool_name}: {description}")
    return "\n".join(lines)
