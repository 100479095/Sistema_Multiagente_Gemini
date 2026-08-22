"""Bounded ReAct loop — the core inference engine (PDR §4).

One :meth:`Orchestrator.run` is a single user turn:

1. Build context = system prompt (declares the agents/tools) + user prompt.
2. **Inference.** Ask the model; it either requests tool calls or answers.
3. **Execution.** Dispatch each requested tool; append its result to context
   verbatim (the re-injection of PDR §6 — no filtering).
4. Repeat from (2) with the enriched context until the model returns a final
   answer (no tool calls) or ``max_iterations`` is hit (default 5).

The minimal useful case (one tool + an answer) is exactly two inferences. The
loop records the trace and the metrics §9.3 needs — number of inferences and
invocations, the chain of agents, whether the iteration cap was reached, and the
**Automatic Agent Invocation** signal: whether an agent's (untrusted) output in a
prior iteration led the model to invoke a tool in a later one. Attack *success*
itself is not judged here; the experiment layer reads the home state for that
(PDR §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from config import OrchestratorSettings, get_settings
from llm.client import LLMClient
from orchestrator.dispatcher import DispatchResult, Dispatcher
from orchestrator.memory import ShortTermMemory
from orchestrator.prompt_builder import build_system_prompt
from orchestrator.tool_registry import ToolRegistry
from provenance import Fragment, trusted


@dataclass
class IterationTrace:
    """Everything that happened in one inference of the loop (for §9.2 logs)."""

    index: int
    messages_sent: list[dict[str, Any]]
    response_text: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    dispatch_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunResult:
    """Outcome + metrics of a full run (PDR §9.3)."""

    user_prompt: str
    final_answer: str | None
    iterations: list[IterationTrace]
    num_inferences: int
    num_invocations: int
    max_iterations_reached: bool
    chained_agents: list[str]
    automatic_agent_invocation: bool
    provenance: list[Fragment]

    def full_messages(self) -> list[dict[str, Any]]:
        """The complete conversation the loop built (for the harmful judge).

        Short-term memory grows monotonically, so the *last* iteration's
        ``messages_sent`` is the largest snapshot: the whole conversation up to —
        but not including — the final assistant reply. Appending ``final_answer``
        (when the run terminated with one) reconstructs the full transcript,
        system prompt through final answer. Returns ``[]`` when there were no
        iterations. In the ``max_iterations_reached`` case ``final_answer`` is
        ``None``, so the trailing tool-calling turn is not appended — but there is
        no delivered final answer to score there anyway.
        """
        if not self.iterations:
            return []
        messages = [dict(m) for m in self.iterations[-1].messages_sent]
        if self.final_answer is not None:
            messages.append({"role": "assistant", "content": self.final_answer})
        return messages


class Orchestrator:
    """Drives the bounded ReAct loop over an :class:`LLMClient` and registry."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        settings: OrchestratorSettings | None = None,
        *,
        dispatcher: Dispatcher | None = None,
        on_event: Callable[[DispatchResult], None] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings or get_settings().orchestrator
        self.dispatcher = dispatcher or Dispatcher(registry, on_event=on_event)

    def run(
        self,
        user_prompt: str,
        *,
        seed: int | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> RunResult:
        emit = emit or (lambda event: None)
        system_prompt = build_system_prompt(self.registry)
        memory = ShortTermMemory()
        memory.add_system(system_prompt)
        memory.add_user(user_prompt)
        tools = self.registry.tool_schemas()
        emit(
            {
                "event": "run_start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
            }
        )

        iterations: list[IterationTrace] = []
        num_invocations = 0
        chained_agents: list[str] = []
        automatic_agent_invocation = False
        final_answer: str | None = None
        max_iterations_reached = False
        # Provenance of every context fragment: the prompts are trusted; agent
        # output is tagged untrusted as it arrives (PDR §6.3, logging only).
        provenance: list[Fragment] = [
            trusted(system_prompt, "system"),
            trusted(user_prompt, "user"),
        ]

        for index in range(self.settings.max_iterations):
            messages_sent = memory.snapshot()
            assistant = self.llm.chat(messages_sent, tools=tools, seed=seed)
            memory.add_assistant(assistant)

            serialized_tool_calls = [tc.to_openai() for tc in assistant.tool_calls]
            emit(
                {
                    "event": "inference",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "iteration": index,
                    "messages_sent": messages_sent,
                    "response_text": assistant.content,
                    "tool_calls": serialized_tool_calls,
                    # "length" == the reply hit max_tokens and is cut mid-text.
                    # results.csv stores only the text, where a truncated answer
                    # looks complete; this is the only place it is recorded.
                    "finish_reason": assistant.finish_reason,
                }
            )

            if not assistant.has_tool_calls:
                # Model produced the final answer -> loop terminates (RF-4.2).
                final_answer = assistant.content
                iterations.append(
                    IterationTrace(
                        index=index,
                        messages_sent=messages_sent,
                        response_text=assistant.content,
                    )
                )
                break

            # The model asked to act: dispatch every requested call and feed each
            # result back into context for the next inference.
            dispatch_results: list[dict[str, Any]] = []
            for tool_call in assistant.tool_calls:
                result = self.dispatcher.dispatch(tool_call, iteration=index)
                num_invocations += 1
                if result.agent is not None:
                    chained_agents.append(result.agent)
                memory.add_tool_result(result.to_tool_message(tool_call.id))
                record = result.to_log_record()
                dispatch_results.append(record)
                provenance.extend(result.provenance)
                emit({"event": "tool_result", **record})

            # AAI signal: tool results from a prior iteration are already in
            # context, and at index >= 1 the model chose to invoke again (§9.3).
            if index >= 1:
                automatic_agent_invocation = True

            iterations.append(
                IterationTrace(
                    index=index,
                    messages_sent=messages_sent,
                    response_text=assistant.content,
                    tool_calls=serialized_tool_calls,
                    dispatch_results=dispatch_results,
                )
            )
        else:
            # Loop ran to the cap without a tool-free final answer (RF-4.2).
            max_iterations_reached = True

        emit(
            {
                "event": "run_end",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "final_answer": final_answer,
                "num_inferences": len(iterations),
                "num_invocations": num_invocations,
                "max_iterations_reached": max_iterations_reached,
                "chained_agents": chained_agents,
                "automatic_agent_invocation": automatic_agent_invocation,
                "provenance": [f.to_dict() for f in provenance],
            }
        )

        return RunResult(
            user_prompt=user_prompt,
            final_answer=final_answer,
            iterations=iterations,
            num_inferences=len(iterations),
            num_invocations=num_invocations,
            max_iterations_reached=max_iterations_reached,
            chained_agents=chained_agents,
            automatic_agent_invocation=automatic_agent_invocation,
            provenance=provenance,
        )
