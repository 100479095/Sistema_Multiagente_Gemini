"""Observability — structured JSONL run logs + a readable rich console (PDR §9).

A :class:`RunLogger` is an *event sink*: the orchestrator calls :meth:`emit` with
one plain dict per event during a run. The logger does two things with it:

* **JSONL** — append the event verbatim as one line to ``logs/run-<ts>.jsonl``
  (RF-9.1, one line per event). Across a run the file captures everything §9.2
  asks for: the user prompt; per iteration the messages sent to the LLM, the
  response text and ``tool_calls``, each tool's agent/args/result and the
  provenance of every fragment; and the final answer plus the §9.3 metrics.
* **Console** — a concise, step-by-step ``rich`` rendering of the same flow
  (RF-9.4) so a human can watch what the orchestrator decided.

Logs are local only (RF-9.5). The logger is decoupled from the orchestrator via
the dict-based sink, so neither imports the other.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from config import Settings, get_settings


def _make_console() -> Console:
    """A rich console that won't crash on non-ASCII output.

    Agent text and model answers (event titles, email bodies, the assistant's
    reply) routinely contain characters a legacy Windows console (cp1252) can't
    encode, which would otherwise raise ``UnicodeEncodeError`` mid-run. Switch
    stdout to UTF-8 with a replacing error handler so observability never takes
    the orchestrator down. No-op where stdout can't be reconfigured.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass
    return Console()


def _preview(value: Any, limit: int = 2000) -> str:
    """JSON-encode ``value`` for the console, truncating very long content.

    Tool results (e.g. ``list_events`` with a large injected payload re-injected
    verbatim) can be huge; cap them so watching a run stays readable. The full,
    untruncated content is always in the JSONL log.
    """
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return f"{text[:limit]}… [{len(text)} chars total]"
    return text


class RunLogger:
    """Write run events to a JSONL file and/or render them to the console."""

    def __init__(
        self,
        logs_dir: str | Path | None = None,
        *,
        jsonl: bool = True,
        console: bool = True,
        console_obj: Console | None = None,
    ) -> None:
        self.path: Path | None = None
        self._fh = None
        if jsonl:
            if logs_dir is None:
                raise ValueError("logs_dir is required when jsonl=True")
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            name = f"run-{stamp}-{uuid.uuid4().hex[:6]}.jsonl"
            self.path = Path(logs_dir) / name
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

        if console_obj is not None:
            self.console: Console | None = console_obj
        elif console:
            self.console = _make_console()
        else:
            self.console = None

    # -- factory ------------------------------------------------------------ #

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, **overrides: Any
    ) -> "RunLogger":
        settings = settings or get_settings()
        kwargs: dict[str, Any] = {
            "logs_dir": settings.paths.resolve(settings.paths.logs_dir),
            "jsonl": settings.logging.jsonl,
            "console": settings.logging.console,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # -- sink --------------------------------------------------------------- #

    def emit(self, event: dict[str, Any]) -> None:
        """Record one run event (called by the orchestrator)."""
        if self._fh is not None:
            self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        if self.console is not None:
            self._render(event)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- console rendering -------------------------------------------------- #

    def _render(self, event: dict[str, Any]) -> None:
        assert self.console is not None
        kind = event.get("event")
        if kind == "run_start":
            self.console.rule("[bold]Run start")
            self.console.print(f"[bold cyan]User:[/] {event.get('user_prompt', '')}")
        elif kind == "inference":
            idx = event.get("iteration")
            self.console.print(f"\n[bold]Inference {idx}[/]")
            text = event.get("response_text")
            if text:
                self.console.print(f"  [white]model:[/] {text}")
            for tc in event.get("tool_calls", []):
                fn = tc.get("function", {})
                self.console.print(
                    f"  [yellow]→ tool_call:[/] {fn.get('name')}({fn.get('arguments', '')})"
                )
        elif kind == "tool_result":
            status = "[green]ok[/]" if event.get("ok") else "[red]error[/]"
            self.console.print(
                f"  [magenta]{event.get('agent')}.{event.get('tool_name')}[/] {status}"
            )
            if event.get("ok"):
                # The verbatim content fed back to the model for the next
                # inference — the re-injection surface under study (PDR §6).
                self.console.print(f"    [dim]returns:[/] {_preview(event.get('result'))}")
            untrusted = [
                p for p in event.get("provenance", []) if p.get("provenance") == "untrusted"
            ]
            if untrusted:
                self.console.print(
                    f"    [dim]untrusted fragments re-injected: {len(untrusted)}[/]"
                )
            if not event.get("ok"):
                self.console.print(f"    [red]{event.get('error')}[/]")
        elif kind == "run_end":
            self.console.rule("[bold]Run end")
            if event.get("final_answer"):
                self.console.print(f"[bold green]Assistant:[/] {event['final_answer']}")
            self.console.print(
                "[dim]"
                f"inferences={event.get('num_inferences')} "
                f"invocations={event.get('num_invocations')} "
                f"chain={event.get('chained_agents')} "
                f"AAI={event.get('automatic_agent_invocation')} "
                f"max_iter_reached={event.get('max_iterations_reached')}"
                "[/]"
            )
