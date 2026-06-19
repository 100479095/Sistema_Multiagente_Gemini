"""CLI / REPL entry point for the multi-agent assistant (the target system).

Run ``python main.py --help`` for available commands. The interactive REPL and
batch/scenario commands are implemented in later phases; this module wires the
``src/`` layout onto ``sys.path`` and exposes the Typer application.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ layout importable when running ``python main.py`` directly.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import typer  # noqa: E402  (import after sys.path bootstrap)

app = typer.Typer(
    add_completion=False,
    help="Local simulated multi-agent assistant (TFG promptware testbed).",
)


@app.command()
def version() -> None:
    """Print the testbed version and the configured orchestrator model."""
    from config import get_settings

    settings = get_settings()
    typer.echo("promptware-testbed 0.1.0")
    typer.echo(f"orchestrator model: {settings.llm.model} @ {settings.llm.base_url}")


@app.command()
def chat(
    scenario: str = typer.Option(
        None, "--scenario", "-s", help="Load data/scenarios/<name>/ before starting."
    ),
    reset: bool = typer.Option(
        False, "--reset", help="Restore the working stores to the benign seeds first."
    ),
    seed: int = typer.Option(
        None, "--seed", help="Deterministic LLM seed (reproducible runs)."
    ),
) -> None:
    """Start the interactive REPL against the local orchestrator (needs Ollama).

    Each line is one independent user turn (the bounded ReAct loop of PDR §4):
    the assistant may call agent tools and then answers. All turns share the
    on-disk home/calendar/mailbox state and one JSONL run log.
    """
    import click

    from app import build_session, load_scenario, reset_data
    from config import get_settings

    settings = get_settings()
    if reset:
        reset_data(settings)
    if scenario:
        load_scenario(scenario, settings)

    orchestrator, logger = build_session(settings)
    typer.echo("Multi-agent assistant REPL — type 'exit' (or Ctrl-D) to quit.")
    try:
        while True:
            try:
                prompt = typer.prompt("you")
            except (EOFError, click.exceptions.Abort):
                break
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            result = orchestrator.run(prompt, seed=seed, emit=logger.emit)
            if result.final_answer:
                typer.echo(f"assistant> {result.final_answer}")
    finally:
        logger.close()
        if logger.path is not None:
            typer.echo(f"\nLog written to: {logger.path}")


@app.command()
def run(
    scenario: str = typer.Argument(
        ..., help="Scenario name under data/scenarios/ to batch-run."
    ),
    reset: bool = typer.Option(
        False, "--reset", help="Restore the working stores to the benign seeds first."
    ),
    seed: int = typer.Option(
        None, "--seed", help="Deterministic LLM seed (reproducible runs)."
    ),
) -> None:
    """Batch-run a scenario's prompts.json through the orchestrator (needs Ollama)."""
    from app import run_scenario

    results = run_scenario(scenario, reset=reset, seed=seed)
    typer.echo(f"Ran {len(results)} prompt(s) from scenario {scenario!r}.")
    for i, result in enumerate(results):
        answer = result.final_answer or "(no final answer)"
        typer.echo(f"  [{i}] {answer}")


if __name__ == "__main__":
    app()
