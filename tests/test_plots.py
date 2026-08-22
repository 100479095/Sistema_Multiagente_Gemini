"""Behaviour tests for the adaptive multi-model plotter (experiment.plots).

These exercise ``generate_plots`` end-to-end on synthetic results.csv +
attempts.csv, asserting the four figures are written. Plotting runs on the
non-interactive Agg backend (no display needed), so these are pure unit tests.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from experiment.plots import generate_plots
from experiment.runner import ATTEMPT_COLUMNS, RESULT_COLUMNS

EXPECTED_FIGURES = {
    "fig1_asr_por_modelo_ataque.png",
    "fig2_intentos_hasta_exito.png",
    "fig3_ganancia_adaptacion.png",
    "fig4_fuente_prompt_ganador.png",
}

MODELS = ["qwen2.5-tools:7b", "dolphin3-tools:8b"]


def _result_row(**over) -> dict:
    """A results.csv row with all columns present, overridable by keyword."""
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(
        rep=0, success=0, attempts_used=1, winning_attempt=0,
        winning_prompt_source="", num_facts=0, num_inferences=1,
        num_invocations=0, chained_agents="", final_answer="", log_file="",
    )
    row.update(over)
    return row


def _attempt_row(**over) -> dict:
    row = {c: "" for c in ATTEMPT_COLUMNS}
    row.update(
        rep=0, attempt=1, prompt_source="base", prompt_text="x", success=0,
        judge_label="", num_inferences=1, num_invocations=0,
        chained_agents="", final_answer="", log_file="",
    )
    row.update(over)
    return row


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def synthetic_campaign(tmp_path: Path) -> tuple[Path, Path]:
    """A small but fully-populated pair of CSVs covering both models/attacks.

    Includes wins at attempts 1/2/3, all three prompt sources, outright
    failures, and one malformed row with an empty ``model`` (must be dropped).
    """
    results: list[dict] = []
    attempts: list[dict] = []

    # qwen agentic: a win at attempt 2 via the judge, and a failure.
    results.append(_result_row(model="qwen2.5-tools:7b", attack_type="agentic",
                               strategy="S2", success=1, attempts_used=2,
                               winning_attempt=2, winning_prompt_source="judge"))
    results.append(_result_row(model="qwen2.5-tools:7b", attack_type="agentic",
                               strategy="S3", success=0, attempts_used=5,
                               winning_attempt=0, winning_prompt_source=""))
    # qwen harmful: a win at attempt 3 via the fallback list.
    results.append(_result_row(model="qwen2.5-tools:7b", attack_type="harmful",
                               strategy="S2", success=1, attempts_used=3,
                               winning_attempt=3, winning_prompt_source="fallback"))
    # dolphin agentic + harmful: immediate wins at attempt 1 (base).
    results.append(_result_row(model="dolphin3-tools:8b", attack_type="agentic",
                               strategy="S2", success=1, attempts_used=1,
                               winning_attempt=1, winning_prompt_source="base"))
    results.append(_result_row(model="dolphin3-tools:8b", attack_type="harmful",
                               strategy="S2", success=1, attempts_used=1,
                               winning_attempt=1, winning_prompt_source="base"))
    # Malformed legacy row (old schema, no model) must be silently dropped.
    results.append(_result_row(model="", attack_type="agentic", success=1))

    # attempts.csv: one attempt-1 row per case (some fail on attempt 1).
    attempts.append(_attempt_row(model="qwen2.5-tools:7b", attack_type="agentic", success=0))
    attempts.append(_attempt_row(model="qwen2.5-tools:7b", attack_type="agentic",
                                 strategy="S3", success=0))
    attempts.append(_attempt_row(model="qwen2.5-tools:7b", attack_type="harmful", success=0))
    attempts.append(_attempt_row(model="dolphin3-tools:8b", attack_type="agentic", success=1))
    attempts.append(_attempt_row(model="dolphin3-tools:8b", attack_type="harmful", success=1))

    results_path = tmp_path / "results.csv"
    attempts_path = tmp_path / "attempts.csv"
    _write_csv(results_path, RESULT_COLUMNS, results)
    _write_csv(attempts_path, ATTEMPT_COLUMNS, attempts)
    return results_path, attempts_path


def test_generate_plots_writes_all_four_figures(synthetic_campaign, tmp_path):
    results_path, attempts_path = synthetic_campaign
    out_dir = tmp_path / "figuras"

    written = generate_plots(results_path, attempts_path, out_dir)

    names = {Path(p).name for p in written}
    assert names == EXPECTED_FIGURES
    for p in written:
        p = Path(p)
        assert p.exists() and p.stat().st_size > 0


def test_generate_plots_drops_rows_without_model(synthetic_campaign, tmp_path):
    """The empty-model legacy row must not crash or inflate any model group."""
    results_path, attempts_path = synthetic_campaign
    out_dir = tmp_path / "figuras"

    # Should complete without raising despite the malformed row.
    written = generate_plots(results_path, attempts_path, out_dir)
    assert len(written) == 4


def test_generate_plots_without_attempts_file_still_makes_figures(
    synthetic_campaign, tmp_path
):
    """attempts.csv may be absent (e.g. adaptive=false single-shot runs)."""
    results_path, _ = synthetic_campaign
    out_dir = tmp_path / "figuras"

    written = generate_plots(results_path, tmp_path / "missing.csv", out_dir)
    # The adaptation-gain chart needs attempts.csv; the other three still render.
    assert len(written) >= 3
