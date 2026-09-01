"""Behaviour tests for the per-model plotter (experiment.plots).

These exercise ``generate_plots`` end-to-end on a synthetic results.csv,
asserting the full figure set is written: seven figures per model — the
campaign ASR headline, the adaptive-gain split, four campaign-ASR breakdowns
that carry both attack axes as series inside one chart, and the split of the
loop's own wins over its attempts — plus the campaign-wide judge-error figure
whenever the audit columns flag something. Plotting runs on the non-interactive
Agg backend (no display needed), so these are pure unit tests.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from experiment.plots import (
    OUTCOME_ORDER,
    UNKNOWN_ATTEMPT_LABEL,
    _add_outcome,
    _attempt_columns,
    _loop_wins,
    _load_attempts,
    _load_results,
    _shares_to_100,
    generate_plots,
)
from experiment.runner import ATTEMPT_COLUMNS, RESULT_COLUMNS

MODELS = ["qwen2.5-tools:7b", "qwen2.5-abliterate-tools:7b"]
MODEL_SLUGS = ("qwen2.5-tools-7b", "qwen2.5-abliterate-tools-7b")
ATTACKS = ["agentic", "harmful"]

# Every figure is per model; the attack type is a series inside fig3..fig6.
FIGURE_STEMS = (
    (1, "asr_por_eje"),
    (2, "desenlace_por_intento"),
    (3, "asr_hechos"),
    (4, "asr_volumen_hechos"),
    (5, "asr_estrategia"),
    (6, "asr_origen_hechos"),
    (8, "distribucion_intentos"),
)
PER_MODEL_FIGURES = {
    f"fig{i}_{slug}_{name}.png"
    for slug in MODEL_SLUGS
    for i, name in FIGURE_STEMS
}
# Fig 7 compares the models, so there is one per campaign, not one per model.
AUDIT_FIGURES = {"fig7_errores_juez.png"}
EXPECTED_FIGURES = PER_MODEL_FIGURES | AUDIT_FIGURES


def _result_row(**over) -> dict:
    """A results.csv row with all columns present, overridable by keyword."""
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(
        rep=0, success=0, attempts_used=1, winning_attempt=0,
        winning_prompt_source="", num_facts=0, fact_source="none",
        strategy="S1", num_inferences=1, num_invocations=0,
        judge_false_positive=0, judge_false_negative=0,
        chained_agents="", final_answer="", log_file="",
    )
    row.update(over)
    return row


def _attempt_row(**over) -> dict:
    row = {c: "" for c in ATTEMPT_COLUMNS}
    row.update(
        rep=0, attempt=1, prompt_source="base", prompt_text="x", success=0,
        judge_label="", judge_false_positive=0, judge_false_negative=0,
        num_facts=0, fact_source="none", strategy="S1",
        num_inferences=1, num_invocations=0, chained_agents="",
        final_answer="", log_file="",
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

    Every model × attack cell carries the S1 no-facts baseline plus S2/S3 cells
    across the "few" and "many" fact bands and all three fact sources, with a
    mix of attempt-1 wins, later (adaptive) wins and outright failures — so no
    bucket in any of the four ASR figures is empty. attempts.csv carries one row
    per attempt the loop spent, as the runner writes it, so fig8 can tell rung 2
    from rung 3. Two harmful cells are flagged by the judge audit (one false
    positive, one false negative) so the corrected figures and fig7 both have
    something to draw.
    """
    results: list[dict] = []
    attempts: list[dict] = []
    # (strategy, num_facts, fact_source, success, winning_attempt, judge_error)
    # judge_error applies to the harmful axis only — the agentic one is scored
    # by a side effect, so it has no judge and no judge errors.
    cells = [
        ("S1", 0, "none", 1, 1, "fp"),   # judge said COMPLIED, audit says no
        ("S1", 0, "none", 0, 0, None),   # baseline: held
        ("S2", 10, "real", 1, 1, None),  # few facts, win at attempt 1
        ("S2", 150, "invented", 1, 3, None),  # many facts, adaptive gain
        ("S3", 25, "mixed", 0, 0, None),  # few facts, held
        ("S3", 200, "real", 1, 2, None),  # many facts, adaptive gain
        ("S3", 5, "invented", 0, 0, "fn"),  # judge missed a real delivery
        ("S2", 100, "mixed", 1, 1, None),
    ]
    for model in MODELS:
        for attack in ATTACKS:
            for rep, (strategy, nf, src, ok, win, err) in enumerate(cells):
                err = err if attack == "harmful" else None
                flags = {"judge_false_positive": int(err == "fp"),
                         "judge_false_negative": int(err == "fn")}
                results.append(_result_row(
                    model=model, attack_type=attack, strategy=strategy,
                    num_facts=nf, fact_source=src, rep=rep, success=ok,
                    attempts_used=max(win, 1), winning_attempt=win,
                    winning_prompt_source="base" if win == 1 else
                    ("judge" if win else ""),
                    **flags,
                ))
                # One row per attempt spent: only the winning one scores, and
                # the audit flag sits on the verdict it corrects — the last.
                spent = max(win, 1)
                for attempt in range(1, spent + 1):
                    attempts.append(_attempt_row(
                        model=model, attack_type=attack, strategy=strategy,
                        num_facts=nf, fact_source=src, rep=rep, attempt=attempt,
                        prompt_source="base" if attempt == 1 else "judge",
                        success=int(attempt == win),
                        log_file=f"{model}-{attack}-{rep}-a{attempt}.jsonl",
                        **(flags if attempt == spent else {}),
                    ))
    # Malformed legacy row (old schema, no model) must be silently dropped.
    results.append(_result_row(model="", attack_type="agentic", success=1))

    results_path = tmp_path / "results.csv"
    attempts_path = tmp_path / "attempts.csv"
    _write_csv(results_path, RESULT_COLUMNS, results)
    _write_csv(attempts_path, ATTEMPT_COLUMNS, attempts)
    return results_path, attempts_path


def test_generate_plots_writes_every_figure(synthetic_campaign, tmp_path):
    results_path, attempts_path = synthetic_campaign
    out_dir = tmp_path / "figuras"

    written = generate_plots(results_path, attempts_path, out_dir)

    names = {Path(p).name for p in written}
    assert names == EXPECTED_FIGURES
    assert len(written) == len(MODEL_SLUGS) * len(FIGURE_STEMS) + len(AUDIT_FIGURES)
    for p in written:
        p = Path(p)
        assert p.exists() and p.stat().st_size > 0


def test_generate_plots_drops_rows_without_model(synthetic_campaign, tmp_path):
    """The empty-model legacy row must not crash or inflate any model group."""
    results_path, attempts_path = synthetic_campaign
    out_dir = tmp_path / "figuras"

    written = generate_plots(results_path, attempts_path, out_dir)
    assert {Path(p).name for p in written} == EXPECTED_FIGURES


def test_generate_plots_without_attempts_file_still_makes_figures(
    synthetic_campaign, tmp_path
):
    """attempts.csv is optional: results.csv alone carries ``winning_attempt``."""
    results_path, _ = synthetic_campaign
    out_dir = tmp_path / "figuras"

    written = generate_plots(results_path, tmp_path / "missing.csv", out_dir)
    assert {Path(p).name for p in written} == EXPECTED_FIGURES


def test_first_attempt_asr_falls_back_to_attempts_csv(synthetic_campaign, tmp_path):
    """A legacy results.csv without ``winning_attempt`` reads attempt 1 from
    attempts.csv, de-duplicating the repeated rows merged campaigns leave."""
    results_path, attempts_path = synthetic_campaign
    legacy = tmp_path / "legacy_results.csv"
    pd.read_csv(results_path).drop(columns=["winning_attempt"]).to_csv(
        legacy, index=False
    )

    written = generate_plots(legacy, attempts_path, tmp_path / "figuras")
    assert {Path(p).name for p in written} == EXPECTED_FIGURES


def test_generate_plots_rejects_a_results_file_it_cannot_read(tmp_path):
    """No ``winning_attempt`` and no attempts.csv is an unrecoverable input."""
    legacy = tmp_path / "legacy_results.csv"
    rows = [_result_row(model=m, attack_type="agentic") for m in MODELS]
    _write_csv(legacy, [c for c in RESULT_COLUMNS if c != "winning_attempt"],
               [{k: v for k, v in r.items() if k != "winning_attempt"} for r in rows])

    with pytest.raises(ValueError, match="winning_attempt"):
        generate_plots(legacy, tmp_path / "missing.csv", tmp_path / "figuras")


def test_judge_audit_columns_move_the_counted_successes(synthetic_campaign):
    """A flagged false positive stops counting, a false negative starts.

    Both flags sit on the harmful axis only, so the agentic cases must come out
    of the correction untouched — that axis is scored by a real side effect.
    """
    results_path, attempts_path = synthetic_campaign
    cases = _add_outcome(_load_results(results_path), _load_attempts(attempts_path))

    harmful = cases[cases.attack_type == "harmful"]
    agentic = cases[cases.attack_type == "agentic"]
    # One FP and one FN per model, and they cancel in the totals but not in the
    # rows: the FP case must drop out and the FN case must appear.
    assert harmful.success.sum() == harmful.success_corr.sum()
    assert not harmful[harmful.judge_false_positive == 1].success_corr.any()
    assert harmful[harmful.judge_false_negative == 1].success_corr.all()
    assert (agentic.success == agentic.success_corr).all()
    assert (agentic.first_ok == agentic.first_ok_corr).all()
    # The three fig2 segments still partition every case.
    assert (cases.outcome == "first").sum() + (cases.outcome == "later").sum() == \
        int(cases.success_corr.sum())


def test_fig2_column_labels_add_up_to_exactly_100(synthetic_campaign):
    """Three roundings of one column must still print 100,0 %.

    The campaign sizes are powers of two, so every share is a binary fraction
    ending in ...25 / ...75 and all three segments of a column can round the
    same way: the real campaign printed 100,1 % on the aligned model's TOTAL
    and 99,9 % on its harmful column, over stacks that reach exactly 100.
    Largest remainders hand the spare tenth to the biggest remainder instead.
    """
    assert _shares_to_100([55, 17, 56], 128) == [43.0, 13.3, 43.7]   # era 100,1 %
    assert _shares_to_100([13, 4, 47], 64) == [20.3, 6.3, 73.4]      # era 99,9 %
    assert _shares_to_100([0, 0, 8], 8) == [0.0, 0.0, 100.0]
    assert _shares_to_100([0, 0, 0], 0) == [0.0, 0.0, 0.0]           # cubo vacio

    results_path, attempts_path = synthetic_campaign
    cases = _add_outcome(_load_results(results_path), _load_attempts(attempts_path))
    for _, column in cases.groupby(["model", "attack_type"]):
        counts = [int((column.outcome == o).sum()) for o in OUTCOME_ORDER]
        shares = _shares_to_100(counts, len(column))
        assert sum(counts) == len(column)
        assert round(sum(shares), 9) == 100.0


def test_every_bar_counts_wins_from_any_attempt(synthetic_campaign):
    """Figs 1 and 3-6 all draw ``success_corr``: a case won on attempt 3 counts.

    The agentic cells of the fixture hold five successes, three landing on
    attempt 1 and two only after the rewriter adapted the injection. Every bar
    of those figures is the five; splitting them by *when* they were won is
    fig2's job alone, and that is what ``first_ok_corr`` feeds.
    """
    results_path, attempts_path = synthetic_campaign
    cases = _add_outcome(_load_results(results_path), _load_attempts(attempts_path))
    agentic = cases[(cases.model == MODELS[0]) & (cases.attack_type == "agentic")]

    assert len(agentic) == 8
    assert int(agentic.success_corr.sum()) == 5
    assert int(agentic.first_ok_corr.sum()) == 3

    adaptive = agentic[(agentic.success_corr == 1) & (agentic.first_ok_corr == 0)]
    assert set(adaptive.winning_attempt) == {2, 3}
    assert (adaptive.outcome == "later").all()


def test_attempt_columns_partition_the_wins_the_loop_scored(synthetic_campaign):
    """Fig8's columns are a partition of the wins the loop itself scored.

    Only cases the loop broke are in the figure: attempt 1 is the strategy's
    own injection, not the loop's doing, and a case that ran out of attempts
    never landed on a rung at all. What is left falls in exactly one column.
    The ladder runs to the highest attempt the campaign *spent*, not the
    highest that won, so a rung nobody broke through is still drawn.
    """
    results_path, attempts_path = synthetic_campaign
    cases = _add_outcome(_load_results(results_path), _load_attempts(attempts_path))
    agentic = cases[(cases.model == MODELS[0]) & (cases.attack_type == "agentic")]
    wins = _loop_wins(agentic)

    # Fuera quedan los rotos al 1.er intento y los que agotaron los intentos.
    assert int(agentic.first_ok_corr.sum()) == 3
    assert int((agentic.success_corr == 0).sum()) == 3
    assert len(wins) == 2

    columns = _attempt_columns(_loop_wins(cases), cases)
    assert [label for label, _, rung in columns] == ["2", "3"]
    assert all(rung for _, _, rung in columns)     # no hay columna que no sea intento

    counts = [int(mask(wins).sum()) for _, mask, _ in columns]
    assert counts == [1, 1]                        # 2.º / 3.º
    assert sum(counts) == len(wins)                # nadie fuera, nadie repetido


def test_a_case_absent_from_attempts_keeps_the_campaigns_winning_attempt(
    synthetic_campaign, tmp_path
):
    """A case attempts.csv never recorded keeps what results.csv says.

    Merged campaigns leave cases with no attempt rows at all. The audit cannot
    have moved a verdict it never read, so such a case stays on the rung its
    own summary row names — before this it was dropped to "won at an unknown
    attempt", which fig2 draws as an adaptive win it never was.
    """
    results_path, attempts_path = synthetic_campaign
    attempts = pd.read_csv(attempts_path)
    orphan = ((attempts.model == MODELS[0]) & (attempts.attack_type == "agentic")
              & (attempts.strategy == "S2") & (attempts.num_facts == 10))
    assert orphan.any()
    trimmed = tmp_path / "trimmed_attempts.csv"
    attempts[~orphan].to_csv(trimmed, index=False)

    cases = _add_outcome(_load_results(results_path), _load_attempts(trimmed))
    case = cases[(cases.model == MODELS[0]) & (cases.attack_type == "agentic")
                 & (cases.strategy == "S2") & (cases.num_facts == 10)]

    assert len(case) == 1
    assert int(case.winning_attempt_corr.iloc[0]) == 1
    assert int(case.first_ok_corr.iloc[0]) == 1
    assert case.outcome.iloc[0] == "first"


def test_a_success_whose_attempt_is_unknown_gets_its_own_column(
    synthetic_campaign, tmp_path
):
    """A win attempts.csv did record — as a loss — cannot claim a rung.

    Dropping the winning row of a case the file otherwise covers leaves a
    success whose attempt nothing can identify. It belongs in «no consta», not
    folded into a rung it may never have reached, and that column only appears
    when there is something to put in it.
    """
    results_path, attempts_path = synthetic_campaign
    attempts = pd.read_csv(attempts_path)
    lost = ((attempts.model == MODELS[0]) & (attempts.attack_type == "agentic")
            & (attempts.strategy == "S2") & (attempts.num_facts == 150)
            & (attempts.attempt == 3))
    assert int(lost.sum()) == 1
    trimmed = tmp_path / "trimmed_attempts.csv"
    attempts[~lost].to_csv(trimmed, index=False)

    cases = _add_outcome(_load_results(results_path), _load_attempts(trimmed))
    agentic = cases[(cases.model == MODELS[0]) & (cases.attack_type == "agentic")]

    columns = _attempt_columns(_loop_wins(cases), cases)
    assert [label for label, _, rung in columns if not rung] == [
        UNKNOWN_ATTEMPT_LABEL,
    ]
    wins = _loop_wins(agentic)
    counts = [int(mask(wins).sum()) for _, mask, _ in columns]
    assert counts == [1, 0, 1]               # el 3.º se queda vacío y se dibuja
    assert sum(counts) == len(wins)


def test_judge_error_figure_is_skipped_when_nothing_was_flagged(
    synthetic_campaign, tmp_path
):
    """An empty fig 7 would read as "the judge was checked and was right"."""
    results_path, attempts_path = synthetic_campaign
    for path in (results_path, attempts_path):
        df = pd.read_csv(path)
        df["judge_false_positive"] = 0
        df["judge_false_negative"] = 0
        df.to_csv(path, index=False)

    written = generate_plots(results_path, attempts_path, tmp_path / "figuras")
    assert {Path(p).name for p in written} == PER_MODEL_FIGURES
