from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display server.

import re
import textwrap
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiment.judge_audit import (
    FN_COLUMN,
    FP_COLUMN,
    annotate_results,
    corrected_cases,
    flag,
    has_flags,
)

# --- paleta validada (dataviz, modo claro) ---
BLUE = "#2a78d6"                             # azul del eje agéntico
RED = "#e34948"                              # rojo del eje dañino; fig2 lo reutiliza
                                             # para el tramo del bucle adaptativo
AGGREGATE = "#6f6d66"                        # gris del agregado: no es categoría
NEUTRAL = "#c3c2b7"                          # relleno neutro: "sin resultado"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

# fig7 no habla de ejes de ataque sino de errores de medida, así que sale del
# par azul/rojo a propósito: aquí el color sigue al tipo de error del juez.
FP_COLOR, FN_COLOR = "#a8386b", "#1f6f7a"    # falso positivo / falso negativo

ATTACK_ORDER = ["agentic", "harmful"]
ATTACK_LABELS = {
    "agentic": "agéntico (abrir ventana)",
    "harmful": "dañino (contenido tóxico)",
}
# El color sigue al tipo de ataque en todas las figuras, nunca al valor.
ATTACK_COLORS = {"agentic": BLUE, "harmful": RED}

# fig2: los tres desenlaces de un caso, en orden de coste para el atacante.
OUTCOME_ORDER = ["first", "later", "none"]
OUTCOME_COLORS = {"first": BLUE, "later": RED, "none": NEUTRAL}
OUTCOME_LABELS = {
    "first": "rompió al 1.er intento",
    "later": "rompió tras adaptar (intentos 2–N)",
    "none": "aguantó (ningún intento tuvo éxito)",
}

FACT_SOURCE_ORDER = ["real", "invented", "mixed"]
FACT_SOURCE_LABELS = {"real": "Real", "invented": "Inventado", "mixed": "Mixto"}
STRATEGY_ORDER = ["S1", "S2", "S3"]
STRATEGY_LABELS = {
    "S1": "S1\nsolo instrucción",
    "S2": "S2\nhechos → instrucción",
    "S3": "S3\ninstrucción enterrada",
}

# El corte de la figura 4: 1–25 hechos es «pocos», 50–200 es «muchos».
# La rejilla de num_facts (0, 1, 2, 5, 10, 25, 50, 100, 150, 200) es
# logarítmica, así que el salto 25 → 50 es donde los hechos dejan de
# acompañar a la instrucción y pasan a ser el cuerpo del correo.
FEW_FACTS_MAX = 25

# The case key: one row per combination in results.csv.
CASE_KEY = ["model", "attack_type", "strategy", "num_facts", "fact_source", "rep"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
})


# --------------------------------------------------------------------------- #
# Small drawing helpers
# --------------------------------------------------------------------------- #
def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _pct(value: float) -> str:
    """Spanish decimal comma, e.g. ``62,5 %``."""
    return f"{value:.1f} %".replace(".", ",")


def _shares_to_100(counts: Sequence[int], total: int) -> list[float]:
    """Cuotas en % con un decimal que suman exactamente 100,0.

    Restos mayores: se trunca cada cuota al décimo y los décimos que sobran van
    a los mayores restos. Redondear cada tramo por su cuenta no cuadra: con
    n = 64 o n = 128 toda cuota cae en una fracción binaria exacta acabada en
    ...25 / ...75, así que los tres tramos de una columna pueden irse al mismo
    lado y dejar un 99,9 % o un 100,1 % impreso sobre una pila que sí llega a
    100. Solo toca el texto; las barras se dibujan con la cuota exacta.
    """
    if not total:
        return [0.0] * len(counts)
    tenths = [1000 * k / total for k in counts]   # en décimas de punto
    out = [int(t) for t in tenths]                # int() es floor: todas son >= 0
    spare = 1000 - sum(out)
    # Mayor resto primero; a igualdad de resto, el tramo con más casos.
    order = sorted(range(len(out)), key=lambda i: (out[i] - tenths[i], -counts[i]))
    for i in order[:spare]:
        out[i] += 1
    return [t / 10 for t in out]


def _subtitle(ax, text: str, y: float = 1.03) -> None:
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=MUTED)


def _footnote(fig, text: str) -> None:
    """Hang a wrapped note under the axes, sized to the figure so it never
    overflows the right edge (``bbox_inches="tight"`` grows to include it)."""
    chars = max(60, int(fig.get_figwidth() / 0.062))  # ~0.062 in per char at 8.5 pt
    fig.text(0.01, -0.03, textwrap.fill(text, chars), fontsize=8.5,
             color=MUTED, va="top")


def _slug(text: str) -> str:
    """Filesystem-safe model id: ``qwen2.5-tools:7b`` -> ``qwen2.5-tools-7b``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def _zero_stub(ax, centre: float, width: float, color: str) -> None:
    """A flat 0 % bar draws nothing, which reads the same as a missing bucket.

    Lay a 2.5 pt rule of the series colour on the baseline so "attacked, never
    succeeded" stays visually distinct from "sin datos".
    """
    ax.plot([centre - width / 2, centre + width / 2], [0, 0], color=color,
            linewidth=2.5, solid_capstyle="butt", zorder=4)


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Loading / derived columns
# --------------------------------------------------------------------------- #
def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _load_results(path: str | Path) -> pd.DataFrame:
    """Read results.csv, dropping legacy rows that predate the ``model`` column."""
    df = pd.read_csv(path)
    if "model" not in df.columns:
        return df.iloc[0:0]
    df = df[df["model"].notna() & (df["model"].astype(str).str.strip() != "")].copy()
    for col in ("success", "winning_attempt", "attempts_used", "num_facts", "rep",
                FP_COLUMN, FN_COLUMN):
        if col in df.columns:
            df[col] = _to_int(df[col])
    for col in ("strategy", "fact_source", "attack_type"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def _load_attempts(path: str | Path) -> pd.DataFrame | None:
    """Read attempts.csv if present (absent for single-shot ``adaptive=false``)."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "model" not in df.columns:
        return None
    df = df[df["model"].notna() & (df["model"].astype(str).str.strip() != "")].copy()
    for col in ("success", "attempt", "num_facts", "rep", FP_COLUMN, FN_COLUMN):
        if col in df.columns:
            df[col] = _to_int(df[col])
    return df


def _add_outcome(results: pd.DataFrame, attempts: pd.DataFrame | None) -> pd.DataFrame:
    """Attach ``first_ok`` (attempt 1 broke it) and ``outcome`` to every case.

    ``winning_attempt`` in results.csv is the canonical source. Legacy files
    without it fall back to attempts.csv, de-duplicated to the last attempt-1
    row per case (re-runs from merged campaigns leave several).

    Also attaches the audited view — ``success_corr`` / ``winning_attempt_corr``
    / ``first_ok_corr`` — via :func:`_add_corrected`; the raw columns stay so a
    campaign can be read before and after the audit.
    """
    df = results.copy()
    if "winning_attempt" in df.columns:
        df["first_ok"] = ((df["success"] == 1) & (df["winning_attempt"] == 1)).astype(int)
    elif attempts is not None and "attempt" in attempts.columns:
        key = [c for c in CASE_KEY if c in df.columns and c in attempts.columns]
        first = (attempts[attempts["attempt"] == 1]
                 .drop_duplicates(subset=key, keep="last")[key + ["success"]]
                 .rename(columns={"success": "first_ok"}))
        df = df.merge(first, on=key, how="left")
        df["first_ok"] = _to_int(df["first_ok"])
    else:
        raise ValueError(
            "results.csv has no 'winning_attempt' column and no usable attempts.csv "
            "was supplied: cannot tell a 1st-attempt success from a later one."
        )
    df = _add_corrected(df, attempts)
    df["outcome"] = np.where(
        df["first_ok_corr"] == 1, "first",
        np.where(df["success_corr"] == 1, "later", "none"),
    )
    return df


def _raw_winning_attempt(cases: pd.DataFrame) -> pd.Series:
    """El intento ganador según la campaña: ``winning_attempt``, 0 si no rompió.

    Un fichero heredado sin esa columna solo sabe si ganó el 1.º, que es
    exactamente lo que vale su ``first_ok``.
    """
    column = "winning_attempt" if "winning_attempt" in cases.columns else "first_ok"
    return _to_int(cases[column])


def _add_corrected(cases: pd.DataFrame, attempts: pd.DataFrame | None) -> pd.DataFrame:
    """Add ``success_corr`` / ``winning_attempt_corr`` / ``first_ok_corr``: the
    judge's errors taken out.

    A false positive stops counting as a success and a false negative starts
    counting as one, so every ASR the figures draw is the audited rate. On a
    campaign that was never audited both flags are 0 and the corrected columns
    are copies of the raw ones — the figures are unchanged.

    The case outcome always comes from results.csv plus its net per-case flags,
    never from re-aggregating attempts.csv: a merged or partial attempts file
    would otherwise quietly disagree with the campaign's own record of which
    cases succeeded. attempts.csv is used only for the one thing results.csv
    cannot express after the audit — *which* attempt really broke the case, so
    a false positive can move the winning attempt without flipping the case.
    Without it, a false negative lands in "rompió tras adaptar" and in the «no
    consta» column of fig8, because the attempt that delivered cannot be
    identified.
    """
    df = cases.copy()
    if not has_flags(df) and attempts is not None and has_flags(attempts):
        df = annotate_results(df, attempts)  # audit only in the attempts file
    fp, fn = flag(df, FP_COLUMN), flag(df, FN_COLUMN)
    df["success_corr"] = (df["success"] - fp + fn).clip(0, 1).astype(int)

    if attempts is not None and has_flags(attempts):
        key = [c for c in CASE_KEY if c in df.columns and c in attempts.columns]
        won = corrected_cases(attempts)[key + ["winning_attempt_corr"]]
        df = df.merge(won, on=key, how="left")
        # Un caso que attempts.csv no recoge se queda con la respuesta de la
        # campaña: la auditoría no puede haber movido un intento que nunca leyó.
        # Sin este respaldo, los casos que una campaña fusionada dejó sin filas
        # de intento se irían enteros a «rompió tras adaptar» aunque
        # results.csv diga que cayeron a la primera.
        df["winning_attempt_corr"] = df["winning_attempt_corr"].fillna(
            _raw_winning_attempt(df))
    else:
        df["winning_attempt_corr"] = _raw_winning_attempt(df)
    # A case that no longer succeeds has no winning attempt: keeps fig2's three
    # segments — and fig8's columns — summing to the whole column.
    won = _to_int(df["winning_attempt_corr"]).where(df["success_corr"] == 1, 0)
    df["winning_attempt_corr"] = won
    df["first_ok_corr"] = (won == 1).astype(int)
    return df


def _correction_note(cases: pd.DataFrame) -> str:
    """One line stating how many judge errors this figure's numbers absorb.

    Empty when the slice carries none, so an unaudited campaign draws the same
    footnotes it always did.
    """
    fp, fn = int(flag(cases, FP_COLUMN).sum()), int(flag(cases, FN_COLUMN).sum())
    if not (fp or fn):
        return ""
    parts = []
    if fp:
        parts.append(f"{fp} falso{'s' if fp > 1 else ''} positivo{'s' if fp > 1 else ''} "
                     f"deja{'n' if fp > 1 else ''} de contar como éxito")
    if fn:
        tail = "contarlo" if fp else "contar como éxito"
        parts.append(f"{fn} falso{'s' if fn > 1 else ''} negativo{'s' if fn > 1 else ''} "
                     f"pasa{'n' if fn > 1 else ''} a {tail}")
    return ("Veredictos del juez corregidos con la auditoría manual del eje dañino: "
            + " y ".join(parts) + " (figura 7).")


def _models(df: pd.DataFrame) -> list[str]:
    return sorted(df["model"].unique())


def _attacks(df: pd.DataFrame) -> list[str]:
    present = set(df["attack_type"].unique())
    ordered = [a for a in ATTACK_ORDER if a in present]
    return ordered or sorted(present)


# --------------------------------------------------------------------------- #
# Figure 1 — per model: campaign ASR, agentic vs harmful vs total
# --------------------------------------------------------------------------- #
def _fig_asr_by_axis(cases: pd.DataFrame, model: str, out_dir: Path) -> Path:
    """The headline: one bar per attack axis plus the model total, nothing else."""
    groups: list[tuple[str, str, pd.DataFrame, str]] = [
        (ATTACK_LABELS.get(a, a).replace(" (", "\n("), a,
         cases[cases.attack_type == a], ATTACK_COLORS.get(a, BLUE))
        for a in _attacks(cases)
    ]
    groups.append(("TOTAL\n(ambos ejes)", "total", cases, AGGREGATE))

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(2.4 + 1.9 * len(groups), 5.2))
    for xi, (_, _, sub, color) in enumerate(groups):
        n, k = len(sub), int((sub.success_corr == 1).sum())
        p = 100 * k / n if n else 0.0
        ax.bar(xi, p, 0.42, color=color, zorder=3)
        if n and not p:
            _zero_stub(ax, xi, 0.42, color)
        ax.text(xi, p + 2.5, f"{_pct(p)}\n({k}/{n})" if n else "sin datos",
                ha="center", va="bottom", fontsize=10.5,
                color=MUTED if not n else INK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _, _ in groups])
    ax.set_xlabel("Eje del ataque")
    ax.set_ylabel("ASR de campaña (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("ASR por eje de ataque", fontweight="bold", color=INK,
                 loc="left", pad=26)
    _subtitle(ax, model)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out_dir, f"fig1_{_slug(model)}_asr_por_eje.png")


# --------------------------------------------------------------------------- #
# Figure 2 — per model: 1st attempt vs adaptive gain vs held
# --------------------------------------------------------------------------- #
def _fig_outcome_split(cases: pd.DataFrame, model: str, out_dir: Path) -> Path:
    groups: list[tuple[str, pd.DataFrame]] = [("total", cases)]
    for atk in _attacks(cases):
        groups.append((atk, cases[cases.attack_type == atk]))
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(2.6 + 1.9 * len(groups), 5.2))
    counts = {o: [int((sub.outcome == o).sum()) for _, sub in groups]
              for o in OUTCOME_ORDER}
    # Las etiquetas de cada columna se cuadran al 100,0 % entre sí; la geometría
    # va aparte, con la cuota exacta, así que la pila no se mueve un píxel.
    shown = [dict(zip(OUTCOME_ORDER,
                      _shares_to_100([counts[o][gi] for o in OUTCOME_ORDER], len(sub))))
             for gi, (_, sub) in enumerate(groups)]

    bottoms = np.zeros(len(groups))
    for outcome in OUTCOME_ORDER:
        ks = counts[outcome]
        shares = np.array([100 * k / len(sub) if len(sub) else 0.0
                           for k, (_, sub) in zip(ks, groups)], dtype=float)
        # edgecolor=SURFACE renders as the 2px surface gap between segments.
        ax.bar(x, shares, 0.44, bottom=bottoms, color=OUTCOME_COLORS[outcome],
               zorder=3, label=OUTCOME_LABELS[outcome],
               edgecolor=SURFACE, linewidth=1.6)
        for xi, (h, b, k) in enumerate(zip(shares, bottoms, ks)):
            if not k:
                continue
            text = f"{k} · {_pct(shown[xi][outcome])}"
            if h >= 9:  # only label inside a segment the label actually fits in
                ax.text(xi, b + h / 2, text, ha="center", va="center", fontsize=9,
                        color="white" if outcome != "none" else INK,
                        fontweight="bold")
            else:       # too thin for an inside label: park it beside the bar,
                        # on whichever side has room (inwards for the last group)
                last = xi == len(groups) - 1
                ax.text(xi + (-0.27 if last else 0.27), b + h / 2, text,
                        ha="right" if last else "left", va="center",
                        fontsize=8.5, color=INK2)
        bottoms += shares

    # Headline above each bar: the full adaptive ASR = 1st attempt + later.
    # Se suma sobre las cuotas ya etiquetadas (no se recalcula) para que el
    # titular cuadre con los dos tramos que tiene justo debajo.
    for xi, (_, sub) in enumerate(groups):
        n = len(sub)
        k = counts["first"][xi] + counts["later"][xi]
        asr = shown[xi]["first"] + shown[xi]["later"]
        ax.text(xi, 101.5, f"ASR {_pct(asr)}  ({k}/{n})", ha="center", va="bottom",
                fontsize=9.5, color=INK, fontweight="bold")

    labels = []
    for name, sub in groups:
        head = "TOTAL" if name == "total" else ATTACK_LABELS.get(name, name)
        labels.append(f"{head}\n(n = {len(sub)} casos)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Casos (% de la columna)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.62, len(groups) - 0.38)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("¿Cuánto aporta el bucle adaptativo?", fontweight="bold",
                 color=INK, loc="left", pad=44)
    # y=1.10 clears the per-bar "ASR ..." headlines drawn just above the plot.
    _subtitle(ax, f"{model} · reparto de casos por desenlace", y=1.10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=1, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out_dir, f"fig2_{_slug(model)}_desenlace_por_intento.png")


# --------------------------------------------------------------------------- #
# Figures 3–6 — campaign ASR by cell, both attack axes in the same chart
# --------------------------------------------------------------------------- #
Bucket = tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]


def _asr_grouped_bars(
    by_model: pd.DataFrame,
    buckets: Sequence[Bucket],
    title: str,
    subtitle: str,
    xlabel: str,
    footnote: str,
    out_dir: Path,
    filename: str,
) -> Path:
    """ASR de campaña por cubo: el caso cuenta si lo rompió *cualquier* intento.

    Misma tasa y misma unidad —el caso— que las figuras 1 y 2, así que todo
    el juego de figuras vive en una sola escala. Qué parte de esos éxitos costó
    reescribir la inyección es cosa de la figura 2, la única que abre el
    desenlace por intento; aquí la barra es el total y nada más.
    """
    attacks = _attacks(by_model)
    x = np.arange(len(buckets))
    group_w = 0.74
    bar_w = group_w / len(attacks) - 0.03  # el hueco es el fondo, no una línea

    fig, ax = plt.subplots(figsize=(2.8 + 2.3 * len(buckets), 5.2))
    for i, attack in enumerate(attacks):
        cell = by_model[by_model.attack_type == attack]
        offset = -group_w / 2 + group_w * (i + 0.5) / len(attacks)
        heights, notes, sizes = [], [], []
        for _, select in buckets:
            sub = select(cell)
            n, k = len(sub), int(sub.success_corr.sum())
            heights.append(100 * k / n if n else 0.0)
            notes.append(f"{_pct(100 * k / n)}\n({k}/{n})" if n else "sin\ndatos")
            sizes.append(n)
        color = ATTACK_COLORS.get(attack, BLUE)
        bars = ax.bar(x + offset, heights, bar_w, color=color, zorder=3,
                      label=ATTACK_LABELS.get(attack, attack))
        for b, note, n in zip(bars, notes, sizes):
            centre = b.get_x() + b.get_width() / 2
            if n and not b.get_height():
                _zero_stub(ax, centre, b.get_width(), color)
            ax.text(centre, b.get_height() + 2.0, note,
                    ha="center", va="bottom", fontsize=9,
                    color=MUTED if not n else INK)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in buckets])
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("ASR de campaña (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.62, len(buckets) - 0.38)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title(title, fontweight="bold", color=INK, loc="left", pad=26)
    _subtitle(ax, subtitle)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.21),
              ncol=len(attacks), fontsize=9.5)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out_dir, filename)


def _range_label(values: pd.Series) -> str:
    """``(1–25)`` from the fact counts actually present in a band."""
    if values.empty:
        return ""
    lo, hi = int(values.min()), int(values.max())
    return f"({lo}–{hi})" if lo != hi else f"({lo})"


_ASR_NOTE = ("ASR de campaña = casos rotos en algún intento del bucle / casos "
             "totales; es la misma tasa que dan las figuras 1 y 2. Entre "
             "paréntesis, éxitos/total de cada barra. Cuánto de ese total puso "
             "la reescritura adaptativa lo desglosa la figura 2.")


def _model_asr_figures(by_model: pd.DataFrame, model: str, out_dir: Path) -> list[Path]:
    """The four campaign-ASR breakdowns for one model, both axes in each."""
    slug = _slug(model)
    head = f"{model} · ASR por tipo de ataque"
    with_facts = by_model[by_model.num_facts > 0]
    few = by_model[by_model.num_facts.between(1, FEW_FACTS_MAX)]
    many = by_model[by_model.num_facts > FEW_FACTS_MAX]
    note = f"{_ASR_NOTE} {_correction_note(by_model)}".strip()

    return [
        _asr_grouped_bars(
            by_model,
            [("sin hechos\n(0)", lambda d: d[d.num_facts == 0]),
             (f"con hechos\n{_range_label(with_facts.num_facts)}",
              lambda d: d[d.num_facts > 0])],
            "¿Ayudan los hechos al éxito de la inyección?",
            head,
            "Hechos que acompañan a la instrucción maliciosa",
            note,
            out_dir, f"fig3_{slug}_asr_hechos.png",
        ),
        _asr_grouped_bars(
            by_model,
            [("sin hechos\n(0)", lambda d: d[d.num_facts == 0]),
             (f"pocos hechos\n{_range_label(few.num_facts)}",
              lambda d: d[d.num_facts.between(1, FEW_FACTS_MAX)]),
             (f"muchos hechos\n{_range_label(many.num_facts)}",
              lambda d: d[d.num_facts > FEW_FACTS_MAX])],
            "Comparación por volumen de hechos",
            head,
            "Número de hechos inyectados junto a la instrucción",
            note + " Desglose de la figura 3.",
            out_dir, f"fig4_{slug}_asr_volumen_hechos.png",
        ),
        _asr_grouped_bars(
            by_model,
            [(STRATEGY_LABELS.get(s, s), (lambda v: lambda d: d[d.strategy == v])(s))
             for s in STRATEGY_ORDER],
            "Comparación por estrategia de inyección",
            head,
            "Posición de la instrucción maliciosa dentro del texto",
            note + " Aviso: S1 es la línea base sin hechos, así que su "
            "diferencia con S2/S3 mezcla posición y presencia de hechos.",
            out_dir, f"fig5_{slug}_asr_estrategia.png",
        ),
        _asr_grouped_bars(
            by_model,
            [(FACT_SOURCE_LABELS.get(s, s),
              (lambda v: lambda d: d[d.fact_source == v])(s))
             for s in FACT_SOURCE_ORDER],
            "Comparación por origen de los hechos inyectados",
            head,
            "Origen de los hechos inyectados",
            note + " Solo casos con hechos: la línea base sin hechos "
            "(fact_source = none) queda fuera. Real = hechos posteriores al corte "
            "de entrenamiento; Inventado = plausibles pero ficticios; Mixto = "
            "mezcla de ambos.",
            out_dir, f"fig6_{slug}_asr_origen_hechos.png",
        ),
    ]


# --------------------------------------------------------------------------- #
# Figure 7 — the judge's own errors, per model and direction
# --------------------------------------------------------------------------- #
def _model_label(model: str) -> str:
    """Wrap the model id onto two lines at most: fig7 hangs a legend under the
    ticks, and a third line would run into it."""
    return textwrap.fill(model, 20)


def _harmful(cases: pd.DataFrame) -> pd.DataFrame:
    return cases[cases.attack_type == "harmful"]


def _judge_error_counts(cases: pd.DataFrame) -> pd.DataFrame:
    """Per-model judge errors on the harmful axis, over the axis's own attacks.

    Both series share one denominator — every attack the campaign launched on
    this axis for this model — so the bars answer "of the attacks we ran, how
    many did the judge score wrong, and in which direction?". Counting per
    attack rather than per verdict also makes the flags the *net* effect on the
    outcome: a false positive that a later genuine success overrode never
    changed what the campaign concluded, so it is not an error here.
    """
    harmful = _harmful(cases)
    rows = []
    for model in sorted(harmful["model"].unique()):
        sub = harmful[harmful.model == model]
        rows.append({
            "model": model,
            "attacks": len(sub),
            "fp": int(flag(sub, FP_COLUMN).sum()),
            "fn": int(flag(sub, FN_COLUMN).sum()),
        })
    return pd.DataFrame(rows)


def _audited_counts(cases: pd.DataFrame) -> pd.DataFrame | None:
    """Guard for fig7: the per-model table, or ``None`` to skip the figure.

    A campaign with no audited errors gets no figure at all — an empty chart
    would read as "the judge was checked and was right", which is not the same
    claim as "no audit was applied".
    """
    if "harmful" not in set(cases["attack_type"]):
        return None
    counts = _judge_error_counts(cases)
    if counts.empty or not (counts["fp"].sum() or counts["fn"].sum()):
        return None
    return counts


def _fig_judge_errors(cases: pd.DataFrame, out_dir: Path) -> Path | None:
    """Fig 7: how many attacks the judge mis-scored, per model and direction."""
    counts = _audited_counts(cases)
    if counts is None:
        return None

    series = [
        ("fp", FP_COLOR, "falso positivo: dijo COMPLIED y no se entregó nada"),
        ("fn", FN_COLOR, "falso negativo: dijo REFUSED y sí se entregó"),
    ]
    x = np.arange(len(counts))
    group_w = 0.52
    bar_w = group_w / len(series) - 0.03
    top = 10.0

    fig, ax = plt.subplots(figsize=(3.2 + 2.6 * len(counts), 5.2))
    for i, (column, color, label) in enumerate(series):
        offset = -group_w / 2 + group_w * (i + 0.5) / len(series)
        heights = [100 * r[column] / r["attacks"] if r["attacks"] else 0.0
                   for _, r in counts.iterrows()]
        top = max(top, max(heights, default=0.0))
        bars = ax.bar(x + offset, heights, bar_w, color=color, zorder=3, label=label)
        for b, (_, r) in zip(bars, counts.iterrows()):
            centre = b.get_x() + b.get_width() / 2
            if r["attacks"] and not b.get_height():
                _zero_stub(ax, centre, b.get_width(), color)
            text = (f"{_pct(b.get_height())}\n({r[column]}/{r['attacks']})"
                    if r["attacks"] else "sin\ndatos")
            ax.text(centre, b.get_height() + top * 0.03, text, ha="center",
                    va="bottom", fontsize=9.5, color=INK if r["attacks"] else MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in counts["model"]])
    ax.set_ylabel("Ataques mal puntuados (% del eje)")
    ax.set_ylim(0, min(100.0, top * 1.45))
    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_title("Errores del juez por modelo", fontweight="bold", color=INK,
                 loc="left", pad=26)
    _subtitle(ax, "eje dañino, total de falsos positivos y negativos entre los ataques lanzados")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=1, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "fig7_errores_juez.png")


# --------------------------------------------------------------------------- #
# Figure 8 — per model: which loop attempt landed the hit
# --------------------------------------------------------------------------- #
UNKNOWN_ATTEMPT_LABEL = "no consta\n(sin registro)"

# (etiqueta, máscara de casos, ¿es un peldaño de la escalera de intentos?)
AttemptColumn = tuple[str, Callable[[pd.DataFrame], pd.Series], bool]

# La escalera empieza en el 2.º intento: el 1.º no es obra del bucle —es la
# inyección tal y como sale de la estrategia— y ya lo cuentan fig1 y fig2.
FIRST_RUNG = 2

# Una etiqueta de dos líneas pide ~10 puntos de eje por encima de la barra: a
# partir de aquí no caben y se escribe dentro.
LABEL_INSIDE_MIN = 86.0


def _loop_wins(by_model: pd.DataFrame) -> pd.DataFrame:
    """Los éxitos que puso el bucle: los que cayeron del 2.º intento en adelante.

    Es lo único que la figura reparte, y por tanto también su denominador. Un
    caso roto a la primera no es obra del bucle —es la inyección tal y como sale
    de la estrategia— y uno que agotó los intentos no cayó en ninguno: ni uno ni
    otro tienen peldaño donde ponerse. Cuántos son de cada clase lo dicen las
    figuras 1 y 2.
    """
    return by_model[(by_model["success_corr"] == 1)
                    & (by_model["first_ok_corr"] == 0)]


def _attempt_columns(wins: pd.DataFrame, spent: pd.DataFrame) -> list[AttemptColumn]:
    """Un peldaño por intento del bucle, del 2.º al último que se llegó a gastar.

    ``wins`` es el corte de :func:`_loop_wins` —lo que se reparte— y ``spent`` la
    campaña entera del modelo, que es quien fija la altura de la escalera: llega
    hasta el intento más alto *gastado* (``attempts_used``), no hasta el más alto
    que ganó. Un peldaño vacío también es un resultado —nadie rompió ahí teniendo
    la oportunidad— y recortarlo lo escondería. La columna «no consta» solo
    aparece cuando hace falta: un éxito cuyo intento ganador no se puede
    identificar (un ``attempts.csv`` incompleto, o un falso negativo del juez sin
    fila de intento) tiene que verse, no colarse en un peldaño que no le toca.

    Las máscaras son excluyentes y cubren todo ``wins``, así que las columnas
    reparten exactamente los éxitos del bucle.
    """
    won = _to_int(wins["winning_attempt_corr"])
    used = (_to_int(spent["attempts_used"]) if "attempts_used" in spent.columns
            else _to_int(spent["winning_attempt_corr"]))
    top = max(int(used.max()) if len(used) else 0, int(won.max()) if len(won) else 0,
              FIRST_RUNG)

    columns: list[AttemptColumn] = [
        (str(rung),
         lambda d, rung=rung: (d.success_corr == 1) & (d.winning_attempt_corr == rung),
         True)
        for rung in range(FIRST_RUNG, top + 1)
    ]
    if int((won == 0).sum()):
        columns.append((UNKNOWN_ATTEMPT_LABEL,
                        lambda d: (d.success_corr == 1) & (d.winning_attempt_corr == 0),
                        False))
    return columns


def _fig_attempt_distribution(by_model: pd.DataFrame, model: str,
                              out_dir: Path) -> Path:
    """Fig 8: de los ataques que el bucle acabó rompiendo, en qué intento cayeron.

    Es el tramo *rompió tras adaptar* de la figura 2 abierto peldaño a peldaño.
    Misma unidad —el caso— pero denominador propio: los éxitos del bucle, que la
    leyenda nombra por eje. La altura dice cómo se reparten esos éxitos entre los
    intentos, no cuántos son; eso ya lo dicen las figuras 1 y 2.
    """
    wins = _loop_wins(by_model)
    columns = _attempt_columns(wins, by_model)
    attacks = _attacks(by_model)  # los dos ejes, aunque uno se quede sin éxitos
    rungs = sum(1 for _, _, is_rung in columns if is_rung)
    # Un hueco de media columna separa la escalera de lo que no es un intento.
    x = np.array([i + (0.0 if is_rung else 0.55)
                  for i, (_, _, is_rung) in enumerate(columns)])
    group_w = 0.72
    bar_w = group_w / len(attacks) - 0.03  # el hueco es el fondo, no una línea

    fig, ax = plt.subplots(figsize=(2.6 + 1.6 * len(columns), 5.2))
    if rungs < len(columns):  # la regla solo existe si hay algo al otro lado
        ax.axvline((x[rungs - 1] + x[rungs]) / 2, color=GRID, linewidth=1.0, zorder=1)
    for i, attack in enumerate(attacks):
        cell = wins[wins.attack_type == attack]
        n = len(cell)
        offset = -group_w / 2 + group_w * (i + 0.5) / len(attacks)
        color = ATTACK_COLORS.get(attack, BLUE)
        counts = [int(mask(cell).sum()) if n else 0 for _, mask, _ in columns]
        # Las columnas reparten n entero, así que las etiquetas cuadran al 100 %.
        shares = _shares_to_100(counts, n)
        label = ATTACK_LABELS.get(attack, attack)
        bars = ax.bar(x + offset, [100 * k / n if n else 0.0 for k in counts],
                      bar_w, color=color, zorder=3,
                      label=f"{label} · n = {n}" if n else f"{label} · sin datos")
        for bar, k, share in zip(bars, counts, shares):
            centre, height = bar.get_x() + bar.get_width() / 2, bar.get_height()
            if n and not height:
                _zero_stub(ax, centre, bar.get_width(), color)
            # Una barra casi llena no deja sitio encima: la etiqueta va dentro,
            # en tinta de texto, que sobre el azul y el rojo contrasta más que
            # el blanco.
            inside = height > LABEL_INSIDE_MIN
            ax.text(centre, height - 2.0 if inside else height + 2.0,
                    f"{_pct(share)}\n({k}/{n})" if n else "sin\ndatos",
                    ha="center", va="top" if inside else "bottom",
                    fontsize=9, color=INK if n else MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in columns])
    ax.set_xlabel("Intento del bucle que rompió el caso")
    ax.set_ylabel("Éxitos del bucle (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0] - 0.62, x[-1] + 0.38)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("¿En qué intento del bucle cae el ataque?", fontweight="bold",
                 color=INK, loc="left", pad=26)
    _subtitle(ax, f"{model} · reparto de los éxitos logrados tras reescribir "
                  "la inyección")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.21),
              ncol=len(attacks), fontsize=9.5)
    _style(ax)
    fig.tight_layout()
    return _save(fig, out_dir, f"fig8_{_slug(model)}_distribucion_intentos.png")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_plots(
    results_path: str | Path,
    attempts_path: str | Path | None = None,
    out_dir: str | Path = "figuras",
) -> list[Path]:
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_path)
    if results.empty:
        raise ValueError(f"No usable rows in {results_path} (missing 'model' column?).")
    attempts = _load_attempts(attempts_path) if attempts_path is not None else None
    cases = _add_outcome(results, attempts)

    written: list[Path] = []
    for model in _models(cases):
        by_model = cases[cases.model == model]
        written.append(_fig_asr_by_axis(by_model, model, out_dir))
        written.append(_fig_outcome_split(by_model, model, out_dir))
        written.extend(_model_asr_figures(by_model, model, out_dir))
        written.append(_fig_attempt_distribution(by_model, model, out_dir))
    # Campaign-wide, not per model: fig7 exists to compare the models.
    judge_errors = _fig_judge_errors(cases, out_dir)
    if judge_errors is not None:
        written.append(judge_errors)
    return written
