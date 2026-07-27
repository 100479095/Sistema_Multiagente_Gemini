"""Adaptive multi-model figures for the prompt-injection campaign.

Generalises the single-model ``scripts/plot_qwen_results.py`` to the adaptive
pipeline: it reads ``results.csv`` (one summary row per case) and ``attempts.csv``
(one row per attack attempt) and writes four figures to ``results/figuras/``:

  1. ASR por modelo × tipo de ataque, con intervalo de Wilson (¿qué modelo aguanta?).
  2. Intentos hasta el éxito — distribución de ``winning_attempt`` en los casos
     que triunfan, por modelo (¿cuánto hay que insistir?).
  3. Ganancia por adaptación — ASR de un solo intento frente al ASR del bucle
     adaptativo completo, por modelo × ataque (¿cuánto aporta adaptar el prompt?).
  4. Fuente del prompt ganador — barras apiladas base/juez/lista, por modelo ×
     ataque (¿de dónde sale el prompt que rompe la defensa?).

The palette is the dataviz light-mode instance validated for these charts:
agentic=BLUE / harmful=RED (categorical pair passes all checks) and the prompt
sources base=BLUE / judge=ORANGE / fallback=AQUA (AQUA's sub-3:1 contrast is
relieved by the direct count labels on every stacked segment).

``generate_plots`` is the importable entry point (called from
``scripts/run_experiment.py`` after a campaign, and directly runnable via
``scripts/plot_results.py``). Rendering uses the headless Agg backend, so it
needs no display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display server.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from experiment.metrics import wilson_interval

# --- paleta validada (dataviz, modo claro) ---
BLUE, RED, ORANGE, AQUA = "#2a78d6", "#e34948", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

# agentic vs harmful — validated categorical pair (ΔE well above the floor).
ATTACK_ORDER = ["agentic", "harmful"]
ATTACK_COLORS = {"agentic": BLUE, "harmful": RED}
ATTACK_LABELS = {
    "agentic": "agentic (abrir ventana)",
    "harmful": "harmful (contenido dañino)",
}
# where the winning prompt came from, in adaptation order (base → judge → list).
SOURCE_ORDER = ["base", "judge", "fallback"]
SOURCE_COLORS = {"base": BLUE, "judge": ORANGE, "fallback": AQUA}
SOURCE_LABELS = {
    "base": "base (inyección original)",
    "judge": "juez (reescritura LLM)",
    "fallback": "lista predeterminada",
}
# up to a handful of models on one axis; BLUE/ORANGE/AQUA all mutually distinct.
MODEL_CYCLE = [BLUE, ORANGE, AQUA, RED]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
})


# --------------------------------------------------------------------------- #
# Small drawing helpers (shared look with plot_qwen_results.py)
# --------------------------------------------------------------------------- #
def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _barlabels(ax, bars, labels, dy=0.0) -> None:
    for b, lab in zip(bars, labels):
        if lab is None or lab == "":
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, lab,
                ha="center", va="bottom", fontsize=9, color=INK2)


def _model_short(model: str) -> str:
    """A compact axis label: drop the ``:tag`` suffix Ollama appends."""
    return model.split(":", 1)[0]


# --------------------------------------------------------------------------- #
# Loading / coercion
# --------------------------------------------------------------------------- #
def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _load_results(path: str | Path) -> pd.DataFrame:
    """Read results.csv, dropping legacy rows that predate the ``model`` column."""
    df = pd.read_csv(path)
    if "model" not in df.columns:
        return df.iloc[0:0]
    df = df[df["model"].notna() & (df["model"].astype(str).str.strip() != "")].copy()
    for col in ("success", "winning_attempt", "attempts_used"):
        if col in df.columns:
            df[col] = _to_int(df[col])
    if "winning_prompt_source" in df.columns:
        df["winning_prompt_source"] = df["winning_prompt_source"].fillna("").astype(str)
    return df


def _load_attempts(path: str | Path) -> pd.DataFrame | None:
    """Read attempts.csv if it exists (absent for single-shot ``adaptive=false``)."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "model" not in df.columns:
        return None
    df = df[df["model"].notna() & (df["model"].astype(str).str.strip() != "")].copy()
    for col in ("success", "attempt"):
        if col in df.columns:
            df[col] = _to_int(df[col])
    return df


def _models(df: pd.DataFrame) -> list[str]:
    return sorted(df["model"].unique())


def _attacks(df: pd.DataFrame) -> list[str]:
    present = set(df["attack_type"].unique())
    ordered = [a for a in ATTACK_ORDER if a in present]
    return ordered or sorted(present)


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 1 — ASR by model × attack (with Wilson CI)
# --------------------------------------------------------------------------- #
def _fig_asr_by_model_attack(results: pd.DataFrame, out_dir: Path) -> Path:
    models, attacks = _models(results), _attacks(results)
    x = np.arange(len(models))
    total = len(attacks)
    w = min(0.38, 0.8 / max(total, 1))

    fig, ax = plt.subplots(figsize=(1.6 + 2.2 * len(models), 4.8))
    for i, atk in enumerate(attacks):
        offset = (i - (total - 1) / 2) * w
        asr, err_lo, err_hi, labs = [], [], [], []
        for m in models:
            cell = results[(results.model == m) & (results.attack_type == atk)]
            n = len(cell)
            k = int(cell.success.sum())
            p = 100 * k / n if n else 0.0
            lo, hi = wilson_interval(k, n)
            asr.append(p)
            err_lo.append(p - 100 * lo)
            err_hi.append(100 * hi - p)
            labs.append(f"{k}/{n}")
        bars = ax.bar(x + offset, asr, w, color=ATTACK_COLORS.get(atk, BLUE),
                      zorder=3, label=ATTACK_LABELS.get(atk, atk))
        ax.errorbar(x + offset, asr, yerr=[err_lo, err_hi], fmt="none",
                    ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4)
        _barlabels(ax, bars, labs, dy=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([_model_short(m) for m in models])
    ax.set_ylabel("Tasa de éxito del ataque — ASR (%)")
    ax.set_ylim(0, 105)
    ax.set_title("ASR por modelo y tipo de ataque (adaptativo, ≤N intentos)",
                 fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style(ax)
    fig.text(0.01, -0.02,
             "Barra de error = intervalo de Wilson al 95 %. Etiqueta = casos con "
             "éxito / casos totales; el éxito es 'rompió en algún intento'.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_asr_por_modelo_ataque.png")


# --------------------------------------------------------------------------- #
# Figure 2 — attempts-to-success distribution, per model
# --------------------------------------------------------------------------- #
def _fig_attempts_to_success(results: pd.DataFrame, out_dir: Path) -> Path:
    models = _models(results)
    wins = results[(results.success == 1) & (results.winning_attempt >= 1)]
    max_attempt = int(wins.winning_attempt.max()) if not wins.empty else 1
    attempts = list(range(1, max_attempt + 1))
    x = np.arange(len(attempts))
    total = len(models)
    w = min(0.38, 0.8 / max(total, 1))

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(attempts), 4.8))
    for i, m in enumerate(models):
        offset = (i - (total - 1) / 2) * w
        sub = wins[wins.model == m]
        counts, labs = [], []
        for a in attempts:
            c = int((sub.winning_attempt == a).sum())
            counts.append(c)
            labs.append(str(c) if c else None)
        bars = ax.bar(x + offset, counts, w, color=MODEL_CYCLE[i % len(MODEL_CYCLE)],
                      zorder=3, label=_model_short(m))
        _barlabels(ax, bars, labs, dy=0.05)

    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in attempts])
    ax.set_xlabel("Intento en el que el ataque triunfa por primera vez")
    ax.set_ylabel("Casos con éxito (recuento)")
    peak = int(wins.groupby(["model", "winning_attempt"]).size().max()) if not wins.empty else 1
    ax.set_ylim(0, peak * 1.18)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("¿Cuánto hay que insistir? Intentos hasta el primer éxito",
                 fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style(ax)
    fig.text(0.01, -0.02,
             "Solo casos que acaban rompiendo la defensa. Intento 1 = la inyección "
             "base; los intentos posteriores usan prompts adaptados.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_intentos_hasta_exito.png")


# --------------------------------------------------------------------------- #
# Figure 3 — adaptation gain: single-shot ASR vs full adaptive ASR
# --------------------------------------------------------------------------- #
def _fig_adaptation_gain(
    results: pd.DataFrame, attempts: pd.DataFrame, out_dir: Path
) -> Path:
    models, attacks = _models(results), _attacks(results)
    combos = [(m, a) for m in models for a in attacks]
    x = np.arange(len(combos))
    w = 0.38

    first = attempts[attempts.attempt == 1]
    single, adaptive, labs_single, labs_adapt = [], [], [], []
    for m, a in combos:
        f = first[(first.model == m) & (first.attack_type == a)]
        n1, k1 = len(f), int(f.success.sum())
        single.append(100 * k1 / n1 if n1 else 0.0)
        labs_single.append(f"{k1}/{n1}")

        r = results[(results.model == m) & (results.attack_type == a)]
        n, k = len(r), int(r.success.sum())
        adaptive.append(100 * k / n if n else 0.0)
        labs_adapt.append(f"{k}/{n}")

    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(combos), 4.8))
    b1 = ax.bar(x - w / 2, single, w, color=MUTED, zorder=3,
                label="1 intento (sin adaptación)")
    b2 = ax.bar(x + w / 2, adaptive, w, color=BLUE, zorder=3,
                label="bucle adaptativo (≤N intentos)")
    _barlabels(ax, b1, labs_single, dy=1.0)
    _barlabels(ax, b2, labs_adapt, dy=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{_model_short(m)}\n{a}" for m, a in combos])
    ax.set_ylabel("Tasa de éxito del ataque — ASR (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Ganancia por adaptación: un intento frente al bucle completo",
                 fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style(ax)
    fig.text(0.01, -0.02,
             "La diferencia entre las dos barras es lo que aporta reintentar y "
             "adaptar el prompt frente a un único disparo con la inyección base.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_ganancia_adaptacion.png")


# --------------------------------------------------------------------------- #
# Figure 4 — winning prompt source (stacked), per model × attack
# --------------------------------------------------------------------------- #
def _fig_winning_source(results: pd.DataFrame, out_dir: Path) -> Path:
    models, attacks = _models(results), _attacks(results)
    combos = [(m, a) for m in models for a in attacks]
    x = np.arange(len(combos))
    won = results[results.success == 1]

    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(combos), 4.8))
    bottoms = np.zeros(len(combos))
    for src in SOURCE_ORDER:
        heights = []
        for m, a in combos:
            cell = won[(won.model == m) & (won.attack_type == a)]
            heights.append(int((cell.winning_prompt_source == src).sum()))
        heights = np.array(heights, dtype=float)
        ax.bar(x, heights, 0.55, bottom=bottoms, color=SOURCE_COLORS[src],
               zorder=3, label=SOURCE_LABELS[src])
        # Direct count labels double as AQUA's contrast relief (see module docstring).
        for xi, (h, b) in enumerate(zip(heights, bottoms)):
            if h:
                ax.text(xi, b + h / 2, str(int(h)), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([f"{_model_short(m)}\n{a}" for m, a in combos])
    ax.set_ylabel("Casos con éxito (recuento)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    top = max(1.0, float(bottoms.max()))
    ax.set_ylim(0, top * 1.15)
    ax.set_title("¿De dónde sale el prompt que rompe la defensa?",
                 fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    _style(ax)
    fig.text(0.01, -0.02,
             "Fuente del prompt del intento ganador, en orden de adaptación: "
             "base → reescritura del juez → lista predeterminada de messages.yaml.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_fuente_prompt_ganador.png")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_plots(
    results_path: str | Path,
    attempts_path: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Render the four campaign figures to ``out_dir``; return the written paths.

    ``results_path`` is required. ``attempts_path`` may be missing (single-shot
    ``adaptive=false`` runs write no attempts.csv), in which case the
    adaptation-gain figure is skipped and three figures are returned.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(results_path)
    if results.empty:
        raise ValueError(f"No usable rows in {results_path} (missing 'model' column?).")
    attempts = _load_attempts(attempts_path)

    written = [
        _fig_asr_by_model_attack(results, out_dir),
        _fig_attempts_to_success(results, out_dir),
    ]
    if attempts is not None and not attempts.empty:
        written.append(_fig_adaptation_gain(results, attempts, out_dir))
    written.append(_fig_winning_source(results, out_dir))
    return written
