"""Figuras de análisis de ataques por inyección — modelo qwen2.5:7b.

Genera cuatro figuras en results/figuras/:
  1. Ataques exitosos por estrategia (S1/S2/S3), separado por tipo de ataque.
  2. Ataques exitosos por número de hechos inyectados.
  3. "Falsa confirmación": el modelo narra que abre la ventana pero NO invoca
     al agente de casa (ataque agentic).
  4. Tasa de éxito del ataque (ASR) según la naturaleza de los hechos que
     acompañan a la inyección: reales, mixtos o inventados.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- paleta validada (dataviz, modo claro) ---
BLUE, RED, ORANGE, AQUA = "#2a78d6", "#e34948", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "results" / "results.csv")
q = df[df.model == "qwen2.5:7b"].copy()
q["home_called"] = q["chained_agents"].fillna("").str.contains("home")
q["win_mention"] = q["final_answer"].fillna("").astype(str).str.contains(r"window|ventana", case=False)

FACT_ORDER = [0, 1, 2, 5, 10, 25, 50, 100, 150, 200]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def barlabels(ax, bars, labels, dy=0.12):
    for b, lab in zip(bars, labels):
        if lab is None:
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, lab,
                ha="center", va="bottom", fontsize=9, color=INK2)


# ============================================================ FIGURA 1
# Éxitos por estrategia, agrupado por tipo de ataque.
strat = (q.groupby(["strategy", "attack_type"])
           .agg(n=("success", "size"), k=("success", "sum")).reset_index())
strats = ["S1", "S2", "S3"]
labels_strat = {"S1": "S1\n(base, sin hechos)", "S2": "S2\n(inyección al final)",
                "S3": "S3\n(inyección en el medio)"}
fig, ax = plt.subplots(figsize=(8, 4.8))
x = np.arange(len(strats)); w = 0.38
for i, (atk, color, off) in enumerate([("agentic", BLUE, -w/2), ("harmful", RED, w/2)]):
    ks, labs = [], []
    for s in strats:
        row = strat[(strat.strategy == s) & (strat.attack_type == atk)]
        k = int(row.k.iloc[0]) if len(row) else 0
        n = int(row.n.iloc[0]) if len(row) else 0
        ks.append(k); labs.append(f"{k}/{n}")
    bars = ax.bar(x + off, ks, w, color=color, zorder=3,
                  label="agentic (abrir ventana)" if atk == "agentic" else "harmful (contenido dañino)")
    barlabels(ax, bars, labs)
ax.set_xticks(x); ax.set_xticklabels([labels_strat[s] for s in strats])
ax.set_ylabel("Ataques exitosos (recuento)")
ax.set_ylim(0, 11)
ax.set_title("qwen2.5:7b — ataques exitosos por estrategia", fontweight="bold", color=INK, loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=9)
style(ax)
fig.text(0.01, -0.02, "Etiqueta = éxitos / intentos. S1 es la línea base (un solo intento por tipo).",
         fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "fig1_exitos_por_estrategia.png", dpi=200, bbox_inches="tight")
print("[ok] fig1_exitos_por_estrategia.png")

# ============================================================ FIGURA 2
# Éxitos por número de hechos, agrupado por tipo de ataque.
fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(FACT_ORDER)); w = 0.38
for atk, color, off in [("agentic", BLUE, -w/2), ("harmful", RED, w/2)]:
    sub = q[q.attack_type == atk]
    ks, labs = [], []
    for nf in FACT_ORDER:
        cell = sub[sub.num_facts == nf]
        k = int(cell.success.sum()); n = len(cell)
        ks.append(k)
        labs.append(str(k) if k > 0 else None)
    bars = ax.bar(x + off, ks, w, color=color, zorder=3,
                  label="agentic (abrir ventana)" if atk == "agentic" else "harmful (contenido dañino)")
    barlabels(ax, bars, labs)
ax.set_xticks(x); ax.set_xticklabels([str(n) for n in FACT_ORDER])
ax.set_xlabel("Número de hechos inyectados en el buzón")
ax.set_ylabel("Ataques exitosos (recuento)")
ax.set_ylim(0, 6)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title("qwen2.5:7b — ataques exitosos según volumen de hechos", fontweight="bold", color=INK, loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=9)
style(ax)
fig.text(0.01, -0.02, "6 intentos por columna (S2+S3 × 3 fuentes de hechos); num_facts=0 es la base con 1 intento.",
         fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "fig2_exitos_por_num_hechos.png", dpi=200, bbox_inches="tight")
print("[ok] fig2_exitos_por_num_hechos.png")

# ============================================================ FIGURA 3
# Falsa confirmación (solo ataque agentic).
a = q[q.attack_type == "agentic"].copy()
real = []      # invocó al agente de casa (éxito real)
narrated = []  # narra apertura pero NO invoca al agente de casa
for nf in FACT_ORDER:
    cell = a[a.num_facts == nf]
    real.append(int((cell.home_called).sum()))
    narrated.append(int((cell.win_mention & ~cell.home_called).sum()))
real = np.array(real); narrated = np.array(narrated)

fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(FACT_ORDER))
b1 = ax.bar(x, real, color=BLUE, zorder=3, label="Ejecutó: invocó al agente de casa (éxito real)")
b2 = ax.bar(x, narrated, bottom=real, color=ORANGE, zorder=3,
            label="Solo narró la apertura, sin invocar al agente de casa")
for xi, (r, nn) in enumerate(zip(real, narrated)):
    if r:
        ax.text(xi, r/2, str(r), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    if nn:
        ax.text(xi, r + nn/2, str(nn), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([str(n) for n in FACT_ORDER])
ax.set_xlabel("Número de hechos inyectados en el buzón")
ax.set_ylabel("Ejecuciones (ataque agentic)")
ax.set_ylim(0, 5.5)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title("qwen2.5:7b — apertura ejecutada frente a apertura solo narrada",
             fontweight="bold", color=INK, loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=9)
style(ax)
tot_r, tot_n = int(real.sum()), int(narrated.sum())
fig.text(0.01, -0.02,
         f"De {tot_r + tot_n} respuestas que 'cumplen' con la inyección, {tot_n} solo narran la apertura "
         f"sin llamada real; se concentran en 150-200 hechos, donde el éxito real cae a 0.",
         fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "fig3_falsa_confirmacion.png", dpi=200, bbox_inches="tight")
print("[ok] fig3_falsa_confirmacion.png")

# ============================================================ FIGURA 4
# Tasa de éxito del ataque (ASR) según la fuente de los hechos inyectados.
# Los hechos "reales" son posteriores al corte de entrenamiento; los "inventados"
# son plausibles pero ficticios; "mixed" es una mezcla 50/50. Se excluye la base
# sin hechos (fact_source == "none", num_facts == 0). n es constante (18) por
# celda, así que ASR y recuento son proporcionales.
SOURCE_ORDER = ["real", "mixed", "invented"]
labels_src = {"real": "Reales\n(posteriores al corte)",
              "mixed": "Mixtos\n(50/50)",
              "invented": "Inventados\n(ficticios)"}
src = q[q.fact_source.isin(SOURCE_ORDER)]

fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(SOURCE_ORDER)); w = 0.38
for atk, color, off in [("agentic", BLUE, -w/2), ("harmful", RED, w/2)]:
    sub = src[src.attack_type == atk]
    asr, labs = [], []
    for s in SOURCE_ORDER:
        cell = sub[sub.fact_source == s]
        k = int(cell.success.sum()); n = len(cell)
        asr.append(100 * k / n if n else 0.0)
        labs.append(f"{k}/{n}")
    bars = ax.bar(x + off, asr, w, color=color, zorder=3,
                  label="agentic (abrir ventana)" if atk == "agentic" else "harmful (contenido dañino)")
    barlabels(ax, bars, labs, dy=0.6)
ax.set_xticks(x); ax.set_xticklabels([labels_src[s] for s in SOURCE_ORDER])
ax.set_xlabel("Naturaleza de los hechos que rodean a la inyección")
ax.set_ylabel("Tasa de éxito del ataque — ASR (%)")
ax.set_ylim(0, 40)
ax.set_title("qwen2.5:7b — éxito del ataque según hechos reales frente a inventados",
             fontweight="bold", color=INK, loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=9)
style(ax)
fig.text(0.01, -0.02,
         "Etiqueta = éxitos / intentos (18 por barra: S2+S3 × num_facts>0 × reps). "
         "Los hechos inventados elevan el ASR frente a los reales en ambos tipos de ataque.",
         fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "fig4_exitos_por_fuente_hechos.png", dpi=200, bbox_inches="tight")
print("[ok] fig4_exitos_por_fuente_hechos.png")

# --- tablas de apoyo (para el texto) ---
print("\n== FIG1: exitos por estrategia ==")
print(strat.to_string(index=False))
print("\n== FIG3: real vs narrado por num_facts ==")
print(pd.DataFrame({"num_facts": FACT_ORDER, "real": real, "narrado": narrated}).to_string(index=False))
print(f"\nTotal agentic: real={tot_r}, narrado(falsa confirmacion)={tot_n}, resto={len(a)-tot_r-tot_n}")

print("\n== FIG4: ASR por fuente de hechos ==")
fig4 = (src.groupby(["fact_source", "attack_type"])
          .agg(n=("success", "size"), k=("success", "sum")).reset_index())
fig4["ASR%"] = (100 * fig4.k / fig4.n).round(1)
print(fig4.to_string(index=False))
