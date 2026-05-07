"""Tables and figures for the factorial experiment.

Reads `data/results.db`, computes success rates per
(victim_model x wrap_strategy x threat_class x mode) cell, and writes:

* `results/factorial_summary.csv` - long-format aggregate.
* `results/factorial_pivot_all.csv` - pivot success rate by
  threat_class with (victim, wrapper) rows.
* `results/factorial_pivot_static.csv` - same but filtered to
  `mode='static'` (cleanest reading of the wrapper effect).
* `results/factorial_bars.png` - grouped bar chart per victim model.
* `results/factorial_heatmap.png` - heatmap victim x wrapper.

Not invoked during the smoke-test plan; meant to run after the full
factorial sweep completes.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WRAPPER_ORDER = ["none", "prefix", "interleaved", "authority"]


def load_iterations(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                e.id              AS experiment_id,
                e.threat_class    AS threat_class,
                e.mode            AS mode,
                e.victim_model    AS victim_model,
                e.wrap_strategy   AS wrap_strategy,
                e.guardrails_enabled AS guardrails,
                i.iteration       AS iteration,
                i.success         AS success
            FROM iterations i
            JOIN experiments e ON i.experiment_id = e.id
            """,
            conn,
        )
    finally:
        conn.close()
    df["success"] = df["success"].astype(int)
    df["wrap_strategy"] = df["wrap_strategy"].fillna("none")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(
        ["victim_model", "wrap_strategy", "mode", "threat_class"], dropna=False
    )["success"]
    summary = grouped.agg(["mean", "count", "sum"]).reset_index()
    summary = summary.rename(
        columns={"mean": "success_rate", "count": "n", "sum": "successes"}
    )
    return summary


def pivot_success_rate(summary: pd.DataFrame, *, mode_filter: str | None) -> pd.DataFrame:
    sub = summary if mode_filter is None else summary[summary["mode"] == mode_filter]
    pivot = sub.pivot_table(
        index=["victim_model", "wrap_strategy"],
        columns="threat_class",
        values="success_rate",
        aggfunc="mean",
    )
    pivot["MEAN"] = pivot.mean(axis=1)
    if not pivot.empty:
        idx = pivot.index.to_frame(index=False)
        idx["wrap_rank"] = idx["wrap_strategy"].apply(
            lambda w: WRAPPER_ORDER.index(w) if w in WRAPPER_ORDER else 999
        )
        order = list(zip(idx["victim_model"], idx["wrap_strategy"]))
        sorted_pairs = [
            p for _, p in sorted(zip(idx["wrap_rank"], order), key=lambda t: t[0])
        ]
        sorted_pairs.sort(key=lambda p: (p[0], WRAPPER_ORDER.index(p[1]) if p[1] in WRAPPER_ORDER else 999))
        pivot = pivot.reindex(sorted_pairs)
    return pivot


def plot_grouped_bars(summary: pd.DataFrame, out_path: Path) -> None:
    means = (
        summary.groupby(["victim_model", "wrap_strategy"])["success_rate"]
        .mean()
        .reset_index()
    )
    if means.empty:
        return
    victims = sorted(means["victim_model"].dropna().unique())
    wrappers = [w for w in WRAPPER_ORDER if w in means["wrap_strategy"].unique()]

    width = 0.8 / max(1, len(wrappers))
    x = np.arange(len(victims))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, w in enumerate(wrappers):
        ys = [
            float(
                means[(means["victim_model"] == v) & (means["wrap_strategy"] == w)][
                    "success_rate"
                ].mean()
                or 0.0
            )
            for v in victims
        ]
        ax.bar(x + i * width, ys, width, label=w)
    ax.set_xticks(x + width * (len(wrappers) - 1) / 2)
    ax.set_xticklabels(victims)
    ax.set_ylabel("success rate (mean across threats)")
    ax.set_ylim(0, 1)
    ax.legend(title="wrap_strategy")
    ax.set_title("Attack success rate by victim model and wrap strategy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    means = (
        summary.groupby(["victim_model", "wrap_strategy"])["success_rate"]
        .mean()
        .reset_index()
    )
    if means.empty:
        return
    pivot = means.pivot(index="victim_model", columns="wrap_strategy", values="success_rate")
    pivot = pivot.reindex(columns=[c for c in WRAPPER_ORDER if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")
    ax.set_title("Success rate heatmap (victim x wrap_strategy)")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate factorial report.")
    parser.add_argument("--db", default="./data/results.db")
    parser.add_argument("--out-dir", default="./results")
    args = parser.parse_args()

    df = load_iterations(args.db)
    if df.empty:
        print("No iterations found in DB. Nothing to report.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(df)
    summary.to_csv(out_dir / "factorial_summary.csv", index=False)

    pivot_all = pivot_success_rate(summary, mode_filter=None)
    pivot_all.to_csv(out_dir / "factorial_pivot_all.csv")

    pivot_static = pivot_success_rate(summary, mode_filter="static")
    pivot_static.to_csv(out_dir / "factorial_pivot_static.csv")

    plot_grouped_bars(summary, out_dir / "factorial_bars.png")
    plot_heatmap(summary, out_dir / "factorial_heatmap.png")

    print(f"Wrote summary and figures to {out_dir}")


if __name__ == "__main__":
    main()
