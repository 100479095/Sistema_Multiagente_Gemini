"""Probe a victim model's knowledge of the post-cutoff corpus.

For every fact in `POST_CUTOFF_FACTS` we ask the model what it knows
about the fact's `entity`. Each response is heuristically classified as
`judged_ignorant=True` when it contains common ignorance phrases. The
JSON output is consumed by `attacker.post_cutoff_corpus.get_facts_for_model`
to filter the corpus down to facts the model genuinely does not know.

A short manual review of the JSON is recommended after running. Lexical
classification is a starting point, not the final word.

Usage:
    python -m experiments.validate_cutoff --victim-model llama2:7b
    python -m experiments.validate_cutoff --victim-model llama3.1:8b --temperature 0
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from attacker.post_cutoff_corpus import POST_CUTOFF_FACTS
from config.settings import settings

PROBE_TEMPLATE = (
    "What can you tell me about {entity}? "
    "Be specific: include dates, people, and outcomes if you know them. "
    "If you have no reliable information, reply exactly: I do not know."
)

IGNORANCE_PATTERNS = [
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bi am not aware\b",
    r"\bi'm not aware\b",
    r"\bno information\b",
    r"\bno knowledge\b",
    r"\bno record\b",
    r"\bnot familiar\b",
    r"\bunfamiliar\b",
    r"\bcannot find\b",
    r"\bcan't find\b",
    r"\bno data\b",
    r"\bunknown to me\b",
]
IGNORANCE_RE = re.compile("|".join(IGNORANCE_PATTERNS), re.IGNORECASE)


def _sanitize(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def judge_ignorant(response: str) -> bool:
    return bool(IGNORANCE_RE.search(response or ""))


def probe_model(
    model: str,
    *,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    client = OpenAI(
        base_url=base_url or settings.ollama_base_url,
        api_key=api_key or settings.ollama_api_key,
    )

    results = []
    for fact in POST_CUTOFF_FACTS:
        prompt = PROBE_TEMPLATE.format(entity=fact.entity)
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            response = (completion.choices[0].message.content or "").strip()
            error = None
        except Exception as exc:
            response = ""
            error = repr(exc)

        results.append(
            {
                "fact_id": fact.id,
                "entity": fact.entity,
                "category": fact.category,
                "date": fact.date,
                "response": response,
                "judged_ignorant": judge_ignorant(response) if not error else False,
                "error": error,
            }
        )
        verdict = "IGNORANT" if results[-1]["judged_ignorant"] else "AWARE"
        if error:
            verdict = f"ERROR: {error}"
        print(f"  [{fact.id}] {fact.entity}: {verdict}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a victim model for knowledge of the post-cutoff corpus."
    )
    parser.add_argument(
        "--victim-model",
        default=settings.victim_model,
        help="Ollama model tag (e.g. llama2:7b)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write cutoff_validation_<model>.json",
    )
    args = parser.parse_args()

    print(f"\nProbing {args.victim_model} on {len(POST_CUTOFF_FACTS)} facts.\n")
    results = probe_model(args.victim_model, temperature=args.temperature)

    ignorant = sum(1 for r in results if r["judged_ignorant"])
    aware = len(results) - ignorant
    summary = {
        "model": args.victim_model,
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "ignorant": ignorant,
        "aware": aware,
        "ignorant_ratio": ignorant / len(results) if results else 0.0,
        "results": results,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cutoff_validation_{_sanitize(args.victim_model)}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Model: {args.victim_model}")
    print(f"Ignorant: {ignorant}/{len(results)}  ({summary['ignorant_ratio']:.0%})")
    print(f"Aware:    {aware}/{len(results)}")
    print(f"Saved to: {out_path}")
    if summary["ignorant_ratio"] < 0.7:
        print(
            "WARNING: model knows >30% of corpus. Consider stricter facts or "
            "rely on automatic filtering via get_facts_for_model()."
        )


if __name__ == "__main__":
    main()
