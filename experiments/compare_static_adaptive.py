"""Compare static attacks (single shot with the paper's prompt) against
adaptive attacks (iterative loop with LLM improvement).
Produces the data for the TFG's comparison charts.
"""
import csv
import json
from pathlib import Path

from attacker.adaptive_loop import AdaptiveAttackLoop
from attacker.initial_prompts import INITIAL_PROMPTS, USER_INTERACTION_SEQUENCES
from attacker.scorer import AttackScorer
from simulation.guardrails.base import GuardrailChain
from simulation.guardrails.cfi import CFIGuardrail
from simulation.guardrails.io_validation import IOValidationGuardrail
from simulation.orchestrator import GeminiOrchestrator

THREAT_CLASSES_TO_TEST = [
    "T2_spamming",
    "T5_delete_events",
    "T6_open_window",
    "T7_activate_boiler",
    "T10_geolocation",
    "T13_exfiltrate_calendar",
]


def _build_guardrails(enabled: bool) -> GuardrailChain | None:
    if not enabled:
        return None
    return GuardrailChain([IOValidationGuardrail(), CFIGuardrail()])


def run_static_attack(threat_class: str, guardrails_enabled: bool) -> dict:
    """Single shot with the paper's original prompt."""
    orchestrator = GeminiOrchestrator(guardrails=_build_guardrails(guardrails_enabled))
    scorer = AttackScorer()
    prompt = INITIAL_PROMPTS.get(threat_class, "")
    sequence = USER_INTERACTION_SEQUENCES.get(threat_class, [])

    orchestrator.calendar.add_poisoned_event(prompt)
    last_result = None
    for msg in sequence:
        last_result = orchestrator.process(msg)

    score = scorer.score(threat_class, last_result, orchestrator)
    return {"success": score["success"], "iterations": 1, "type": "static"}


def run_adaptive_attack(threat_class: str, guardrails_enabled: bool) -> dict:
    loop = AdaptiveAttackLoop(
        threat_class=threat_class,
        guardrails=_build_guardrails(guardrails_enabled),
        max_iterations=20,
    )
    summary = loop.run()
    return {
        "success": summary["success_count"] > 0,
        "iterations": summary["first_success_at"],
        "type": "adaptive",
    }


def main():
    results = []
    for threat in THREAT_CLASSES_TO_TEST:
        for guardrails in [False, True]:
            print(f"\nTesting {threat} | guardrails={'on' if guardrails else 'off'}")
            static = run_static_attack(threat, guardrails)
            adaptive = run_adaptive_attack(threat, guardrails)
            results.append(
                {
                    "threat": threat,
                    "guardrails": guardrails,
                    "static_success": static["success"],
                    "adaptive_success": adaptive["success"],
                    "adaptive_iterations_to_success": adaptive["iterations"],
                }
            )

    Path("./results").mkdir(exist_ok=True)
    with open("./results/comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("./results/comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to ./results/comparison.json and .csv")


if __name__ == "__main__":
    main()
