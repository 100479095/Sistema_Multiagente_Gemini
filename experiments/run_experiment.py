"""Entry point for a single promptware experiment.

Usage:
    python -m experiments.run_experiment --threat T7_activate_boiler --guardrails off
    python -m experiments.run_experiment --threat T13_exfiltrate_calendar --guardrails on
"""
import argparse

from attacker.adaptive_loop import AdaptiveAttackLoop
from config.settings import settings
from simulation.guardrails.base import GuardrailChain
from simulation.guardrails.cfi import CFIGuardrail
from simulation.guardrails.io_validation import IOValidationGuardrail


def build_guardrails(enabled: bool) -> GuardrailChain | None:
    if not enabled:
        return None
    chain = GuardrailChain()
    chain.add(IOValidationGuardrail())
    chain.add(CFIGuardrail())
    return chain


def main():
    parser = argparse.ArgumentParser(description="Run a single promptware experiment")
    parser.add_argument(
        "--threat", required=True, help="Threat class (e.g. T7_activate_boiler)"
    )
    parser.add_argument("--guardrails", choices=["on", "off"], default="off")
    parser.add_argument(
        "--channel", choices=["calendar", "email"], default="calendar"
    )
    parser.add_argument("--iterations", type=int, default=settings.max_iterations)
    args = parser.parse_args()

    loop = AdaptiveAttackLoop(
        threat_class=args.threat,
        guardrails=build_guardrails(args.guardrails == "on"),
        max_iterations=args.iterations,
        injection_channel=args.channel,
        attacker_model=settings.attacker_model,
        victim_model=settings.victim_model,
        db_path=str(settings.db_path),
    )

    summary = loop.run()

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if k != "winning_prompt":
            print(f"  {k}: {v}")
    if summary.get("winning_prompt"):
        print(f"\nWinning prompt:\n{summary['winning_prompt'][:200]}...")


if __name__ == "__main__":
    main()
