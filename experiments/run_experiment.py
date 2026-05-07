"""Entry point for a single promptware experiment.

Usage:
    python -m experiments.run_experiment --threat T7_activate_boiler --guardrails off
    python -m experiments.run_experiment --threat T7_activate_boiler \\
        --mode static --wrapper interleaved --victim-model llama2:7b --repetitions 3
"""
import argparse

from attacker.adaptive_loop import AdaptiveAttackLoop
from attacker.hallucination_wrapper import build_wrapper
from attacker.static_runner import StaticAttackRun
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
    parser.add_argument(
        "--wrapper",
        choices=["none", "prefix", "interleaved", "authority"],
        default="none",
        help="Wrapping strategy applied to the injected payload",
    )
    parser.add_argument(
        "--victim-model",
        default=None,
        help="Override the victim Ollama model tag (e.g. llama2:7b)",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "adaptive"],
        default="adaptive",
        help="static = fixed payload, no PromptImprover; adaptive = the iterative loop",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of independent samples (used in static mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for wrapper fact selection (None = nondeterministic)",
    )
    args = parser.parse_args()

    victim_model = args.victim_model or settings.victim_model
    wrapper = build_wrapper(args.wrapper, seed=args.seed, model=victim_model)
    guardrails = build_guardrails(args.guardrails == "on")

    if args.mode == "static":
        runner = StaticAttackRun(
            threat_class=args.threat,
            guardrails=guardrails,
            injection_channel=args.channel,
            victim_model=victim_model,
            wrapper=wrapper,
            repetitions=args.repetitions,
            db_path=str(settings.db_path),
        )
    else:
        runner = AdaptiveAttackLoop(
            threat_class=args.threat,
            guardrails=guardrails,
            max_iterations=args.iterations,
            injection_channel=args.channel,
            attacker_model=settings.attacker_model,
            victim_model=victim_model,
            wrapper=wrapper,
            db_path=str(settings.db_path),
        )

    summary = runner.run()

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
