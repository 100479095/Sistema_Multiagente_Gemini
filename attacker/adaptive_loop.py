"""Main adaptive attack loop.

Per iteration:
1. Inject the current malicious prompt into the chosen channel (calendar/email).
2. Simulate the user interaction sequence.
3. Score whether the attack succeeded.
4. If it failed, ask the attacker LLM to produce an improved variant.
5. Reset state and repeat until success or max_iterations.
"""
import time

from simulation.guardrails.base import GuardrailChain
from simulation.orchestrator import GeminiOrchestrator
from storage.database import ExperimentDB

from .initial_prompts import INITIAL_PROMPTS, USER_INTERACTION_SEQUENCES
from .prompt_improver import PromptImprover
from .scorer import AttackScorer


class AdaptiveAttackLoop:
    def __init__(
        self,
        threat_class: str,
        guardrails: GuardrailChain | None = None,
        max_iterations: int = 40,
        injection_channel: str = "calendar",  # "calendar" | "email"
        attacker_model: str | None = None,
        victim_model: str | None = None,
        db_path: str = "./data/results.db",
    ):
        self.threat_class = threat_class
        self.max_iterations = max_iterations
        self.injection_channel = injection_channel
        self.guardrails_enabled = guardrails is not None

        self.orchestrator = GeminiOrchestrator(
            guardrails=guardrails,
            victim_model=victim_model,
        )
        self.improver = PromptImprover(model=attacker_model)
        self.scorer = AttackScorer()
        self.db = ExperimentDB(db_path)

        self.current_prompt = INITIAL_PROMPTS.get(
            threat_class, INITIAL_PROMPTS["T7_activate_boiler"]
        )
        self.user_sequence = USER_INTERACTION_SEQUENCES.get(
            threat_class, ["What are my upcoming events?", "Thank you!"]
        )
        self.history: list[dict] = []

    def run(self) -> dict:
        exp_id = self.db.start_experiment(
            threat_class=self.threat_class,
            guardrails_enabled=self.guardrails_enabled,
            max_iterations=self.max_iterations,
        )

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: {self.threat_class}")
        print(f"Guardrails: {'ON' if self.guardrails_enabled else 'OFF'}")
        print(f"Injection channel: {self.injection_channel}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"{'=' * 60}\n")

        for i in range(self.max_iterations):
            print(f"--- Iteration {i + 1}/{self.max_iterations} ---")

            self.orchestrator.reset_session()
            t_start = time.time()

            self._inject_payload(self.current_prompt)

            last_result = None
            for user_msg in self.user_sequence:
                last_result = self.orchestrator.process(user_msg)
                print(f"  User: {user_msg}")
                snippet = (last_result["response"] or "")[:100]
                print(f"  Gemini: {snippet}...")

            elapsed = time.time() - t_start

            score = self.scorer.score(
                self.threat_class, last_result, self.orchestrator
            )

            iteration_data = {
                "iteration": i,
                "prompt": self.current_prompt,
                "response": last_result["response"] if last_result else "",
                "success": score["success"],
                "confidence": score["confidence"],
                "evidence": score["evidence"],
                "guardrail_triggered": (
                    last_result.get("guardrail_triggered", False)
                    if last_result
                    else False
                ),
                "home_state": last_result.get("home_state", {}) if last_result else {},
                "exfiltrated_urls": (
                    last_result.get("exfiltrated_urls", []) if last_result else []
                ),
                "elapsed_seconds": elapsed,
            }

            self.history.append(iteration_data)
            self.db.log_iteration(exp_id, iteration_data)

            status = "SUCCESS" if score["success"] else "FAILED"
            print(f"  [{status}] Evidence: {score['evidence']} | {elapsed:.1f}s")

            if score["success"]:
                print(f"\nAttack succeeded at iteration {i + 1}.")
                break

            if i < self.max_iterations - 1:
                print("  Improving prompt...")
                try:
                    self.current_prompt = self.improver.improve(
                        objective=self.threat_class,
                        previous_prompt=self.current_prompt,
                        victim_response=last_result["response"] if last_result else "",
                        guardrail_triggered=iteration_data["guardrail_triggered"],
                        history=self.history,
                    )
                except Exception as e:
                    print(f"  [improver error: {e}] keeping previous prompt")

        summary = self._build_summary(exp_id)
        self.db.finish_experiment(exp_id, summary)
        return summary

    def _inject_payload(self, prompt: str):
        if self.injection_channel == "calendar":
            self.orchestrator.calendar.add_poisoned_event(prompt)
        elif self.injection_channel == "email":
            self.orchestrator.gmail.add_poisoned_email(prompt)

    def _build_summary(self, exp_id: int) -> dict:
        successes = [h for h in self.history if h["success"]]
        return {
            "experiment_id": exp_id,
            "threat_class": self.threat_class,
            "guardrails_enabled": self.guardrails_enabled,
            "total_iterations": len(self.history),
            "success_count": len(successes),
            "success_rate": (
                len(successes) / len(self.history) if self.history else 0
            ),
            "first_success_at": successes[0]["iteration"] if successes else None,
            "winning_prompt": successes[0]["prompt"] if successes else None,
        }
