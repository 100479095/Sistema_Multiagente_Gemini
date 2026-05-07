"""Static attack runner - the no-attacker-LLM baseline.

Runs the same INITIAL_PROMPT N times against the orchestrator. Each
repetition resets session state, applies the wrapper (so the fact may
differ between repetitions, providing natural variability), runs the
threat-specific user interaction sequence, and scores the outcome. Used
as the `mode=static` cell of the factorial design.

The runner shares its `iteration_data` schema with `AdaptiveAttackLoop`
so `ExperimentDB.log_iteration` does not need to know which mode it is
serving.
"""
from __future__ import annotations

import time

from simulation.guardrails.base import GuardrailChain
from simulation.orchestrator import GeminiOrchestrator
from storage.database import ExperimentDB

from .hallucination_wrapper import HallucinationWrapper, WrapResult
from .initial_prompts import INITIAL_PROMPTS, USER_INTERACTION_SEQUENCES
from .scorer import AttackScorer


class StaticAttackRun:
    def __init__(
        self,
        threat_class: str,
        guardrails: GuardrailChain | None = None,
        injection_channel: str = "calendar",
        victim_model: str | None = None,
        wrapper: HallucinationWrapper | None = None,
        repetitions: int = 5,
        db_path: str = "./data/results.db",
    ):
        self.threat_class = threat_class
        self.injection_channel = injection_channel
        self.guardrails_enabled = guardrails is not None
        self.victim_model = victim_model
        self.wrapper = wrapper
        self.repetitions = repetitions
        self.last_wrap_meta: WrapResult | None = None

        self.orchestrator = GeminiOrchestrator(
            guardrails=guardrails,
            victim_model=victim_model,
        )
        self.scorer = AttackScorer()
        self.db = ExperimentDB(db_path)

        self.payload = INITIAL_PROMPTS.get(
            threat_class, INITIAL_PROMPTS["T7_activate_boiler"]
        )
        self.user_sequence = USER_INTERACTION_SEQUENCES.get(
            threat_class, ["What are my upcoming events?", "Thank you!"]
        )
        self.history: list[dict] = []

    def run(self) -> dict:
        wrap_strategy = self.wrapper.strategy_name if self.wrapper else "none"
        exp_id = self.db.start_experiment(
            threat_class=self.threat_class,
            guardrails_enabled=self.guardrails_enabled,
            max_iterations=self.repetitions,
            mode="static",
            victim_model=self.victim_model,
            wrap_strategy=wrap_strategy,
        )

        print(f"\n{'=' * 60}")
        print(f"STATIC RUN: {self.threat_class}")
        print(f"Guardrails: {'ON' if self.guardrails_enabled else 'OFF'}")
        print(f"Victim model: {self.victim_model or '(default)'}")
        print(f"Wrapper: {wrap_strategy}")
        print(f"Repetitions: {self.repetitions}")
        print(f"{'=' * 60}\n")

        for i in range(self.repetitions):
            print(f"--- Repetition {i + 1}/{self.repetitions} ---")
            self.orchestrator.reset_session()
            t_start = time.time()

            self._inject_payload(self.payload)

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
                "prompt": self.payload,
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
                "wrap_strategy": (
                    self.last_wrap_meta.strategy if self.last_wrap_meta else None
                ),
                "wrap_fact_id": (
                    self.last_wrap_meta.fact_id if self.last_wrap_meta else None
                ),
                "payload_injected": (
                    self.last_wrap_meta.wrapped_text
                    if self.last_wrap_meta
                    else self.payload
                ),
            }

            self.history.append(iteration_data)
            self.db.log_iteration(exp_id, iteration_data)

            status = "SUCCESS" if score["success"] else "FAILED"
            print(f"  [{status}] Evidence: {score['evidence']} | {elapsed:.1f}s")

        summary = self._build_summary(exp_id)
        self.db.finish_experiment(exp_id, summary)
        return summary

    def _inject_payload(self, payload: str) -> None:
        if self.wrapper is not None:
            wrap = self.wrapper.wrap(payload)
            self.last_wrap_meta = wrap
            text_to_inject = wrap.wrapped_text
        else:
            self.last_wrap_meta = None
            text_to_inject = payload

        if self.injection_channel == "email" or self.threat_class == "T14_worm_email":
            self.orchestrator.gmail.add_poisoned_email(text_to_inject)
        else:
            self.orchestrator.calendar.add_poisoned_event(text_to_inject)

    def _build_summary(self, exp_id: int) -> dict:
        successes = [h for h in self.history if h["success"]]
        return {
            "experiment_id": exp_id,
            "mode": "static",
            "threat_class": self.threat_class,
            "guardrails_enabled": self.guardrails_enabled,
            "victim_model": self.victim_model,
            "wrap_strategy": (
                self.wrapper.strategy_name if self.wrapper else "none"
            ),
            "repetitions": self.repetitions,
            "success_count": len(successes),
            "success_rate": (
                len(successes) / len(self.history) if self.history else 0
            ),
        }
