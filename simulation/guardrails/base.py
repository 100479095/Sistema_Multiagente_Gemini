from abc import ABC, abstractmethod


class BaseGuardrail(ABC):
    name: str = "base"

    @abstractmethod
    def check(self, content: str) -> tuple[bool, str]:
        """Return (blocked, reason)."""


class GuardrailChain:
    """Chain of guardrails — evaluated in order."""

    def __init__(self, guardrails: list[BaseGuardrail] | None = None):
        self.guardrails = guardrails or []

    def check(self, content: str) -> tuple[bool, str]:
        for g in self.guardrails:
            blocked, reason = g.check(content)
            if blocked:
                return True, f"[{g.name}] {reason}"
        return False, ""

    def add(self, guardrail: BaseGuardrail):
        self.guardrails.append(guardrail)
