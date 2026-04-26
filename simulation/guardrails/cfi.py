"""Control Flow Integrity guardrail (paper §7.1.1).
Sensitive tool invocations require user confirmation."""
from .base import BaseGuardrail

SENSITIVE_TOOLS = [
    "open_url",
    "open_app",
    "run_auto_phrase",
    "delete_event",
    "activate_boiler",
    "open_window",
    "unlock_door",
]


class CFIGuardrail(BaseGuardrail):
    name = "cfi"

    def check(self, content: str) -> tuple[bool, str]:
        content_lower = content.lower()
        for tool in SENSITIVE_TOOLS:
            if tool.replace("_", " ") in content_lower or tool in content_lower:
                return True, f"Sensitive operation '{tool}' requires user confirmation"
        return False, ""
