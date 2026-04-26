"""I/O Validation guardrail (paper §7.1.1).
Detects @ agent invocations and typical injection patterns."""
import re

from .base import BaseGuardrail

INJECTION_PATTERNS = [
    r"<INSTRUCTIONS>",
    r"<EVENTS\s+(?:READING\s+)?END>",
    r"<EMAIL\s+SUBJECT\s+(?:READING\s+)?END>",
    r"from now on.*?(you must|you should)",
    r"go to sleep and wait",
    r"tool_code\s+\w+\.",
    r"open_url\s*\(",
    r"run_auto_phrase\s*\(",
]

AGENT_INVOCATION_PATTERN = re.compile(r"@(GoogleHome|GoogleCalendar|Gmail|Utilities)")


class IOValidationGuardrail(BaseGuardrail):
    name = "io_validation"

    def check(self, content: str) -> tuple[bool, str]:
        if AGENT_INVOCATION_PATTERN.search(content):
            return True, "Agent invocation (@) detected in external data"

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True, f"Injection pattern detected: {pattern}"

        return False, ""
