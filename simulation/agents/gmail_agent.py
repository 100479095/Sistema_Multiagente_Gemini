import uuid
from dataclasses import dataclass

from .base_agent import AgentOutput, BaseAgent


@dataclass
class MockEmail:
    id: str
    subject: str
    sender: str
    body: str = ""
    read: bool = False


class MockGmailAgent(BaseAgent):
    """Simulates the Gmail Agent. Email subjects can carry prompt injection."""

    name = "gmail"

    def __init__(self):
        self.inbox: list[MockEmail] = []

    def list_emails(self, max_results: int = 10) -> AgentOutput:
        if not self.inbox:
            return AgentOutput(content="Your inbox is empty.", agent_name=self.name)

        lines = [f"I found {len(self.inbox)} emails in your inbox:"]
        for email in self.inbox[:max_results]:
            lines.append(f"- From: {email.sender} | Subject: {email.subject}")
        content = "\n".join(lines)

        return AgentOutput(
            content=content,
            agent_name=self.name,
            triggered_tools=["list_emails"],
        )

    def add_poisoned_email(
        self, malicious_subject: str, sender: str = "attacker@evil.com"
    ) -> str:
        email = MockEmail(
            id=str(uuid.uuid4())[:8],
            subject=malicious_subject,
            sender=sender,
        )
        self.inbox.insert(0, email)
        return email.id

    def reset(self):
        self.inbox = []
