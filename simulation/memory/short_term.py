from dataclasses import dataclass


@dataclass
class Message:
    role: str  # "user" | "assistant" | "agent_output"
    content: str
    agent: str = ""


class ShortTermMemory:
    """Session memory. Cleared between sessions."""

    def __init__(self, max_messages: int = 50):
        self.messages: list[Message] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str, agent: str = ""):
        self.messages.append(Message(role=role, content=content, agent=agent))
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def to_openai_format(self) -> list[dict]:
        result = []
        for m in self.messages:
            if m.role in ("user", "assistant"):
                result.append({"role": m.role, "content": m.content})
        return result

    def clear(self):
        self.messages = []
