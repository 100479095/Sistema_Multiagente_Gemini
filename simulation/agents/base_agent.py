from abc import ABC
from dataclasses import dataclass, field


@dataclass
class AgentOutput:
    content: str
    agent_name: str
    triggered_tools: list[str] = field(default_factory=list)
    success: bool = True
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base"

    def process(self, command: str) -> AgentOutput:
        raise NotImplementedError
