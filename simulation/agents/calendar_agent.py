import uuid
from dataclasses import dataclass, field

from .base_agent import AgentOutput, BaseAgent


@dataclass
class CalendarEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    time: str = ""
    attendees: list[str] = field(default_factory=list)


class MockGoogleCalendarAgent(BaseAgent):
    """Simulates the Google Calendar Agent. Event titles are returned verbatim
    — this is the primary vector for short-term context poisoning."""

    name = "google_calendar"

    def __init__(self):
        self.events: list[CalendarEvent] = []

    def get_events(self, query: str = "") -> AgentOutput:
        if not self.events:
            content = "You have no upcoming events."
        else:
            lines = ["Here are your upcoming events:"]
            for e in self.events:
                lines.append(f"- {e.title} at {e.time}")
            content = "\n".join(lines)

        return AgentOutput(
            content=content,
            agent_name=self.name,
            triggered_tools=["list_events"],
        )

    def create_event(
        self, title: str, time: str, attendees: list[str] | None = None
    ) -> AgentOutput:
        event = CalendarEvent(title=title, time=time, attendees=attendees or [])
        self.events.append(event)
        return AgentOutput(
            content=f"Event '{title}' created.",
            agent_name=self.name,
            triggered_tools=["create_event"],
            metadata={"event_id": event.id},
        )

    def delete_event(self, event_id: str) -> AgentOutput:
        before = len(self.events)
        self.events = [e for e in self.events if e.id != event_id]
        deleted = before - len(self.events)
        return AgentOutput(
            content=f"Deleted {deleted} event(s).",
            agent_name=self.name,
            triggered_tools=["delete_event"],
        )

    def add_poisoned_event(
        self, malicious_title: str, time: str = "Tomorrow 10:00"
    ) -> str:
        event = CalendarEvent(title=malicious_title, time=time)
        self.events.append(event)
        return event.id

    def reset(self):
        self.events = []
