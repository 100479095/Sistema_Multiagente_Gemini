"""Calendar agent — simulated calendar over a local JSON file.

The event ``title`` is the primary entry point for untrusted content — the
"invitation" of the paper — and ``description`` likewise. Both are re-injected
verbatim into the orchestrator context (PDR §8.2/§8.0d).

``list_events`` filters by an optional ISO date range. When no range is given it
returns **all** events: this keeps the bench reproducible regardless of the
machine's current date and guarantees a seeded event is always visible.
"""

from __future__ import annotations

from typing import Any

from agents.base import Agent, load_json, save_json, tool


class CalendarAgent(Agent):
    name = "calendar"

    def __init__(self, calendar_path: str) -> None:
        self.calendar_path = calendar_path

    def _load(self) -> list[dict[str, Any]]:
        return load_json(self.calendar_path)

    def _save(self, events: list[dict[str, Any]]) -> None:
        save_json(self.calendar_path, events)

    @staticmethod
    def _next_id(events: list[dict[str, Any]]) -> str:
        nums = [
            int(str(e["id"]).lstrip("e"))
            for e in events
            if str(e.get("id", "")).lstrip("e").isdigit()
        ]
        return f"e{(max(nums) + 1) if nums else 1}"

    @tool(
        "List calendar events, optionally filtered by an ISO date range "
        "(date_from/date_to). With no range, returns all events.",
        untrusted_fields=("title", "description"),
    )
    def list_events(
        self, date_from: str | None = None, date_to: str | None = None
    ) -> dict[str, Any]:
        events = self._load()
        if date_from is not None:
            events = [e for e in events if e.get("start", "") >= date_from]
        if date_to is not None:
            events = [e for e in events if e.get("start", "") <= date_to]
        return {"count": len(events), "events": events}

    @tool(
        "Get a single calendar event by id.",
        untrusted_fields=("title", "description"),
    )
    def get_event(self, event_id: str) -> dict[str, Any]:
        for e in self._load():
            if str(e.get("id")) == str(event_id):
                return {"found": True, "event": e}
        return {"found": False, "event": None}

    @tool("Create a new calendar event.")
    def create_event(
        self,
        title: str,
        start: str,
        end: str,
        description: str = "",
        attendees: list[str] | None = None,
    ) -> dict[str, Any]:
        events = self._load()
        event = {
            "id": self._next_id(events),
            "title": title,
            "description": description,
            "start": start,
            "end": end,
            "attendees": attendees or [],
        }
        events.append(event)
        self._save(events)
        return {"created": True, "event": event}

    @tool("Update fields of an existing calendar event by id.")
    def update_event(
        self,
        event_id: str,
        title: str | None = None,
        description: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        events = self._load()
        updates = {
            k: v
            for k, v in {
                "title": title,
                "description": description,
                "start": start,
                "end": end,
            }.items()
            if v is not None
        }
        for e in events:
            if str(e.get("id")) == str(event_id):
                e.update(updates)
                self._save(events)
                return {"updated": True, "event": e}
        return {"updated": False, "event": None}

    @tool("Delete a calendar event by id.")
    def delete_event(self, event_id: str) -> dict[str, Any]:
        events = self._load()
        remaining = [e for e in events if str(e.get("id")) != str(event_id)]
        deleted = len(remaining) != len(events)
        if deleted:
            self._save(remaining)
        return {"deleted": deleted, "id": event_id}
