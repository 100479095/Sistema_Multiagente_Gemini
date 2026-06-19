"""Email agent — simulated mailbox over a local JSON file (no real IMAP/SMTP).

Email ``subject`` and ``body`` are **untrusted** content (PDR §8.0d): they may
carry an injected payload and are re-injected verbatim into the orchestrator
context. ``draft_email`` never sends; it only returns the draft.
"""

from __future__ import annotations

from typing import Any

from agents.base import Agent, load_json, save_json, tool


class EmailAgent(Agent):
    name = "email"

    def __init__(self, mailbox_path: str) -> None:
        self.mailbox_path = mailbox_path

    def _load(self) -> list[dict[str, Any]]:
        return load_json(self.mailbox_path)

    @staticmethod
    def _header(email: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": email.get("id"),
            "from": email.get("from"),
            "to": email.get("to"),
            "subject": email.get("subject"),
            "date": email.get("date"),
            "read": email.get("read", False),
        }

    @tool(
        "List email headers in a folder (most recent first).",
        untrusted_fields=("subject",),
    )
    def list_emails(self, folder: str = "inbox", limit: int = 10) -> dict[str, Any]:
        emails = [
            e for e in self._load() if e.get("folder", "inbox") == folder
        ]
        headers = [self._header(e) for e in emails][:limit]
        return {"folder": folder, "count": len(headers), "emails": headers}

    @tool(
        "Read a single email by id, including its full body.",
        untrusted_fields=("subject", "body"),
    )
    def read_email(self, email_id: str) -> dict[str, Any]:
        for e in self._load():
            if str(e.get("id")) == str(email_id):
                return {"found": True, "email": e}
        return {"found": False, "email": None}

    @tool(
        "Search emails by free text over subject and body.",
        untrusted_fields=("subject",),
    )
    def search_emails(self, query: str, limit: int = 10) -> dict[str, Any]:
        q = query.lower()
        matches = [
            self._header(e)
            for e in self._load()
            if q in (e.get("subject", "") + " " + e.get("body", "")).lower()
        ]
        return {"query": query, "count": len(matches[:limit]), "emails": matches[:limit]}

    @tool("Create an email draft (does NOT send; returns the draft only).")
    def draft_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        return {
            "drafted": True,
            "sent": False,
            "draft": {"to": to, "subject": subject, "body": body},
        }
