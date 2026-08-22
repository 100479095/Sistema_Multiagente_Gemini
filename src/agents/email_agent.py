"""Email agent — simulated mailbox over a local JSON file (no real IMAP/SMTP).

Email ``subject`` and ``body`` are **untrusted** content (PDR §8.0d): they may
carry an injected payload and are re-injected verbatim into the orchestrator
context. ``draft_email`` never sends; it only returns the draft.
"""

from __future__ import annotations

from typing import Any

from agents.base import Agent, load_json, save_json, tool

#: Fallback when the model sends ``limit: null`` instead of omitting it.
DEFAULT_LIMIT = 10


def _normalize_folder(folder: str | None) -> str | None:
    """Canonicalise a folder name (``"INBOX"`` -> ``"inbox"``); ``None`` = all.

    The model often phrases the folder with different casing; comparing it
    verbatim against the stored value made ``list_emails`` report an empty
    mailbox, so the seeded payload was never seen (mirrors
    :func:`agents.home_agent._normalize_room`).
    """
    if folder is None:
        return None
    return folder.strip().lower() or None


def _clamp_limit(limit: int | None) -> int:
    return DEFAULT_LIMIT if limit is None else limit


def _date_key(email: dict[str, Any]) -> str:
    """Sort key for "most recent first"; undated emails sort last.

    Dates are stored ISO-8601, so lexicographic order is chronological order. An
    empty string for a missing date puts that email at the end under ``reverse``.
    """
    return str(email.get("date") or "")


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
    def list_emails(
        self, folder: str | None = "inbox", limit: int | None = DEFAULT_LIMIT
    ) -> dict[str, Any]:
        wanted = _normalize_folder(folder)
        emails = [
            e
            for e in self._load()
            if wanted is None or _normalize_folder(str(e.get("folder", "inbox"))) == wanted
        ]
        # Sort before slicing: the description promises "most recent first", and
        # limiting file order would drop the newest emails instead of the oldest.
        emails.sort(key=_date_key, reverse=True)
        headers = [self._header(e) for e in emails][: _clamp_limit(limit)]
        return {"folder": wanted or "all", "count": len(headers), "emails": headers}

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
    def search_emails(
        self, query: str, limit: int | None = DEFAULT_LIMIT
    ) -> dict[str, Any]:
        q = query.lower()
        matches = [
            self._header(e)
            for e in self._load()
            if q in (e.get("subject", "") + " " + e.get("body", "")).lower()
        ][: _clamp_limit(limit)]
        return {"query": query, "count": len(matches), "emails": matches}

    @tool("Create an email draft (does NOT send; returns the draft only).")
    def draft_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        return {
            "drafted": True,
            "sent": False,
            "draft": {"to": to, "subject": subject, "body": body},
        }
