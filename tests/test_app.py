"""Tests for the system wiring layer (src/app.py): reset to the benign seeds.

This exercises the shipped ``data/seeds/`` artifacts but redirects the *working*
stores into ``tmp_path`` so the repo's data is never mutated. Resetting is what
the experiment bench runs before every repetition (PDR §7.9c).
"""

from __future__ import annotations

from pathlib import Path

from agents.base import load_json, save_json
from app import reset_data, working_paths
from config import LoggingSettings, OrchestratorSettings, PathSettings, Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path) -> Settings:
    """Settings whose working stores live under tmp_path; seeds are the real ones."""
    work = tmp_path / "work"
    return Settings(
        paths=PathSettings(
            mailbox=str(work / "mailbox.json"),
            calendar=str(work / "calendar.json"),
            home_state=str(work / "home_state.json"),
            seeds_dir=str(REPO_ROOT / "data" / "seeds"),
            logs_dir=str(tmp_path / "logs"),
        ),
        orchestrator=OrchestratorSettings(max_iterations=5),
        logging=LoggingSettings(jsonl=True, console=False),
    )


def test_reset_creates_and_restores_working_stores(tmp_path):
    settings = make_settings(tmp_path)
    restored = reset_data(settings)
    assert set(restored) == {"mailbox", "calendar", "home_state"}

    wp = working_paths(settings)
    assert wp["home_state"].exists()
    assert load_json(wp["home_state"])["boiler"] == "off"

    # Mutate, then reset must restore the seed values.
    home = load_json(wp["home_state"])
    home["boiler"] = "on"
    home["windows"]["living_room"] = "open"
    save_json(wp["home_state"], home)
    assert load_json(wp["home_state"])["boiler"] == "on"

    reset_data(settings)
    restored_home = load_json(wp["home_state"])
    assert restored_home["boiler"] == "off"
    assert restored_home["windows"]["living_room"] == "closed"
