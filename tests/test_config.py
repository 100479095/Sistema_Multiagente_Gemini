"""Config loading precedence (src/config.py).

Documented order (highest first): init args > ``TESTBED_*`` env vars >
``config.yaml`` > model defaults. The env layer must override YAML *per key*,
deep-merging into nested models rather than replacing the whole YAML value.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from config import Settings, get_settings


def _write_config(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """
            llm:
              model: "from-yaml"
            paths:
              facts_dir: "data/facts"
              mailbox: "data/mailbox.json"
            """
        ),
        encoding="utf-8",
    )
    return path


def test_yaml_values_are_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTBED_CONFIG_FILE", str(_write_config(tmp_path / "config.yaml")))
    s = Settings()
    assert s.llm.model == "from-yaml"
    assert s.paths.facts_dir == "data/facts"


def test_env_overrides_yaml_for_nested_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTBED_CONFIG_FILE", str(_write_config(tmp_path / "config.yaml")))
    monkeypatch.setenv("TESTBED_PATHS__FACTS_DIR", "data/other_facts")
    monkeypatch.setenv("TESTBED_LLM__MODEL", "from-env")
    s = Settings()
    # env wins on the overridden keys ...
    assert s.paths.facts_dir == "data/other_facts"
    assert s.llm.model == "from-env"
    # ... while sibling YAML keys survive the deep merge.
    assert s.paths.mailbox == "data/mailbox.json"


def test_get_settings_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTBED_CONFIG_FILE", str(_write_config(tmp_path / "config.yaml")))
    monkeypatch.setenv("TESTBED_PATHS__FACTS_DIR", "data/env_facts")
    get_settings.cache_clear()
    try:
        assert get_settings().paths.facts_dir == "data/env_facts"
    finally:
        get_settings.cache_clear()
