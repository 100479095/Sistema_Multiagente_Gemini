"""Typed configuration loaded from ``config.yaml`` with environment overrides.

Loading precedence (highest first):
1. Environment variables prefixed ``TESTBED_`` (nested keys use ``__``,
   e.g. ``TESTBED_LLM__MODEL=qwen3:8b``).
2. ``config.yaml`` at the repository root (or the path in ``TESTBED_CONFIG_FILE``).
3. The defaults declared on the models below.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Repository root = parent of the ``src`` directory that holds this file.
REPO_ROOT = Path(__file__).resolve().parent.parent


class LLMSettings(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5:7b"
    temperature: float = 0.7
    top_p: float = 1.0
    seed: int | None = None
    timeout_s: int = 120


class OrchestratorSettings(BaseModel):
    max_iterations: int = 5


class PathSettings(BaseModel):
    data_dir: str = "data"
    mailbox: str = "data/mailbox.json"
    calendar: str = "data/calendar.json"
    home_state: str = "data/home_state.json"
    # Canonical golden copies of the three stores. ``reset`` restores the working
    # files above from here; the working files are what the agents mutate.
    seeds_dir: str = "data/seeds"
    facts_dir: str = "data/facts"
    logs_dir: str = "logs"
    results_dir: str = "results"
    # Central prompt file (system prompt, carriers, injections, judge prompt).
    messages: str = "messages.yaml"

    def resolve(self, value: str) -> Path:
        """Resolve a configured path relative to the repository root."""
        p = Path(value)
        return p if p.is_absolute() else REPO_ROOT / p


class LoggingSettings(BaseModel):
    console: bool = True
    jsonl: bool = True


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class _YamlConfigSource(PydanticBaseSettingsSource):
    """A settings source that reads ``config.yaml`` (or ``TESTBED_CONFIG_FILE``).

    Registered *below* the env source (see
    :meth:`Settings.settings_customise_sources`) so that ``TESTBED_*`` env vars
    override individual YAML keys — pydantic deep-merges the source dicts, so an
    override of one nested key (e.g. ``TESTBED_PATHS__FACTS_DIR``) leaves the rest
    of the YAML's ``paths`` intact. Reading YAML *as a source* (rather than
    passing it as init kwargs, which would outrank env) is what makes that
    precedence work.
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        config_file = Path(
            os.environ.get("TESTBED_CONFIG_FILE", REPO_ROOT / "config.yaml")
        )
        self._data: dict[str, Any] = _read_yaml(config_file)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class Settings(BaseSettings):
    """Top-level settings object for the target system."""

    model_config = SettingsConfigDict(
        env_prefix="TESTBED_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority, highest first: init args > env vars > config.yaml > defaults.
        return (init_settings, env_settings, _YamlConfigSource(settings_cls), file_secret_settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once and cached.

    Precedence (highest first): explicit init args, then ``TESTBED_*`` env vars,
    then the YAML file, then the model defaults — so an experiment can override
    the model/seed via the environment without editing config.yaml.
    """
    return Settings()
