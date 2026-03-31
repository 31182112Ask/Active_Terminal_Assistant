from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelsConfig(BaseModel):
    dialogue: str = "qwen3:14b"
    decision: str = "qwen3:1.7b"
    base_url: str = "http://127.0.0.1:11434"
    request_timeout_seconds: float = 120.0
    stream: bool = True
    max_context_turns: int = 12
    dialogue_temperature: float = 0.7
    decision_temperature: float = 0.1


class ProactiveConfig(BaseModel):
    enabled: bool = True
    decision_mode: Literal["rule-assisted", "model-first"] = "rule-assisted"
    min_assistant_cooldown_seconds: int = 30
    min_proactive_cooldown_seconds: int = 45
    question_cooldown_seconds: int = 90
    max_consecutive_proactive_turns: int = 2
    min_sleep_seconds: int = 20
    max_sleep_seconds: int = 120
    duplicate_similarity_threshold: float = 0.92
    wake_up_pattern: str = "Stay conservative. Wake more often in active conversations and less often when the user is idle."
    min_response_interval: int = 20
    max_response_interval: int = 180
    user_disengagement_phrases: list[str] = Field(default_factory=lambda: ["bye", "goodbye", "stop", "leave me alone"])


class CliConfig(BaseModel):
    refresh_hz: int = 4
    max_visible_turns: int = 10
    max_visible_events: int = 8
    debug: bool = True
    show_event_log: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    directory: str = "logs"


class AppConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    cli: CliConfig = Field(default_factory=CliConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


ENV_MAP: dict[str, tuple[str, str, Any]] = {
    "LPA_DIALOGUE_MODEL": ("models", "dialogue", str),
    "LPA_DECISION_MODEL": ("models", "decision", str),
    "LPA_OLLAMA_BASE_URL": ("models", "base_url", str),
    "LPA_REQUEST_TIMEOUT_SECONDS": ("models", "request_timeout_seconds", float),
    "LPA_STREAM": ("models", "stream", bool),
    "LPA_DECISION_MODE": ("proactive", "decision_mode", str),
    "LPA_PROACTIVE_ENABLED": ("proactive", "enabled", bool),
    "LPA_MIN_ASSISTANT_COOLDOWN_SECONDS": ("proactive", "min_assistant_cooldown_seconds", int),
    "LPA_MIN_PROACTIVE_COOLDOWN_SECONDS": ("proactive", "min_proactive_cooldown_seconds", int),
    "LPA_MAX_CONSECUTIVE_PROACTIVE_TURNS": ("proactive", "max_consecutive_proactive_turns", int),
    "LPA_MIN_SLEEP_SECONDS": ("proactive", "min_sleep_seconds", int),
    "LPA_MAX_SLEEP_SECONDS": ("proactive", "max_sleep_seconds", int),
    "LPA_DEBUG": ("cli", "debug", bool),
    "LPA_REFRESH_HZ": ("cli", "refresh_hz", int),
    "LPA_LOG_LEVEL": ("logging", "level", str),
}


def _coerce_value(raw: str, target: Any) -> Any:
    if target is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return target(raw)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    data = config.model_dump()
    for env_name, (section, field, cast) in ENV_MAP.items():
        if env_name not in os.environ:
            continue
        data[section][field] = _coerce_value(os.environ[env_name], cast)
    return AppConfig.model_validate(data)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else Path("agent.toml")
    data: dict[str, Any] = {}
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(data)
    return _apply_env_overrides(config)
