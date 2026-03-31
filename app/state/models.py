from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


class WorkflowPhase(str, Enum):
    STARTING = "starting"
    READY = "ready"
    USER_INPUT_RECEIVED = "user_input_received"
    ASSISTANT_REPLY_GENERATION = "assistant_reply_generation"
    REPLY_FINALIZATION = "reply_finalization"
    IDLE_WAITING = "idle_waiting"
    PROACTIVE_WAKE_UP = "proactive_wake_up"
    SPEAK_WAIT_DECISION = "speak_wait_decision"
    PROACTIVE_GENERATION = "proactive_generation"
    CANCELLATION = "cancellation"
    ERROR_RECOVERY = "error_recovery"
    SHUTDOWN = "shutdown"


@dataclass(slots=True)
class MessageTurn:
    turn_id: int
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime
    proactive: bool = False


@dataclass(slots=True)
class DraftMessage:
    role: Literal["assistant", "system"]
    started_at: datetime
    proactive: bool = False
    content: str = ""


@dataclass(slots=True)
class DecisionRecord:
    decision: Literal["WAIT", "SPEAK"]
    reason: str
    confidence: float
    source: Literal["rule", "model", "combined"] = "model"
    intent: str | None = None
    window: Literal["short", "long"] | None = None
    suggested_topic: str | None = None
    urgency: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    blocked_by_rule: str | None = None


@dataclass(slots=True)
class EventLogEntry:
    timestamp: datetime
    level: Literal["info", "warning", "error", "debug"]
    message: str


@dataclass(slots=True)
class SessionSnapshot:
    session_id: str
    started_at: datetime
    phase: WorkflowPhase
    turns: list[MessageTurn]
    draft: DraftMessage | None
    event_log: list[EventLogEntry]
    proactive_enabled: bool
    debug_enabled: bool
    last_user_activity: datetime | None
    last_assistant_activity: datetime | None
    last_proactive_activity: datetime | None
    last_decision: DecisionRecord | None
    consecutive_proactive_turns: int
    awaiting_user_answer: bool
    pending_proactive_cancelled: bool
    recently_interrupted: bool
    next_wake_up_at: datetime | None
    last_sleep_reason: str | None
    current_proactive_window: Literal["short", "long"] | None
    last_rule_block: str | None
    last_error: str | None
    dialogue_model: str
    decision_model: str
