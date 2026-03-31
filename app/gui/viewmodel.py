from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.state import SessionSnapshot


def relative_time(when: datetime | None) -> str:
    if when is None:
        return "n/a"
    seconds = max(0, int((datetime.now() - when).total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s ago"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m ago"


def countdown(when: datetime | None) -> str:
    if when is None:
        return "n/a"
    return f"{max(int((when - datetime.now()).total_seconds()), 0)}s"


@dataclass(slots=True)
class HeaderViewModel:
    title: str
    subtitle: str
    phase: str
    proactive_enabled: bool


@dataclass(slots=True)
class StatusViewModel:
    phase: str
    last_user: str
    last_assistant: str
    next_wake: str
    window: str
    cooldown_reason: str
    interrupted: str


@dataclass(slots=True)
class DecisionViewModel:
    decision: str
    intent: str
    window: str
    confidence: str
    reason: str
    blocked_by_rule: str


def build_header(snapshot: SessionSnapshot) -> HeaderViewModel:
    return HeaderViewModel(
        title="Active Terminal Assistant",
        subtitle=f"session={snapshot.session_id}  dialogue={snapshot.dialogue_model}  decision={snapshot.decision_model}",
        phase=snapshot.phase.value,
        proactive_enabled=snapshot.proactive_enabled,
    )


def build_status(snapshot: SessionSnapshot) -> StatusViewModel:
    return StatusViewModel(
        phase=snapshot.phase.value,
        last_user=relative_time(snapshot.last_user_activity),
        last_assistant=relative_time(snapshot.last_assistant_activity),
        next_wake=countdown(snapshot.next_wake_up_at),
        window=snapshot.current_proactive_window or "n/a",
        cooldown_reason=snapshot.last_sleep_reason or "n/a",
        interrupted="yes" if snapshot.recently_interrupted else "no",
    )


def build_decision(snapshot: SessionSnapshot) -> DecisionViewModel:
    if snapshot.last_decision is None:
        return DecisionViewModel(
            decision="n/a",
            intent="n/a",
            window="n/a",
            confidence="0.00",
            reason="No proactive decision yet.",
            blocked_by_rule="n/a",
        )
    return DecisionViewModel(
        decision=snapshot.last_decision.decision,
        intent=snapshot.last_decision.intent or "n/a",
        window=snapshot.last_decision.window or "n/a",
        confidence=f"{snapshot.last_decision.confidence:.2f}",
        reason=snapshot.last_decision.reason,
        blocked_by_rule=snapshot.last_decision.blocked_by_rule or "n/a",
    )

