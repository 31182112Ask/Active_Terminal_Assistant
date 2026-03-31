from datetime import datetime, timedelta

from app.gui.viewmodel import build_decision, build_header, build_status
from app.state.models import DecisionRecord, SessionSnapshot, WorkflowPhase


def _snapshot() -> SessionSnapshot:
    now = datetime.now()
    return SessionSnapshot(
        session_id="gui123",
        started_at=now - timedelta(minutes=5),
        phase=WorkflowPhase.IDLE_WAITING,
        turns=[],
        draft=None,
        event_log=[],
        proactive_enabled=True,
        debug_enabled=False,
        last_user_activity=now - timedelta(seconds=20),
        last_assistant_activity=now - timedelta(seconds=8),
        last_proactive_activity=None,
        last_decision=DecisionRecord(
            decision="SPEAK",
            intent="continue",
            window="short",
            reason="one more small point would help",
            confidence=0.77,
        ),
        consecutive_proactive_turns=0,
        awaiting_user_answer=False,
        pending_proactive_cancelled=False,
        recently_interrupted=False,
        next_wake_up_at=now + timedelta(seconds=9),
        last_sleep_reason="light continuation window after recent reply",
        current_proactive_window="short",
        last_rule_block=None,
        last_error=None,
        dialogue_model="qwen3:14b",
        decision_model="qwen3:1.7b",
    )


def test_gui_header_and_status_viewmodel() -> None:
    snapshot = _snapshot()
    header = build_header(snapshot)
    status = build_status(snapshot)
    decision = build_decision(snapshot)
    assert "qwen3:14b" in header.subtitle
    assert status.window == "short"
    assert decision.intent == "continue"
    assert decision.window == "short"
