from datetime import datetime, timedelta

from app.config import ProactiveConfig
from app.services.rules import RuleGateService
from app.state.models import DecisionRecord, EventLogEntry, MessageTurn, SessionSnapshot, WorkflowPhase


def _snapshot(**overrides) -> SessionSnapshot:
    now = datetime.now()
    base = SessionSnapshot(
        session_id="test1234",
        started_at=now - timedelta(minutes=5),
        phase=WorkflowPhase.IDLE_WAITING,
        turns=[
            MessageTurn(1, "user", "Help me plan a trip", now - timedelta(minutes=2)),
            MessageTurn(2, "assistant", "Sure, do you already know your budget?", now - timedelta(minutes=1)),
        ],
        draft=None,
        event_log=[EventLogEntry(timestamp=now, level="info", message="test")],
        proactive_enabled=True,
        debug_enabled=True,
        last_user_activity=now - timedelta(minutes=2),
        last_assistant_activity=now - timedelta(minutes=1),
        last_proactive_activity=None,
        last_decision=None,
        consecutive_proactive_turns=0,
        awaiting_user_answer=False,
        pending_proactive_cancelled=False,
        recently_interrupted=False,
        next_wake_up_at=None,
        last_sleep_reason=None,
        current_proactive_window=None,
        last_rule_block=None,
        last_error=None,
        dialogue_model="dialogue",
        decision_model="decision",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_rule_gate_blocks_when_assistant_is_in_cooldown() -> None:
    config = ProactiveConfig(min_assistant_cooldown_seconds=120)
    gate = RuleGateService(config)
    result = gate.evaluate_pre_model(_snapshot())
    assert result.allowed is False
    assert result.reason == "assistant cooldown active"


def test_rule_gate_blocks_after_user_disengagement() -> None:
    config = ProactiveConfig()
    gate = RuleGateService(config)
    snapshot = _snapshot(
        turns=[
            MessageTurn(1, "user", "Thanks that's all, bye", datetime.now() - timedelta(minutes=2)),
            MessageTurn(2, "assistant", "Understood.", datetime.now() - timedelta(minutes=1)),
        ],
        last_user_activity=datetime.now() - timedelta(minutes=2),
        last_assistant_activity=datetime.now() - timedelta(minutes=1),
    )
    result = gate.evaluate_pre_model(snapshot)
    assert result.allowed is False
    assert result.reason == "user appears disengaged"


def test_post_generation_duplicate_suppression() -> None:
    config = ProactiveConfig(duplicate_similarity_threshold=0.8)
    gate = RuleGateService(config)
    snapshot = _snapshot(
        turns=[
            MessageTurn(1, "user", "Help", datetime.now() - timedelta(minutes=2)),
            MessageTurn(2, "assistant", "I can help you make a checklist.", datetime.now() - timedelta(minutes=1)),
        ],
        last_assistant_activity=datetime.now() - timedelta(minutes=1),
    )
    result = gate.evaluate_post_model(snapshot, "I can help you make a checklist.")
    assert result.allowed is False
    assert "duplicate" in result.reason


def test_rule_gate_blocks_when_recently_interrupted() -> None:
    config = ProactiveConfig(min_assistant_cooldown_seconds=0)
    gate = RuleGateService(config)
    snapshot = _snapshot(
        recently_interrupted=True,
        last_assistant_activity=datetime.now() - timedelta(minutes=2),
    )
    result = gate.evaluate_pre_model(snapshot)
    assert result.allowed is False
    assert result.reason == "user recently interrupted the assistant"

