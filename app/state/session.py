from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from datetime import datetime

from app.state.models import (
    DecisionRecord,
    DraftMessage,
    EventLogEntry,
    MessageTurn,
    SessionSnapshot,
    WorkflowPhase,
)


class SessionStateManager:
    def __init__(self, dialogue_model: str, decision_model: str, debug_enabled: bool, proactive_enabled: bool):
        self._lock = threading.RLock()
        self._session_id = uuid.uuid4().hex[:8]
        self._started_at = datetime.now()
        self._phase = WorkflowPhase.STARTING
        self._turns: list[MessageTurn] = []
        self._event_log: list[EventLogEntry] = []
        self._draft: DraftMessage | None = None
        self._last_user_activity: datetime | None = None
        self._last_assistant_activity: datetime | None = None
        self._last_proactive_activity: datetime | None = None
        self._last_decision: DecisionRecord | None = None
        self._consecutive_proactive_turns = 0
        self._awaiting_user_answer = False
        self._pending_proactive_cancelled = False
        self._next_wake_up_at: datetime | None = None
        self._last_sleep_reason: str | None = None
        self._last_rule_block: str | None = None
        self._last_error: str | None = None
        self._dialogue_model = dialogue_model
        self._decision_model = decision_model
        self._debug_enabled = debug_enabled
        self._proactive_enabled = proactive_enabled
        self._turn_counter = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def set_phase(self, phase: WorkflowPhase) -> None:
        with self._lock:
            self._phase = phase

    def set_debug(self, enabled: bool) -> None:
        with self._lock:
            self._debug_enabled = enabled

    def set_proactive_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._proactive_enabled = enabled

    def log_event(self, message: str, level: str = "info") -> None:
        with self._lock:
            self._event_log.append(EventLogEntry(timestamp=datetime.now(), level=level, message=message))
            self._event_log = self._event_log[-50:]

    def append_turn(self, role: str, content: str, proactive: bool = False) -> MessageTurn:
        with self._lock:
            self._turn_counter += 1
            turn = MessageTurn(
                turn_id=self._turn_counter,
                role=role,
                content=content,
                created_at=datetime.now(),
                proactive=proactive,
            )
            self._turns.append(turn)
            if role == "user":
                self._last_user_activity = turn.created_at
                self._consecutive_proactive_turns = 0
                self._pending_proactive_cancelled = False
            elif role == "assistant":
                self._last_assistant_activity = turn.created_at
                if proactive:
                    self._last_proactive_activity = turn.created_at
                    self._consecutive_proactive_turns += 1
                else:
                    self._consecutive_proactive_turns = 0
            return turn

    def clear_turns(self) -> None:
        with self._lock:
            self._turns.clear()
            self._draft = None
            self._consecutive_proactive_turns = 0
            self._awaiting_user_answer = False
            self._last_rule_block = None
            self._last_decision = None
            self._pending_proactive_cancelled = False
            self._next_wake_up_at = None
            self._last_sleep_reason = None

    def start_draft(self, proactive: bool = False) -> None:
        with self._lock:
            self._draft = DraftMessage(role="assistant", started_at=datetime.now(), proactive=proactive, content="")
            self._pending_proactive_cancelled = False

    def update_draft(self, chunk: str) -> None:
        with self._lock:
            if self._draft is None:
                return
            self._draft.content += chunk

    def cancel_draft(self, user_cancelled: bool = False) -> None:
        with self._lock:
            self._draft = None
            self._pending_proactive_cancelled = user_cancelled

    def finalize_draft(self) -> MessageTurn | None:
        with self._lock:
            if self._draft is None:
                return None
            content = self._draft.content.strip()
            proactive = self._draft.proactive
            self._draft = None
        if not content:
            return None
        turn = self.append_turn("assistant", content, proactive=proactive)
        self.set_awaiting_user_answer(content.rstrip().endswith("?"))
        return turn

    def set_last_decision(self, decision: DecisionRecord) -> None:
        with self._lock:
            self._last_decision = decision
            self._last_rule_block = decision.blocked_by_rule

    def set_sleep_plan(self, seconds: int, reason: str) -> None:
        with self._lock:
            self._next_wake_up_at = datetime.fromtimestamp(datetime.now().timestamp() + seconds)
            self._last_sleep_reason = reason

    def clear_sleep_plan(self) -> None:
        with self._lock:
            self._next_wake_up_at = None

    def set_rule_block(self, reason: str | None) -> None:
        with self._lock:
            self._last_rule_block = reason

    def set_error(self, error: str | None) -> None:
        with self._lock:
            self._last_error = error

    def set_awaiting_user_answer(self, waiting: bool) -> None:
        with self._lock:
            self._awaiting_user_answer = waiting

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(
                session_id=self._session_id,
                started_at=self._started_at,
                phase=self._phase,
                turns=[replace(turn) for turn in self._turns],
                draft=replace(self._draft) if self._draft else None,
                event_log=[replace(entry) for entry in self._event_log],
                proactive_enabled=self._proactive_enabled,
                debug_enabled=self._debug_enabled,
                last_user_activity=self._last_user_activity,
                last_assistant_activity=self._last_assistant_activity,
                last_proactive_activity=self._last_proactive_activity,
                last_decision=replace(self._last_decision) if self._last_decision else None,
                consecutive_proactive_turns=self._consecutive_proactive_turns,
                awaiting_user_answer=self._awaiting_user_answer,
                pending_proactive_cancelled=self._pending_proactive_cancelled,
                next_wake_up_at=self._next_wake_up_at,
                last_sleep_reason=self._last_sleep_reason,
                last_rule_block=self._last_rule_block,
                last_error=self._last_error,
                dialogue_model=self._dialogue_model,
                decision_model=self._decision_model,
            )
