from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.config import ProactiveConfig
from app.state import SessionSnapshot
from app.utils import similarity_ratio


@dataclass(slots=True)
class RuleGateResult:
    allowed: bool
    reason: str


class RuleGateService:
    def __init__(self, config: ProactiveConfig):
        self._config = config

    def evaluate_pre_model(self, snapshot: SessionSnapshot) -> RuleGateResult:
        now = datetime.now()
        if not snapshot.proactive_enabled:
            return RuleGateResult(False, "proactive mode disabled")
        if not snapshot.turns:
            return RuleGateResult(False, "no conversation history yet")
        if snapshot.pending_proactive_cancelled:
            return RuleGateResult(False, "a proactive attempt was just cancelled by user input")
        if snapshot.consecutive_proactive_turns >= self._config.max_consecutive_proactive_turns:
            return RuleGateResult(False, "max consecutive proactive turns reached")
        last_user = next((turn for turn in reversed(snapshot.turns) if turn.role == "user"), None)
        if last_user is None:
            return RuleGateResult(False, "no user turn to follow up on")
        lowered = last_user.content.lower()
        if any(phrase.lower() in lowered for phrase in self._config.user_disengagement_phrases):
            return RuleGateResult(False, "user appears disengaged")
        if snapshot.last_assistant_activity is not None:
            elapsed = (now - snapshot.last_assistant_activity).total_seconds()
            if elapsed < self._config.min_assistant_cooldown_seconds:
                return RuleGateResult(False, "assistant cooldown active")
        if snapshot.last_proactive_activity is not None:
            elapsed = (now - snapshot.last_proactive_activity).total_seconds()
            if elapsed < self._config.min_proactive_cooldown_seconds:
                return RuleGateResult(False, "proactive cooldown active")
        if snapshot.awaiting_user_answer and snapshot.last_assistant_activity is not None:
            elapsed = (now - snapshot.last_assistant_activity).total_seconds()
            if elapsed < self._config.question_cooldown_seconds:
                return RuleGateResult(False, "assistant already asked a question")
        return RuleGateResult(True, "rules allow evaluation")

    def evaluate_post_model(self, snapshot: SessionSnapshot, candidate_text: str) -> RuleGateResult:
        if not candidate_text.strip():
            return RuleGateResult(False, "empty proactive content")
        last_assistant = next((turn for turn in reversed(snapshot.turns) if turn.role == "assistant"), None)
        if last_assistant and similarity_ratio(last_assistant.content, candidate_text) >= self._config.duplicate_similarity_threshold:
            return RuleGateResult(False, "duplicate proactive content")
        proactive_turns = [turn for turn in snapshot.turns if turn.proactive]
        if proactive_turns and similarity_ratio(proactive_turns[-1].content, candidate_text) >= self._config.duplicate_similarity_threshold:
            return RuleGateResult(False, "matches previous proactive message")
        return RuleGateResult(True, "content accepted")

