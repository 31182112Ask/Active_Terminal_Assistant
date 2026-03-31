from __future__ import annotations

from datetime import datetime

from app.state import SessionSnapshot


def _fmt_delta(when: datetime | None) -> str:
    if when is None:
        return "never"
    seconds = int((datetime.now() - when).total_seconds())
    return f"{seconds}s ago"


def build_compact_decision_context(snapshot: SessionSnapshot) -> str:
    recent_turns = snapshot.turns[-6:]
    transcript = "\n".join(
        f"- {turn.role}{' [proactive]' if turn.proactive else ''}: {turn.content}"
        for turn in recent_turns
    )
    if not transcript:
        transcript = "- no conversation yet"
    return (
        f"phase: {snapshot.phase.value}\n"
        f"last_user_activity: {_fmt_delta(snapshot.last_user_activity)}\n"
        f"last_assistant_activity: {_fmt_delta(snapshot.last_assistant_activity)}\n"
        f"last_proactive_activity: {_fmt_delta(snapshot.last_proactive_activity)}\n"
        f"awaiting_user_answer: {snapshot.awaiting_user_answer}\n"
        f"consecutive_proactive_turns: {snapshot.consecutive_proactive_turns}\n"
        f"last_rule_block: {snapshot.last_rule_block or 'none'}\n"
        f"recent_transcript:\n{transcript}"
    )

