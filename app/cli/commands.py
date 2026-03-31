from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from app.workflow import WorkflowOrchestrator


@dataclass(slots=True)
class CommandResult:
    handled: bool
    should_exit: bool = False
    system_message: str | None = None


def _format_snapshot(runtime: WorkflowOrchestrator) -> str:
    snapshot = runtime.state.snapshot()
    now = datetime.now()
    next_wake = (
        f"{int((snapshot.next_wake_up_at - now).total_seconds())}s"
        if snapshot.next_wake_up_at and snapshot.next_wake_up_at > now
        else "n/a"
    )
    return (
        f"session={snapshot.session_id}\n"
        f"phase={snapshot.phase.value}\n"
        f"proactive_enabled={snapshot.proactive_enabled}\n"
        f"debug_enabled={snapshot.debug_enabled}\n"
        f"turns={len(snapshot.turns)}\n"
        f"next_wake={next_wake}\n"
        f"consecutive_proactive_turns={snapshot.consecutive_proactive_turns}\n"
        f"last_decision={(snapshot.last_decision.decision if snapshot.last_decision else 'n/a')}"
    )


def handle_command(raw: str, runtime: WorkflowOrchestrator) -> CommandResult:
    parts = raw.strip().split()
    command = parts[0].lower()
    args = parts[1:]

    if command == "/help":
        return CommandResult(
            handled=True,
            system_message=(
                "Available commands:\n"
                "/help, /quit, /clear, /status, /models, /proactive on|off, /debug on|off,\n"
                "/history, /config, /reset, /poke, /speak-now"
            ),
        )
    if command == "/quit":
        return CommandResult(handled=True, should_exit=True, system_message="Shutting down session.")
    if command == "/clear":
        runtime.clear_history()
        return CommandResult(handled=True, system_message="Conversation history cleared.")
    if command == "/status":
        return CommandResult(handled=True, system_message=_format_snapshot(runtime))
    if command == "/models":
        config = runtime.config
        return CommandResult(
            handled=True,
            system_message=(
                f"dialogue={config.models.dialogue}\n"
                f"decision={config.models.decision}\n"
                f"base_url={config.models.base_url}"
            ),
        )
    if command == "/history":
        snapshot = runtime.state.snapshot()
        if not snapshot.turns:
            return CommandResult(handled=True, system_message="No turns yet.")
        history = "\n".join(
            f"{turn.turn_id}. {turn.role}{' [proactive]' if turn.proactive else ''}: {turn.content}"
            for turn in snapshot.turns
        )
        return CommandResult(handled=True, system_message=history)
    if command == "/config":
        return CommandResult(
            handled=True,
            system_message=json.dumps(runtime.config.model_dump(), ensure_ascii=False, indent=2),
        )
    if command == "/reset":
        runtime.reset_runtime()
        return CommandResult(handled=True, system_message="Runtime state reset.")
    if command == "/poke":
        runtime.trigger_poke()
        return CommandResult(handled=True, system_message="Manual proactive decision cycle triggered.")
    if command == "/speak-now":
        runtime.trigger_speak_now()
        return CommandResult(handled=True, system_message="Manual proactive generation triggered.")
    if command == "/proactive":
        if not args or args[0] not in {"on", "off"}:
            return CommandResult(handled=True, system_message="Usage: /proactive on|off")
        enabled = args[0] == "on"
        runtime.set_proactive_enabled(enabled)
        return CommandResult(handled=True, system_message=f"Proactive mode {'enabled' if enabled else 'disabled'}.")
    if command == "/debug":
        if not args or args[0] not in {"on", "off"}:
            return CommandResult(handled=True, system_message="Usage: /debug on|off")
        enabled = args[0] == "on"
        runtime.set_debug_enabled(enabled)
        return CommandResult(handled=True, system_message=f"Debug mode {'enabled' if enabled else 'disabled'}.")
    return CommandResult(handled=False)

