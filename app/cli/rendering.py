from __future__ import annotations

from datetime import datetime

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.config import AppConfig
from app.state import SessionSnapshot


def _relative_time(when: datetime | None) -> str:
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


def _countdown(when: datetime | None) -> str:
    if when is None:
        return "n/a"
    seconds = int((when - datetime.now()).total_seconds())
    return f"{max(seconds, 0)}s"


def render_dashboard(snapshot: SessionSnapshot, config: AppConfig, input_buffer: str) -> RenderableType:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="input", size=5),
    )
    layout["body"].split_row(Layout(name="conversation", ratio=2), Layout(name="sidebar", ratio=1))
    layout["sidebar"].split_column(
        Layout(name="state", ratio=2),
        Layout(name="decision", ratio=2),
        Layout(name="events", ratio=3),
    )

    uptime = _relative_time(snapshot.started_at)
    header_text = Text()
    header_text.append("Local Proactive CLI Agent", style="bold cyan")
    header_text.append(f"  session={snapshot.session_id}", style="white")
    header_text.append(f"  uptime={uptime}", style="white")
    header_text.append(f"  phase={snapshot.phase.value}", style="green")
    header_text.append(f"  proactive={'on' if snapshot.proactive_enabled else 'off'}", style="yellow")
    header_text.append(f"  dialogue={snapshot.dialogue_model}", style="bright_white")
    header_text.append(f"  decision={snapshot.decision_model}", style="bright_white")
    layout["header"].update(Panel(header_text, border_style="cyan"))

    conversation_parts: list[RenderableType] = []
    visible_turns = snapshot.turns[-config.cli.max_visible_turns :]
    if not visible_turns and snapshot.draft is None:
        conversation_parts.append(Text("No messages yet. Type below or use /help.", style="dim"))
    for turn in visible_turns:
        style = "cyan" if turn.role == "user" else "green"
        if turn.role == "system":
            style = "yellow"
        if turn.proactive:
            style = "magenta"
        title = f"{turn.turn_id}. {turn.role}{' [proactive]' if turn.proactive else ''}  {_relative_time(turn.created_at)}"
        conversation_parts.append(Panel(turn.content, title=title, border_style=style))
    if snapshot.draft:
        title = f"assistant {'[proactive]' if snapshot.draft.proactive else '[reply]'} typing..."
        border = "magenta" if snapshot.draft.proactive else "green"
        conversation_parts.append(Panel(snapshot.draft.content or "...", title=title, border_style=border))
    layout["conversation"].update(Panel(Group(*conversation_parts), title="Conversation", border_style="white"))

    state_table = Table.grid(padding=(0, 1))
    state_table.add_column(style="bold")
    state_table.add_column()
    state_table.add_row("Workflow", snapshot.phase.value)
    state_table.add_row("Last user", _relative_time(snapshot.last_user_activity))
    state_table.add_row("Last assistant", _relative_time(snapshot.last_assistant_activity))
    state_table.add_row("Last proactive", _relative_time(snapshot.last_proactive_activity))
    state_table.add_row("Next wake", _countdown(snapshot.next_wake_up_at))
    state_table.add_row("Sleep reason", snapshot.last_sleep_reason or "n/a")
    state_table.add_row("Awaiting answer", "yes" if snapshot.awaiting_user_answer else "no")
    state_table.add_row("Consecutive proactive", str(snapshot.consecutive_proactive_turns))
    if snapshot.last_error:
        state_table.add_row("Last error", snapshot.last_error)
    layout["state"].update(Panel(state_table, title="State", border_style="blue"))

    decision_table = Table.grid(padding=(0, 1))
    decision_table.add_column(style="bold")
    decision_table.add_column()
    if snapshot.last_decision:
        decision_table.add_row("Decision", snapshot.last_decision.decision)
        decision_table.add_row("Source", snapshot.last_decision.source)
        decision_table.add_row("Confidence", f"{snapshot.last_decision.confidence:.2f}")
        decision_table.add_row("Reason", snapshot.last_decision.reason)
        decision_table.add_row("Blocked rule", snapshot.last_decision.blocked_by_rule or "n/a")
        decision_table.add_row("Time", _relative_time(snapshot.last_decision.timestamp))
    else:
        decision_table.add_row("Decision", "n/a")
        decision_table.add_row("Reason", "No proactive decision yet.")
    layout["decision"].update(Panel(decision_table, title="Decision Diagnostics", border_style="green"))

    event_lines: list[RenderableType] = []
    for entry in snapshot.event_log[-config.cli.max_visible_events :]:
        style = {
            "info": "white",
            "warning": "yellow",
            "error": "red",
            "debug": "dim",
        }.get(entry.level, "white")
        event_lines.append(Text(f"[{entry.timestamp.strftime('%H:%M:%S')}] {entry.message}", style=style))
    if not event_lines:
        event_lines.append(Text("No events yet.", style="dim"))
    layout["events"].update(Panel(Group(*event_lines), title="Event Log", border_style="yellow"))

    hint = "Enter message. Slash commands: /help /status /poke /speak-now /quit"
    input_render = Columns(
        [
            Panel(Text(input_buffer or " ", style="bold white"), title="Input", border_style="cyan"),
            Panel(Text(hint, style="dim"), title="Hints", border_style="white"),
        ]
    )
    layout["input"].update(input_render)
    return layout

