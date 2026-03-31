from __future__ import annotations

import argparse
import asyncio
import json
import tkinter as tk
from contextlib import suppress
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from app.cli.commands import handle_command
from app.config import load_config
from app.gui.viewmodel import build_decision, build_header, build_status, relative_time
from app.state import SessionStateManager
from app.utils import configure_logging
from app.workflow import WorkflowOrchestrator


class AssistantGui:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path)
        self.state = SessionStateManager(
            dialogue_model=self.config.models.dialogue,
            decision_model=self.config.models.decision,
            debug_enabled=self.config.cli.debug,
            proactive_enabled=self.config.proactive.enabled,
        )
        self.logger = configure_logging(self.config.logging.level, self.config.logging.directory, self.state.session_id)
        self.runtime = WorkflowOrchestrator(config=self.config, state=self.state, logger=self.logger)
        self.root = tk.Tk()
        self.root.title(self.config.gui.title)
        self.root.geometry(f"{self.config.gui.width}x{self.config.gui.height}")
        self.root.minsize(980, 720)
        self.root.configure(bg="#f3efe6")

        self._last_render_key: tuple | None = None
        self._developer_open = tk.BooleanVar(value=self.config.gui.developer_panel_open)
        self._proactive_var = tk.BooleanVar(value=self.config.proactive.enabled)
        self._debug_var = tk.BooleanVar(value=self.config.cli.debug)

        self._phase_var = tk.StringVar()
        self._subtitle_var = tk.StringVar()
        self._status_var = tk.StringVar()

        self._build_styles()
        self._build_layout()
        self._bootstrap_runtime()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.config.gui.poll_interval_ms, self._poll_state)

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        with suppress(Exception):
            style.theme_use("clam")
        style.configure("Root.TFrame", background="#f3efe6")
        style.configure("Card.TFrame", background="#fffaf2")
        style.configure("Card.TLabelframe", background="#fffaf2")
        style.configure("Card.TLabelframe.Label", background="#fffaf2", foreground="#4f463a", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", background="#f3efe6", foreground="#1f2a2c", font=("Segoe UI Semibold", 18))
        style.configure("Sub.TLabel", background="#f3efe6", foreground="#6a675f", font=("Segoe UI", 10))
        style.configure("Badge.TLabel", background="#dde7df", foreground="#2f453d", padding=(8, 4))
        style.configure("Primary.TButton", padding=(10, 6))

    def _build_layout(self) -> None:
        root = ttk.Frame(self.root, style="Root.TFrame", padding=14)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root, style="Root.TFrame")
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Active Terminal Assistant", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, textvariable=self._subtitle_var, style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, style="Root.TFrame")
        right = ttk.Frame(body, style="Root.TFrame", width=360)
        body.add(left, weight=4)
        body.add(right, weight=2)

        chat_card = ttk.LabelFrame(left, text="Conversation", style="Card.TLabelframe", padding=10)
        chat_card.pack(fill="both", expand=True)
        self.chat_text = ScrolledText(
            chat_card,
            wrap="word",
            font=("Segoe UI", self.config.gui.chat_font_size),
            background="#fffdf8",
            foreground="#2b2925",
            relief="flat",
            padx=14,
            pady=14,
        )
        self.chat_text.pack(fill="both", expand=True)
        self.chat_text.config(state="disabled")
        self.chat_text.tag_configure("meta", foreground="#7a746a", font=("Segoe UI", 9, "italic"))
        self.chat_text.tag_configure("user", foreground="#14324a", lmargin1=8, lmargin2=8, spacing3=6)
        self.chat_text.tag_configure("assistant", foreground="#213629", lmargin1=8, lmargin2=8, spacing3=6)
        self.chat_text.tag_configure("proactive", foreground="#6b355f", lmargin1=8, lmargin2=8, spacing3=6)
        self.chat_text.tag_configure("system", foreground="#7b5a1a", lmargin1=8, lmargin2=8, spacing3=6)
        self.chat_text.tag_configure("draft", foreground="#7a5b7d", font=("Segoe UI", self.config.gui.chat_font_size, "italic"))

        input_card = ttk.LabelFrame(left, text="Input", style="Card.TLabelframe", padding=10)
        input_card.pack(fill="x", pady=(10, 0))
        self.input_text = tk.Text(input_card, height=4, wrap="word", font=("Segoe UI", 11), relief="flat", padx=10, pady=10)
        self.input_text.pack(fill="x", expand=True)
        self.input_text.bind("<Return>", self._on_return)
        self.input_text.bind("<Shift-Return>", self._on_shift_return)

        input_buttons = ttk.Frame(input_card, style="Card.TFrame")
        input_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(input_buttons, text="Send", style="Primary.TButton", command=self._send_input).pack(side="left")
        ttk.Button(input_buttons, text="Cancel Output", command=self.runtime.cancel_active_output).pack(side="left", padx=(8, 0))
        ttk.Button(input_buttons, text="Clear Chat", command=self.runtime.clear_history).pack(side="left", padx=(8, 0))
        ttk.Button(input_buttons, text="Export Transcript", command=self._export_transcript).pack(side="right")

        status_card = ttk.LabelFrame(right, text="State", style="Card.TLabelframe", padding=10)
        status_card.pack(fill="x")
        self.status_lines = {}
        for label in ("Phase", "Last user", "Last assistant", "Next wake", "Window", "Sleep reason", "Interrupted"):
            row = ttk.Frame(status_card, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=14, bg="#fffaf2", fg="#5b554c", anchor="w", font=("Segoe UI", 9)).pack(side="left")
            value = tk.Label(row, text="-", bg="#fffaf2", fg="#1f2a2c", anchor="w", font=("Segoe UI", 9))
            value.pack(side="left", fill="x", expand=True)
            self.status_lines[label] = value

        controls_card = ttk.LabelFrame(right, text="Controls", style="Card.TLabelframe", padding=10)
        controls_card.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            controls_card,
            text="Proactive mode",
            variable=self._proactive_var,
            command=lambda: self.runtime.set_proactive_enabled(self._proactive_var.get()),
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls_card,
            text="Developer panel",
            variable=self._developer_open,
            command=self._toggle_developer_panel,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls_card,
            text="Debug mode",
            variable=self._debug_var,
            command=lambda: self.runtime.set_debug_enabled(self._debug_var.get()),
        ).pack(anchor="w")

        control_buttons = ttk.Frame(controls_card, style="Card.TFrame")
        control_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(control_buttons, text="Poke", command=self.runtime.trigger_poke).pack(side="left")
        ttk.Button(control_buttons, text="Speak Now", command=self.runtime.trigger_speak_now).pack(side="left", padx=(8, 0))
        ttk.Button(control_buttons, text="Reset", command=self.runtime.reset_runtime).pack(side="left", padx=(8, 0))

        self.developer_card = ttk.LabelFrame(right, text="Developer Panel", style="Card.TLabelframe", padding=10)
        self.developer_card.pack(fill="both", expand=True, pady=(10, 0))
        self.decision_lines = {}
        for label in ("Decision", "Intent", "Window", "Confidence", "Reason", "Blocked rule"):
            row = ttk.Frame(self.developer_card, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=14, bg="#fffaf2", fg="#5b554c", anchor="w", font=("Segoe UI", 9)).pack(side="left")
            if label in {"Reason", "Blocked rule"}:
                value = tk.Message(row, width=250, text="-", background="#fffaf2", foreground="#1f2a2c", font=("Segoe UI", 9))
            else:
                value = tk.Label(row, text="-", bg="#fffaf2", fg="#1f2a2c", anchor="w", font=("Segoe UI", 9))
            value.pack(side="left", fill="x", expand=True)
            self.decision_lines[label] = value

        tk.Label(self.developer_card, text="Recent events", bg="#fffaf2", fg="#5b554c", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 4))
        self.events_text = ScrolledText(
            self.developer_card,
            height=12,
            wrap="word",
            font=("Consolas", 9),
            background="#fffdf8",
            foreground="#2b2925",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.events_text.pack(fill="both", expand=True)
        self.events_text.config(state="disabled")
        self._toggle_developer_panel()

    def _bootstrap_runtime(self) -> None:
        ok, message = asyncio.run(self.runtime.startup_check())
        if not ok:
            messagebox.showerror("Startup Error", message)
            self.root.destroy()
            raise SystemExit(1)
        self.runtime.state.log_event(message)
        self.runtime.post_system_message("输入消息开始聊天。也支持 /help、/status、/poke、/speak-now。")
        self.runtime.start()

    def _toggle_developer_panel(self) -> None:
        if self._developer_open.get():
            self.developer_card.pack(fill="both", expand=True, pady=(10, 0))
        else:
            self.developer_card.pack_forget()

    def _on_return(self, event):
        if event.state & 0x1:
            return
        self._send_input()
        return "break"

    def _on_shift_return(self, event):
        return None

    def _send_input(self) -> None:
        raw = self.input_text.get("1.0", "end").strip()
        if not raw:
            return
        self.input_text.delete("1.0", "end")
        if raw.startswith("/"):
            result = handle_command(raw, self.runtime)
            if result.handled:
                if result.system_message:
                    self.runtime.post_system_message(result.system_message)
                if result.should_exit:
                    self._on_close()
                return
        self.runtime.submit_user_message(raw)

    def _export_transcript(self) -> None:
        snapshot = self.runtime.state.snapshot()
        export_dir = Path("logs")
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"transcript-{snapshot.session_id}.json"
        payload = [
            {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "content": turn.content,
                "proactive": turn.proactive,
                "created_at": turn.created_at.isoformat(),
            }
            for turn in snapshot.turns
        ]
        export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.runtime.post_system_message(f"Transcript exported to {export_path}")

    def _render_chat(self, snapshot) -> None:
        self.chat_text.config(state="normal")
        self.chat_text.delete("1.0", "end")
        for turn in snapshot.turns:
            style = "assistant"
            speaker = "Assistant"
            if turn.role == "user":
                style = "user"
                speaker = "You"
            elif turn.role == "system":
                style = "system"
                speaker = "System"
            elif turn.proactive:
                style = "proactive"
                speaker = "Assistant · proactive"
            self.chat_text.insert("end", f"{speaker}  {relative_time(turn.created_at)}\n", ("meta",))
            self.chat_text.insert("end", f"{turn.content}\n\n", (style,))
        if snapshot.draft:
            speaker = "Assistant · drafting"
            if snapshot.draft.proactive:
                speaker = "Assistant · proactive drafting"
            self.chat_text.insert("end", f"{speaker}\n", ("meta",))
            self.chat_text.insert("end", f"{snapshot.draft.content or '…'}\n", ("draft",))
        self.chat_text.config(state="disabled")
        self.chat_text.see("end")

    def _render_status(self, snapshot) -> None:
        header = build_header(snapshot)
        status = build_status(snapshot)
        decision = build_decision(snapshot)

        self._subtitle_var.set(header.subtitle)
        self.status_lines["Phase"].configure(text=status.phase)
        self.status_lines["Last user"].configure(text=status.last_user)
        self.status_lines["Last assistant"].configure(text=status.last_assistant)
        self.status_lines["Next wake"].configure(text=status.next_wake)
        self.status_lines["Window"].configure(text=status.window)
        self.status_lines["Sleep reason"].configure(text=status.cooldown_reason)
        self.status_lines["Interrupted"].configure(text=status.interrupted)

        for label, value in {
            "Decision": decision.decision,
            "Intent": decision.intent,
            "Window": decision.window,
            "Confidence": decision.confidence,
            "Reason": decision.reason,
            "Blocked rule": decision.blocked_by_rule,
        }.items():
            widget = self.decision_lines[label]
            if isinstance(widget, tk.Message):
                widget.configure(text=value)
            else:
                widget.configure(text=value)

        self.events_text.config(state="normal")
        self.events_text.delete("1.0", "end")
        for entry in snapshot.event_log[-10:]:
            self.events_text.insert("end", f"[{entry.timestamp.strftime('%H:%M:%S')}] {entry.level.upper()} {entry.message}\n")
        self.events_text.config(state="disabled")

        self._proactive_var.set(snapshot.proactive_enabled)
        self._debug_var.set(snapshot.debug_enabled)

    def _poll_state(self) -> None:
        if not self.root.winfo_exists():
            return
        snapshot = self.runtime.state.snapshot()
        render_key = (
            len(snapshot.turns),
            snapshot.turns[-1].turn_id if snapshot.turns else 0,
            snapshot.draft.content if snapshot.draft else "",
            snapshot.phase.value,
            snapshot.last_decision.timestamp.isoformat() if snapshot.last_decision else "",
            len(snapshot.event_log),
            snapshot.proactive_enabled,
            snapshot.debug_enabled,
            snapshot.current_proactive_window,
        )
        if render_key != self._last_render_key:
            self._render_chat(snapshot)
            self._render_status(snapshot)
            self._last_render_key = render_key
        self.root.after(self.config.gui.poll_interval_ms, self._poll_state)

    def _on_close(self) -> None:
        with suppress(Exception):
            self.runtime.stop()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def run_gui(config_path: str | None = None) -> int:
    app = AssistantGui(config_path=config_path)
    return app.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Active Terminal Assistant GUI")
    parser.add_argument("--config", dest="config_path", default=None, help="Optional path to a TOML config file.")
    args = parser.parse_args()
    raise SystemExit(run_gui(args.config_path))
