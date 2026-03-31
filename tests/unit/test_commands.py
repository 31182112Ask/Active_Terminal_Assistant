from dataclasses import dataclass

from app.cli.commands import handle_command


@dataclass
class _DummySnapshot:
    session_id: str = "abc12345"
    phase = type("Phase", (), {"value": "idle_waiting"})()
    proactive_enabled: bool = True
    debug_enabled: bool = True
    turns: list = None
    next_wake_up_at = None
    consecutive_proactive_turns: int = 0
    last_decision = None

    def __post_init__(self):
        if self.turns is None:
            self.turns = []


class _DummyState:
    def snapshot(self):
        return _DummySnapshot()


class _DummyConfig:
    def __init__(self):
        self.models = type("Models", (), {"dialogue": "qwen3:14b", "decision": "qwen3:1.7b", "base_url": "http://127.0.0.1:11434"})()

    def model_dump(self):
        return {"ok": True}


class _DummyRuntime:
    def __init__(self):
        self.state = _DummyState()
        self.config = _DummyConfig()
        self.flags = {}

    def clear_history(self):
        self.flags["clear"] = True

    def reset_runtime(self):
        self.flags["reset"] = True

    def trigger_poke(self):
        self.flags["poke"] = True

    def trigger_speak_now(self):
        self.flags["speak"] = True

    def set_proactive_enabled(self, enabled: bool):
        self.flags["proactive"] = enabled

    def set_debug_enabled(self, enabled: bool):
        self.flags["debug"] = enabled


def test_help_command_returns_help_text() -> None:
    result = handle_command("/help", _DummyRuntime())
    assert result.handled is True
    assert "/quit" in result.system_message


def test_toggle_proactive_command() -> None:
    runtime = _DummyRuntime()
    result = handle_command("/proactive off", runtime)
    assert result.handled is True
    assert runtime.flags["proactive"] is False


def test_unknown_command_is_not_handled() -> None:
    result = handle_command("/does-not-exist", _DummyRuntime())
    assert result.handled is False
