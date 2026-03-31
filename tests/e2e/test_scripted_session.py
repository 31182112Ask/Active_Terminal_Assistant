from app.adapters.decision import DecisionModelAdapter
from app.cli.commands import handle_command


def test_scripted_command_flow() -> None:
    class Runtime:
        def __init__(self):
            self.config = type(
                "Config",
                (),
                {
                    "models": type(
                        "Models", (), {"dialogue": "qwen3:14b", "decision": "qwen3:1.7b", "base_url": "http://127.0.0.1:11434"}
                    )(),
                    "model_dump": lambda self: {"mock": True},
                },
            )()
            self.flags = []
            self.state = type(
                "State",
                (),
                {
                    "snapshot": lambda self: type(
                        "Snapshot",
                        (),
                        {
                            "session_id": "scripted",
                            "phase": type("Phase", (), {"value": "idle_waiting"})(),
                            "proactive_enabled": True,
                            "debug_enabled": True,
                            "turns": [],
                            "next_wake_up_at": None,
                            "consecutive_proactive_turns": 0,
                            "last_decision": None,
                        },
                    )()
                },
            )()

        def clear_history(self):
            self.flags.append("clear")

        def reset_runtime(self):
            self.flags.append("reset")

        def trigger_poke(self):
            self.flags.append("poke")

        def trigger_speak_now(self):
            self.flags.append("speak")

        def set_proactive_enabled(self, enabled):
            self.flags.append(("proactive", enabled))

        def set_debug_enabled(self, enabled):
            self.flags.append(("debug", enabled))

    runtime = Runtime()
    assert handle_command("/status", runtime).handled is True
    assert handle_command("/poke", runtime).handled is True
    assert "poke" in runtime.flags
    assert DecisionModelAdapter.parse('{"decision":"WAIT","reason":"done","confidence":0.5}').decision == "WAIT"
