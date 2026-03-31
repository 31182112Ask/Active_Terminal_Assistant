from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from app.cli.commands import handle_command
from app.cli.input import InputController
from app.cli.rendering import render_dashboard
from app.config import load_config
from app.state import SessionStateManager
from app.utils import configure_logging
from app.workflow import WorkflowOrchestrator


async def run_cli(config_path: str | None = None) -> int:
    config = load_config(config_path)
    bootstrap_state = SessionStateManager(
        dialogue_model=config.models.dialogue,
        decision_model=config.models.decision,
        debug_enabled=config.cli.debug,
        proactive_enabled=config.proactive.enabled,
    )
    logger = configure_logging(config.logging.level, config.logging.directory, bootstrap_state.session_id)
    runtime = WorkflowOrchestrator(config=config, state=bootstrap_state, logger=logger)
    console = Console()

    ok, message = await runtime.startup_check()
    if not ok:
        console.print(Panel(message, title="Startup Error", border_style="red"))
        return 1

    runtime.state.log_event(message)
    runtime.post_system_message("Type your message below. Use /help for commands.")
    runtime.start()
    input_controller = InputController()
    should_exit = False

    try:
        with Live(console=console, refresh_per_second=config.cli.refresh_hz, screen=True) as live:
            while not should_exit:
                for line in input_controller.poll_completed_lines():
                    result = handle_command(line, runtime)
                    if result.handled:
                        if result.system_message:
                            runtime.post_system_message(result.system_message)
                        if result.should_exit:
                            should_exit = True
                            break
                    else:
                        runtime.submit_user_message(line)
                live.update(render_dashboard(runtime.state.snapshot(), config, input_controller.buffer))
                await asyncio.sleep(max(0.1, 1 / max(config.cli.refresh_hz, 1)))
    except KeyboardInterrupt:
        should_exit = True
    finally:
        with suppress(Exception):
            runtime.stop()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Proactive CLI Agent")
    parser.add_argument("--config", dest="config_path", default=None, help="Optional path to a TOML config file.")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_cli(args.config_path)))
