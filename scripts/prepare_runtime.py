from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters import OllamaTransport


async def _list_models(base_url: str) -> list[str]:
    transport = OllamaTransport(base_url=base_url, timeout_seconds=15)
    return await transport.list_models()


def _start_ollama_serve(console: Console) -> bool:
    if shutil.which("ollama") is None:
        console.print(Panel("`ollama` command not found in PATH.", title="Ollama Missing", border_style="red"))
        return False
    console.print("[yellow]Ollama is not reachable. Trying to start `ollama serve` in the background...[/]")
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "shell": False,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(["ollama", "serve"], **kwargs)
    return True


def _pull_missing_models(console: Console, missing: list[str]) -> bool:
    for model in missing:
        console.print(f"[yellow]Pulling missing model: {model}[/]")
        result = subprocess.run(["ollama", "pull", model], check=False)
        if result.returncode != 0:
            console.print(Panel(f"Failed to pull model: {model}", title="Model Pull Failed", border_style="red"))
            return False
    return True


async def _prepare(base_url: str, dialogue_model: str, decision_model: str) -> int:
    console = Console()
    required = [dialogue_model, decision_model]

    try:
        models = await _list_models(base_url)
    except Exception:
        if not _start_ollama_serve(console):
            return 1
        deadline = time.time() + 20
        models = []
        while time.time() < deadline:
            try:
                models = await _list_models(base_url)
                break
            except Exception:
                await asyncio.sleep(1)
        else:
            console.print(Panel("Ollama did not become reachable in time.", title="Startup Timeout", border_style="red"))
            return 1

    missing = [model for model in required if model not in models]
    if missing and not _pull_missing_models(console, missing):
        return 1

    console.print(
        Panel(
            "Local runtime is prepared.\n"
            f"Dialogue model: {dialogue_model}\n"
            f"Decision model: {decision_model}",
            title="Runtime Ready",
            border_style="green",
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure local Ollama runtime and required models are ready.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--dialogue-model", default="qwen3:14b")
    parser.add_argument("--decision-model", default="qwen3:1.7b")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_prepare(args.base_url, args.dialogue_model, args.decision_model)))


if __name__ == "__main__":
    main()
