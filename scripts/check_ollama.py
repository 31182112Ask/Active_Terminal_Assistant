from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters import OllamaTransport


async def _run(base_url: str, dialogue_model: str, decision_model: str) -> int:
    transport = OllamaTransport(base_url=base_url, timeout_seconds=30)
    console = Console()
    try:
        models = await transport.list_models()
    except Exception as exc:
        console.print(Panel(f"Could not reach Ollama at {base_url}\n\n{exc}", title="Health Check Failed", border_style="red"))
        return 1
    missing = [model for model in (dialogue_model, decision_model) if model not in models]
    if missing:
        console.print(
            Panel(
                f"Ollama is reachable, but these models are missing:\n- " + "\n- ".join(missing),
                title="Models Missing",
                border_style="yellow",
            )
        )
        return 1
    console.print(Panel("Ollama is reachable and required models are present.", title="Health Check OK", border_style="green"))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local Ollama availability for the proactive CLI agent.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--dialogue-model", default="qwen3:14b")
    parser.add_argument("--decision-model", default="qwen3:1.7b")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args.base_url, args.dialogue_model, args.decision_model)))


if __name__ == "__main__":
    main()
