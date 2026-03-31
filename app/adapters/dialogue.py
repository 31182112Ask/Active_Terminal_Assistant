from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.adapters.ollama import OllamaTransport
from app.prompts import load_prompt


class GenerationCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class CancellationHandle:
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class DialogueModelAdapter:
    def __init__(self, transport: OllamaTransport, model: str, temperature: float, stream: bool = True):
        self._transport = transport
        self._model = model
        self._temperature = temperature
        self._stream = stream

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[dict[str, str]],
        proactive: bool,
        on_token: Callable[[str], Awaitable[None]],
        cancellation: CancellationHandle | None = None,
    ) -> str:
        system_prompt = load_prompt("proactive_generation.txt" if proactive else "main_assistant.txt")
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "stream": self._stream,
            "options": {"temperature": self._temperature},
        }
        if not self._stream:
            response = await self._transport.chat_once(payload)
            content = response.get("message", {}).get("content", "")
            if cancellation and cancellation.cancelled:
                raise GenerationCancelled("generation cancelled")
            await on_token(content)
            return content

        collected: list[str] = []
        async for chunk in self._transport.chat_stream(payload):
            if cancellation and cancellation.cancelled:
                raise GenerationCancelled("generation cancelled")
            content = chunk.get("message", {}).get("content", "")
            if content:
                collected.append(content)
                await on_token(content)
            if chunk.get("done"):
                break
            await asyncio.sleep(0)
        return "".join(collected)

