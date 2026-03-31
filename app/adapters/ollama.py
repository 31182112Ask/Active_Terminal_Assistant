from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class OllamaTransport:
    def __init__(self, base_url: str, timeout_seconds: float):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def base_url(self) -> str:
        return self._base_url

    async def close(self) -> None:
        return None

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
        return [item["name"] for item in payload.get("models", [])]

    async def chat_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json()

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    yield json.loads(line)
