from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from typing import Any

import httpx


class OllamaTransport:
    def __init__(self, base_url: str, timeout_seconds: float):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._clients: dict[int, httpx.AsyncClient] = {}
        self._clients_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return self._base_url

    def _get_client(self) -> httpx.AsyncClient:
        thread_id = threading.get_ident()
        with self._clients_lock:
            client = self._clients.get(thread_id)
            if client is None:
                client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
                self._clients[thread_id] = client
            return client

    async def close(self) -> None:
        thread_id = threading.get_ident()
        with self._clients_lock:
            client = self._clients.pop(thread_id, None)
        if client is not None:
            await client.aclose()

    async def list_models(self) -> list[str]:
        client = self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        payload = response.json()
        return [item["name"] for item in payload.get("models", [])]

    async def chat_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def chat_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        client = self._get_client()
        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                yield json.loads(line)
