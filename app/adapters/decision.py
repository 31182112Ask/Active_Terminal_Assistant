from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.adapters.ollama import OllamaTransport
from app.prompts import load_prompt


class DecisionOutputParseError(ValueError):
    pass


@dataclass(slots=True)
class ParsedDecision:
    decision: str
    reason: str
    confidence: float
    intent: str | None = None
    window: str | None = None
    urgency: str | None = None
    suggested_topic: str | None = None


class DecisionModelAdapter:
    def __init__(self, transport: OllamaTransport, model: str, temperature: float):
        self._transport = transport
        self._model = model
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    async def decide(self, compact_context: str, recent_messages: list[dict[str, str]]) -> ParsedDecision:
        prompt = load_prompt("proactive_decision.txt")
        context_block = {
            "compact_context": compact_context,
            "recent_messages": recent_messages,
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(context_block, ensure_ascii=False, indent=2)},
            ],
            "options": {"temperature": self._temperature},
        }
        response = await self._transport.chat_once(payload)
        content = response.get("message", {}).get("content", "")
        return self.parse(content)

    @staticmethod
    def parse(raw: str) -> ParsedDecision:
        candidate = raw.strip()
        if not candidate:
            raise DecisionOutputParseError("empty decision output")
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not match:
                raise DecisionOutputParseError(f"unable to parse decision output: {raw}") from None
            data = json.loads(match.group(0))
        decision = str(data.get("decision", "")).upper()
        if decision not in {"WAIT", "SPEAK"}:
            raise DecisionOutputParseError(f"invalid decision: {decision}")
        confidence = float(data.get("confidence", 0.0))
        return ParsedDecision(
            decision=decision,
            reason=str(data.get("reason", "")).strip() or "model returned no reason",
            confidence=max(0.0, min(1.0, confidence)),
            intent=str(data.get("intent")).strip() if data.get("intent") else None,
            window=str(data.get("window")).strip() if data.get("window") else None,
            urgency=str(data.get("urgency")).strip() if data.get("urgency") else None,
            suggested_topic=str(data.get("suggested_topic")).strip() if data.get("suggested_topic") else None,
        )
