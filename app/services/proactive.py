from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from proactiveagent import FunctionBasedSleepCalculator
from proactiveagent.providers.base import BaseProvider
from proactiveagent.scheduler import WakeUpScheduler

from app.config import AppConfig
from app.state import SessionSnapshot


class _NoopProvider(BaseProvider):
    async def generate_response(self, messages, system_prompt=None, triggered_by_user_message=False, **kwargs):
        raise NotImplementedError("generate_response is not used by the CLI scheduler bridge")

    async def should_respond(self, messages, elapsed_time, context):
        raise NotImplementedError("should_respond is not used by the CLI scheduler bridge")

    async def calculate_sleep_time(self, wake_up_pattern, min_sleep_time, max_sleep_time, context):
        return min_sleep_time, "fallback provider sleep time"


class ProactiveSchedulerBridge:
    def __init__(
        self,
        config: AppConfig,
        context_provider: Callable[[], SessionSnapshot],
        sleep_callback: Callable[[int, str], None],
        wake_callback: Callable[[dict[str, Any]], None],
    ):
        self._config = config
        self._context_provider = context_provider
        self._sleep_callback = sleep_callback
        self._wake_callback = wake_callback
        self._provider = _NoopProvider(model=config.models.decision)
        self._scheduler = WakeUpScheduler(
            provider=self._provider,
            config={
                "wake_up_pattern": config.proactive.wake_up_pattern,
                "min_sleep_time": config.proactive.min_sleep_seconds,
                "max_sleep_time": config.proactive.max_sleep_seconds,
            },
            get_sleep_time_callbacks_func=lambda: [self._sleep_callback],
            sleep_time_calculator=FunctionBasedSleepCalculator(self._calculate_sleep_time),
        )
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="proactive-scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._scheduler.stop()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def interrupt_sleep(self) -> None:
        self._scheduler.interrupt_sleep()

    def force_wake(self) -> None:
        snapshot = self._context_provider()
        self._wake_callback(
            {
                "wake_up_time": datetime.now().timestamp(),
                "forced": True,
                "session_id": snapshot.session_id,
            }
        )

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(
                self._scheduler.start(
                    wake_up_callback=self._wake_callback,
                    context_provider=self._build_scheduler_context,
                )
            )
        finally:
            self._loop.close()
            self._loop = None
            asyncio.set_event_loop(None)

    async def _calculate_sleep_time(self, config: dict[str, Any], context: dict[str, Any]) -> tuple[int, str]:
        min_sleep = int(config["min_sleep_time"])
        max_sleep = int(config["max_sleep_time"])
        short_min = int(self._config.proactive.short_window_min_sleep_seconds)
        short_max = int(self._config.proactive.short_window_max_sleep_seconds)
        long_min = int(self._config.proactive.long_window_min_sleep_seconds)
        long_max = int(self._config.proactive.long_window_max_sleep_seconds)
        if not context.get("has_recent_user_turn"):
            return max_sleep, "long:no recent user activity, using max sleep"
        if context.get("awaiting_user_answer"):
            value = min(max_sleep, max(long_min, self._config.proactive.question_cooldown_seconds))
            return value, "long:assistant is waiting for user answer"
        if context.get("consecutive_proactive_turns", 0) > 0:
            value = min(max_sleep, max(long_min, long_min + 18 * context["consecutive_proactive_turns"]))
            return value, "long:backing off after a proactive turn"
        if context.get("last_assistant_seconds") is not None and context["last_assistant_seconds"] < self._config.proactive.recent_reply_threshold_seconds:
            value = max(short_min, min(short_max, 12))
            return value, "short:light continuation window after recent reply"
        value = min(max_sleep, max(long_min, min(long_max, 50)))
        value = max(value, min_sleep)
        return value, "long:default proactive window"

    def _build_scheduler_context(self) -> dict[str, Any]:
        snapshot = self._context_provider()
        now = datetime.now()
        last_user_seconds = (
            int((now - snapshot.last_user_activity).total_seconds()) if snapshot.last_user_activity else None
        )
        last_assistant_seconds = (
            int((now - snapshot.last_assistant_activity).total_seconds()) if snapshot.last_assistant_activity else None
        )
        return {
            "session_id": snapshot.session_id,
            "has_recent_user_turn": snapshot.last_user_activity is not None,
            "awaiting_user_answer": snapshot.awaiting_user_answer,
            "consecutive_proactive_turns": snapshot.consecutive_proactive_turns,
            "last_user_seconds": last_user_seconds,
            "last_assistant_seconds": last_assistant_seconds,
            "phase": snapshot.phase.value,
        }
