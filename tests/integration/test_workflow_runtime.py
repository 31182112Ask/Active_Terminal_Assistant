from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import pytest

from app.adapters.decision import ParsedDecision
from app.adapters.dialogue import GenerationCancelled
from app.config import AppConfig, CliConfig, LoggingConfig, ModelsConfig, ProactiveConfig
from app.state import SessionStateManager
from app.workflow import WorkflowOrchestrator


def make_config() -> AppConfig:
    return AppConfig(
        models=ModelsConfig(stream=True, request_timeout_seconds=5, max_context_turns=8),
        proactive=ProactiveConfig(
            enabled=True,
            min_assistant_cooldown_seconds=0,
            min_proactive_cooldown_seconds=0,
            question_cooldown_seconds=0,
            min_sleep_seconds=1,
            max_sleep_seconds=2,
        ),
        cli=CliConfig(debug=True, refresh_hz=4),
        logging=LoggingConfig(level="INFO", directory="logs"),
    )


def make_runtime() -> WorkflowOrchestrator:
    config = make_config()
    state = SessionStateManager(
        dialogue_model=config.models.dialogue,
        decision_model=config.models.decision,
        debug_enabled=config.cli.debug,
        proactive_enabled=config.proactive.enabled,
    )
    logger = logging.getLogger(f"test-runtime-{state.session_id}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    runtime = WorkflowOrchestrator(config=config, state=state, logger=logger)
    runtime._scheduler.start = lambda: None
    runtime._scheduler.stop = lambda: None
    runtime._scheduler.interrupt_sleep = lambda: None
    runtime._scheduler.force_wake = lambda: runtime._on_scheduler_wake({"forced": True})
    return runtime


async def wait_for(predicate, timeout: float = 3.0) -> None:
    start = asyncio.get_running_loop().time()
    while asyncio.get_running_loop().time() - start < timeout:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition was not met before timeout")


@pytest.mark.asyncio
async def test_runtime_handles_normal_reply_and_proactive_follow_up() -> None:
    runtime = make_runtime()

    async def fake_generate(messages, proactive, on_token, cancellation):
        content = "普通回复" if not proactive else "这是一个主动补充建议。"
        await on_token(content)
        return content

    async def fake_decide(compact_context, recent_messages):
        return ParsedDecision("SPEAK", "unfinished plan", 0.88, "medium", "next steps")

    runtime._dialogue.generate = fake_generate
    runtime._decision_adapter.decide = fake_decide

    runtime.start()
    runtime.submit_user_message("帮我做旅行计划")

    await wait_for(lambda: len(runtime.state.snapshot().turns) >= 2)
    runtime._on_scheduler_wake({"forced": True})
    await wait_for(lambda: any(turn.proactive for turn in runtime.state.snapshot().turns))

    snapshot = runtime.state.snapshot()
    assert snapshot.turns[0].role == "user"
    assert snapshot.turns[1].role == "assistant"
    assert any(turn.proactive for turn in snapshot.turns)
    runtime.stop()


@pytest.mark.asyncio
async def test_runtime_cancels_proactive_generation_when_user_interrupts() -> None:
    runtime = make_runtime()

    async def fake_generate(messages, proactive, on_token, cancellation):
        if proactive:
            for chunk in ["主", "动", "输", "出"]:
                if cancellation.cancelled:
                    raise GenerationCancelled("cancelled")
                await on_token(chunk)
                await asyncio.sleep(0.05)
            return "主动输出"
        await on_token("普通回复")
        return "普通回复"

    async def fake_decide(compact_context, recent_messages):
        return ParsedDecision("SPEAK", "helpful follow-up", 0.8, "low", "follow-up")

    runtime._dialogue.generate = fake_generate
    runtime._decision_adapter.decide = fake_decide

    runtime.start()
    runtime.submit_user_message("先回答我")
    await wait_for(lambda: len(runtime.state.snapshot().turns) >= 2)

    runtime.trigger_speak_now()
    await asyncio.sleep(0.06)
    runtime.submit_user_message("我有新问题")

    await wait_for(lambda: len(runtime.state.snapshot().turns) >= 4)
    snapshot = runtime.state.snapshot()
    proactive_turns = [turn for turn in snapshot.turns if turn.proactive]
    assert proactive_turns == []
    assert snapshot.turns[-2].role == "user"
    assert snapshot.turns[-1].role == "assistant"
    runtime.stop()


@pytest.mark.asyncio
async def test_runtime_recovers_after_generation_error() -> None:
    runtime = make_runtime()

    async def exploding_generate(messages, proactive, on_token, cancellation):
        raise RuntimeError("boom")

    runtime._dialogue.generate = exploding_generate

    runtime.start()
    runtime.submit_user_message("测试错误恢复")
    await wait_for(lambda: runtime.state.snapshot().last_error is not None)
    snapshot = runtime.state.snapshot()
    assert "generation failure" in snapshot.last_error
    assert snapshot.phase.value in {"idle_waiting", "error_recovery"}
    runtime.stop()

