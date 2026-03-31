from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from agere.commander import BasicJob, CommanderAsync, PASS_WORD, handler
from proactiveagent import FunctionBasedDecisionEngine

from app.adapters import (
    CancellationHandle,
    DecisionModelAdapter,
    DecisionOutputParseError,
    DialogueModelAdapter,
    GenerationCancelled,
    OllamaTransport,
)
from app.config import AppConfig
from app.services import ProactiveSchedulerBridge, RuleGateService, build_compact_decision_context
from app.state import DecisionRecord, SessionStateManager, WorkflowPhase


class WorkflowOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        state: SessionStateManager,
        logger: logging.Logger,
    ):
        self._config = config
        self._state = state
        self._logger = logger
        self._transport = OllamaTransport(
            base_url=config.models.base_url,
            timeout_seconds=config.models.request_timeout_seconds,
        )
        self._dialogue = DialogueModelAdapter(
            transport=self._transport,
            model=config.models.dialogue,
            temperature=config.models.dialogue_temperature,
            stream=config.models.stream,
        )
        self._decision_adapter = DecisionModelAdapter(
            transport=self._transport,
            model=config.models.decision,
            temperature=config.models.decision_temperature,
        )
        self._rule_gate = RuleGateService(config.proactive)
        self._decision_engine = FunctionBasedDecisionEngine(self._decision_via_model)
        self._commander = CommanderAsync(logger=logger)
        self._commander_thread: threading.Thread | None = None
        self._generation_lock = threading.Lock()
        self._active_generation: CancellationHandle | None = None
        self._last_parsed_decision = None
        self._scheduler = ProactiveSchedulerBridge(
            config=config,
            context_provider=self._state.snapshot,
            sleep_callback=self._on_sleep_planned,
            wake_callback=self._on_scheduler_wake,
        )

    @property
    def state(self) -> SessionStateManager:
        return self._state

    @property
    def config(self) -> AppConfig:
        return self._config

    async def startup_check(self) -> tuple[bool, str]:
        try:
            models = await self._transport.list_models()
        except Exception as exc:
            return False, f"Ollama unreachable at {self._config.models.base_url}: {exc}"
        missing = [model for model in (self._config.models.dialogue, self._config.models.decision) if model not in models]
        if missing:
            return False, f"Missing Ollama model(s): {', '.join(missing)}"
        return True, "Ollama and required models are ready"

    def start(self) -> None:
        if self._commander_thread and self._commander_thread.is_alive():
            return
        self._state.set_phase(WorkflowPhase.READY)
        self._state.log_event("workflow runtime starting")
        self._commander_thread = threading.Thread(target=self._run_commander, daemon=True, name="agere-commander")
        self._commander_thread.start()
        self._wait_for_commander_ready()
        self._scheduler.start()
        self._schedule_idle_transition("runtime ready")

    def stop(self) -> None:
        self._state.set_phase(WorkflowPhase.SHUTDOWN)
        self._state.log_event("workflow runtime stopping")
        self._cancel_active_generation("shutdown")
        self._scheduler.stop()
        if self._commander.running_status:
            self._commander.exit(wait=True)
        if self._commander_thread and self._commander_thread.is_alive():
            self._commander_thread.join(timeout=5)

    def submit_user_message(self, text: str) -> None:
        self._scheduler.interrupt_sleep()
        self._state.log_event("user input received")
        self._enqueue(self._handle_cancellation("user input preempted proactive output"))
        self._enqueue(self._handle_user_input(text))

    def trigger_poke(self) -> None:
        self._state.log_event("manual proactive poke requested", level="debug")
        self._scheduler.force_wake()

    def trigger_speak_now(self) -> None:
        self._state.log_event("manual proactive generation requested", level="debug")
        self._enqueue(self._handle_proactive_generation({"forced": True, "reason": "manual /speak-now"}))

    def cancel_active_output(self) -> None:
        self._enqueue(self._handle_cancellation("manual cancel requested"))

    def set_proactive_enabled(self, enabled: bool) -> None:
        self._state.set_proactive_enabled(enabled)
        self._state.log_event(f"proactive mode {'enabled' if enabled else 'disabled'}")
        self._scheduler.interrupt_sleep()

    def set_debug_enabled(self, enabled: bool) -> None:
        self._state.set_debug(enabled)
        self._state.log_event(f"debug mode {'enabled' if enabled else 'disabled'}")

    def clear_history(self) -> None:
        self._state.clear_turns()
        self._state.log_event("conversation history cleared")
        self._schedule_idle_transition("history cleared")

    def reset_runtime(self) -> None:
        self._cancel_active_generation("manual reset")
        self._state.clear_turns()
        self._state.set_error(None)
        self._state.log_event("runtime state reset")
        self._scheduler.interrupt_sleep()
        self._schedule_idle_transition("runtime reset")

    def post_system_message(self, message: str) -> None:
        self._state.append_turn("system", message)
        self._state.log_event(f"system message posted: {message[:60]}", level="debug")

    def _run_commander(self) -> None:
        self._commander.run(auto_exit=False)

    def _wait_for_commander_ready(self) -> None:
        deadline = time.time() + 5
        while not self._commander.running_status and time.time() < deadline:
            time.sleep(0.05)

    def _enqueue(self, handler_coro) -> None:
        job = BasicJob(handler_coro)
        if self._commander.running_status:
            self._commander.put_job_threadsafe(job)
        else:
            self._logger.warning("Commander not running; dropped job")

    def _on_sleep_planned(self, sleep_seconds: int, reason: str) -> None:
        window, summary = self._parse_sleep_reason(reason)
        self._state.set_sleep_plan(sleep_seconds, summary, window)
        self._state.log_event(f"next wake scheduled in {sleep_seconds}s ({window or 'n/a'}): {summary}", level="debug")

    def _on_scheduler_wake(self, context: dict[str, Any]) -> None:
        self._enqueue(self._handle_timer_wake(context))

    def _schedule_idle_transition(self, reason: str) -> None:
        self._enqueue(self._handle_idle_transition(reason))

    def _cancel_active_generation(self, reason: str) -> None:
        with self._generation_lock:
            if self._active_generation is None:
                return
            self._active_generation.cancel()
            self._active_generation = None
        self._state.cancel_draft(user_cancelled=True)
        self._state.log_event(f"cancelled active generation: {reason}", level="warning")

    def _recent_messages(self) -> list[dict[str, str]]:
        snapshot = self._state.snapshot()
        return [{"role": turn.role, "content": turn.content} for turn in snapshot.turns[-self._config.models.max_context_turns :]]

    @staticmethod
    def _parse_sleep_reason(reason: str) -> tuple[str | None, str]:
        if ":" not in reason:
            return None, reason
        maybe_window, summary = reason.split(":", 1)
        if maybe_window in {"short", "long"}:
            return maybe_window, summary.strip()
        return None, reason

    async def _decision_via_model(
        self,
        messages: list[dict[str, str]],
        last_user_message_time: float,
        context: dict[str, Any],
        config: dict[str, Any],
        triggered_by_user_message: bool = False,
    ) -> tuple[bool, str]:
        compact_context = context["compact_context"]
        parsed = await self._decision_adapter.decide(compact_context, messages)
        self._last_parsed_decision = parsed
        return parsed.decision == "SPEAK", parsed.reason

    @handler(PASS_WORD)
    async def _handle_cancellation(self, self_handler, reason: str) -> None:
        self._state.set_phase(WorkflowPhase.CANCELLATION)
        self._cancel_active_generation(reason)

    @handler(PASS_WORD)
    async def _handle_user_input(self, self_handler, text: str) -> None:
        self._state.set_phase(WorkflowPhase.USER_INPUT_RECEIVED)
        self._state.append_turn("user", text)
        self._state.log_event(f"user turn stored: {text[:60]}")
        await self_handler.put_job(BasicJob(self._handle_assistant_reply(False, "user message")))

    @handler(PASS_WORD)
    async def _handle_assistant_reply(self, self_handler, proactive: bool, reason: str) -> None:
        self._state.set_phase(
            WorkflowPhase.PROACTIVE_GENERATION if proactive else WorkflowPhase.ASSISTANT_REPLY_GENERATION
        )
        self._state.start_draft(proactive=proactive)
        self._state.log_event(f"starting {'proactive' if proactive else 'assistant'} generation: {reason}")
        cancellation = CancellationHandle()
        with self._generation_lock:
            self._active_generation = cancellation

        async def on_token(chunk: str) -> None:
            self._state.update_draft(chunk)

        try:
            content = await self._dialogue.generate(
                messages=self._recent_messages(),
                proactive=proactive,
                on_token=on_token,
                cancellation=cancellation,
            )
        except GenerationCancelled:
            self._state.cancel_draft(user_cancelled=True)
            self._state.log_event("generation cancelled before finalization", level="warning")
            return
        except Exception as exc:
            await self_handler.put_job(BasicJob(self._handle_error(f"generation failure: {exc}")))
            return
        finally:
            with self._generation_lock:
                if self._active_generation is cancellation:
                    self._active_generation = None

        if proactive:
            post_rule = self._rule_gate.evaluate_post_model(self._state.snapshot(), content)
            if not post_rule.allowed:
                self._state.cancel_draft(user_cancelled=False)
                self._state.log_event(f"proactive output suppressed: {post_rule.reason}", level="warning")
                self._state.set_last_decision(
                    DecisionRecord(
                        decision="WAIT",
                        reason="proactive text suppressed after generation",
                        confidence=1.0,
                        source="combined",
                        blocked_by_rule=post_rule.reason,
                    )
                )
                await self_handler.put_job(BasicJob(self._handle_idle_transition(post_rule.reason)))
                return

        await self_handler.put_job(BasicJob(self._handle_reply_finalization(proactive)))

    @handler(PASS_WORD)
    async def _handle_reply_finalization(self, self_handler, proactive: bool) -> None:
        self._state.set_phase(WorkflowPhase.REPLY_FINALIZATION)
        turn = self._state.finalize_draft()
        if turn is None:
            self._state.log_event("draft finalization skipped because draft is empty", level="warning")
        else:
            label = "proactive" if proactive else "assistant"
            self._state.log_event(f"{label} turn finalized")
        await self_handler.put_job(BasicJob(self._handle_idle_transition("reply finalized")))

    @handler(PASS_WORD)
    async def _handle_idle_transition(self, self_handler, reason: str) -> None:
        self._state.set_phase(WorkflowPhase.IDLE_WAITING)
        self._state.log_event(f"idle transition: {reason}", level="debug")

    @handler(PASS_WORD)
    async def _handle_timer_wake(self, self_handler, context: dict[str, Any]) -> None:
        self._state.clear_sleep_plan()
        self._state.set_phase(WorkflowPhase.PROACTIVE_WAKE_UP)
        self._state.log_event("scheduler wake-up received", level="debug")
        await self_handler.put_job(BasicJob(self._handle_speak_wait_decision(context)))

    @handler(PASS_WORD)
    async def _handle_speak_wait_decision(self, self_handler, context: dict[str, Any]) -> None:
        self._state.set_phase(WorkflowPhase.SPEAK_WAIT_DECISION)
        snapshot = self._state.snapshot()
        pre_rule = self._rule_gate.evaluate_pre_model(snapshot)
        hard_block = pre_rule.reason in {
            "proactive mode disabled",
            "no conversation history yet",
            "a proactive attempt was just cancelled by user input",
            "max consecutive proactive turns reached",
            "user appears disengaged",
            "no user turn to follow up on",
        }
        if self._config.proactive.decision_mode == "rule-assisted" and not pre_rule.allowed:
            decision = DecisionRecord(
                decision="WAIT",
                reason=pre_rule.reason,
                confidence=1.0,
                source="rule",
                blocked_by_rule=pre_rule.reason,
            )
            self._state.set_last_decision(decision)
            self._state.log_event(f"decision blocked by rules: {pre_rule.reason}", level="debug")
            await self_handler.put_job(BasicJob(self._handle_idle_transition(pre_rule.reason)))
            return
        if self._config.proactive.decision_mode == "model-first" and hard_block:
            decision = DecisionRecord(
                decision="WAIT",
                reason=pre_rule.reason,
                confidence=1.0,
                source="rule",
                blocked_by_rule=pre_rule.reason,
            )
            self._state.set_last_decision(decision)
            await self_handler.put_job(BasicJob(self._handle_idle_transition(pre_rule.reason)))
            return

        try:
            compact_context = build_compact_decision_context(snapshot)
            should_respond, model_reason = await self._decision_engine.should_respond(
                self._recent_messages(),
                last_user_message_time=snapshot.last_user_activity.timestamp() if snapshot.last_user_activity else 0.0,
                context={"compact_context": compact_context, "wake_context": context},
                config=self._config.proactive.model_dump(),
                triggered_by_user_message=False,
            )
            parsed = self._last_parsed_decision
            if parsed is None:
                raise DecisionOutputParseError("decision engine did not capture parsed output")
            decision = DecisionRecord(
                decision=parsed.decision,
                reason=parsed.reason or model_reason,
                confidence=parsed.confidence,
                source="model",
                intent=parsed.intent,
                window=parsed.window if parsed.window in {"short", "long"} else snapshot.current_proactive_window,
                suggested_topic=parsed.suggested_topic,
                urgency=parsed.urgency,
            )
            should_respond = should_respond and decision.decision == "SPEAK"
        except DecisionOutputParseError as exc:
            await self_handler.put_job(BasicJob(self._handle_error(f"decision parse failure: {exc}")))
            return
        except Exception as exc:
            await self_handler.put_job(BasicJob(self._handle_error(f"decision failure: {exc}")))
            return
        finally:
            self._last_parsed_decision = None

        if not pre_rule.allowed:
            decision.decision = "WAIT"
            decision.source = "combined"
            decision.blocked_by_rule = pre_rule.reason

        self._state.set_last_decision(decision)
        self._state.log_event(f"decision result: {decision.decision} ({decision.reason})", level="debug")

        if decision.decision == "WAIT":
            await self_handler.put_job(BasicJob(self._handle_idle_transition(decision.reason)))
            return
        await self_handler.put_job(BasicJob(self._handle_proactive_generation(context)))

    @handler(PASS_WORD)
    async def _handle_proactive_generation(self, self_handler, context: dict[str, Any]) -> None:
        self._state.set_phase(WorkflowPhase.PROACTIVE_GENERATION)
        await self_handler.put_job(BasicJob(self._handle_assistant_reply(True, context.get("reason", "scheduler wake"))))

    @handler(PASS_WORD)
    async def _handle_error(self, self_handler, message: str) -> None:
        self._state.set_phase(WorkflowPhase.ERROR_RECOVERY)
        self._state.set_error(message)
        self._state.log_event(message, level="error")
        await self_handler.put_job(BasicJob(self._handle_idle_transition("error recovered to idle")))
