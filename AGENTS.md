# AGENTS.md

## Project Name

Local Proactive CLI Agent

## Project Goal

Build a fully local proactive conversational agent that runs on Ollama, uses `qwen3:14b` as the main dialogue model, uses `qwen3:1.7b` as the decision model for proactive re-speaking, integrates `leomariga/ProactiveAgent` as the proactive scheduling and decision backbone, and uses **agere** to manage the workflow as an explicit agent pipeline.

The system must provide a **standalone CLI conversation interface** designed specifically for this project, not a minimal REPL. The CLI should display as much useful information as is reasonable without becoming noisy, and should support a smooth long-running interactive session.

This project is not a generic chat shell. It is a structured local agent system whose key feature is:

- the assistant can answer normally to user input,
- then, after already having answered,
- it can periodically decide whether it should speak again **without new user input**,
- and do so under explicit workflow control rather than ad hoc polling code.

---

## Core Product Definition

The product is a local terminal-based proactive assistant with these properties:

1. **Fully local inference** through Ollama.
2. **Two-model architecture**:
   - Dialogue model: `qwen3:14b`
   - Decision model: `qwen3:1.7b`
3. **Proactive behavior** built around `leomariga/ProactiveAgent`.
4. **Workflow orchestration** built with **agere**, rather than scattered callback logic.
5. **Standalone CLI UI** with clear panels / sections / status regions.
6. **Text-first interaction** as the primary target.
7. Optional support for voice input/output may be designed as an extension point, but text I/O is the required baseline.

---

## Required Behavioral Scope

### Required

The assistant must support the following baseline behavior:

- normal multi-turn user-assistant dialogue,
- streaming or near-streaming assistant output in CLI,
- memory of recent turns within the current session,
- periodic wake-up checks after the assistant has already spoken,
- proactive decision-making when the user is silent,
- explicit separation between:
  - **decision to speak**,
  - **generation of what to say**,
  - **rendering to the CLI**,
- interruption-safe behavior when the user sends a new message while the system is idle or preparing a proactive response.

### Not required in v1

The following are not mandatory unless implementation is straightforward:

- long-term vector memory,
- GUI,
- remote deployment,
- browser control,
- speech synthesis,
- ASR,
- mobile packaging,
- multimodal input.

These may be left as future extension points, but the architecture should not block them.

---

## Primary User Experience

A user starts the CLI, sees system state, model state, and conversation regions, then chats normally.

After the assistant replies, the system enters a managed idle phase. During idle:

- a periodic scheduler wakes up,
- the decision model evaluates whether the assistant should speak again,
- if the answer is no, the system returns to idle,
- if the answer is yes, the main model generates a proactive follow-up,
- the proactive message is clearly labeled in the CLI.

The user must always be able to type a new message and reclaim control of the session.

The assistant's proactive behavior must feel deliberate rather than random spam.

---

## Architectural Requirements

### High-Level Components

Implement the project with the following components.

#### 1. CLI Application Layer
Responsible for:
- launching the session,
- rendering conversation and status panels,
- accepting user input,
- handling keyboard interrupts / graceful exit,
- showing whether the agent is listening, thinking, idle, deciding, or proactively speaking.

#### 2. Workflow Layer (agere)
Use **agere** as the main workflow manager.

The workflow layer must explicitly model at least the following states or stages:
- user input received,
- assistant reply generation,
- reply finalization,
- idle waiting,
- proactive wake-up,
- speak / wait decision,
- proactive generation,
- cancellation or interruption,
- error recovery.

Do not hide workflow logic in a single monolithic loop.

#### 3. Proactive Scheduling / Decision Layer
Use `leomariga/ProactiveAgent` for the proactive decision backbone.

This layer should:
- manage wake/sleep style periodic checks,
- expose configurable minimum and maximum intervals,
- separate timing policy from response generation,
- allow the system to remain silent for long periods if appropriate.

#### 4. Decision Model Adapter
A dedicated adapter for `qwen3:1.7b` via Ollama.

This model is responsible for deciding whether the assistant should proactively speak again.
It must not generate the full end-user response unless explicitly configured for debugging.

Its output should be normalized into a structured decision such as:
- `WAIT`
- `SPEAK`
- optionally with metadata like reason, confidence, suggested urgency, or suggested topic.

#### 5. Dialogue Model Adapter
A dedicated adapter for `qwen3:14b` via Ollama.

This model is responsible for:
- normal user-facing responses,
- proactive follow-up responses when approved by the decision layer,
- optional streaming token output.

#### 6. Session / Context Manager
Responsible for:
- storing recent dialogue turns,
- tracking timestamps,
- tracking last user activity,
- tracking last assistant output,
- recording whether the assistant has already spoken in the current cycle,
- providing compact context views for the decision model.

#### 7. Configuration Layer
All major timings, model names, prompt templates, and UI settings must be configurable.
Use a structured config file or environment-backed configuration.

---

## Explicit Design Principle: Decision Model vs Dialogue Model

The decision model and dialogue model must remain cleanly separated.

### Decision model (`qwen3:1.7b`)
Use for:
- whether to proactively speak,
- whether silence should continue,
- whether a proactive message would be useful,
- optionally selecting a proactive intent label.

Do not use it as the final assistant persona.

### Dialogue model (`qwen3:14b`)
Use for:
- normal assistant replies,
- proactive follow-up replies,
- final user-visible text generation.

This separation is mandatory.

---

## Decision Policy Requirements

The proactive decision system must not be a trivial “always speak every N seconds” mechanic.

It should consider at least:
- time since last assistant message,
- time since last user message,
- whether the assistant already asked a question,
- whether there is unresolved user intent,
- whether a follow-up would be helpful,
- whether another message would be repetitive or annoying,
- whether the current conversation context gives a natural reason to continue.

A proactive decision should generally be conservative.

The default behavior should prefer silence unless there is a plausible conversational reason to speak.

The project should support two modes:

1. **Rule-assisted decision mode**
   - rule gates before or after model judgment,
   - e.g. cooldown windows, duplicate suppression, max consecutive proactive turns.

2. **Model-first decision mode**
   - the small model is the main judge,
   - rules act only as safety constraints.

Both modes should be configurable.

---

## Minimum Safety / Anti-Spam Constraints

Implement these constraints by default:

- minimum cooldown after any assistant message,
- minimum cooldown after a proactive message,
- cap on consecutive proactive messages without user reply,
- duplicate-content suppression,
- suppression after explicit user disengagement,
- suppression if the last assistant turn already ends with a direct question unless enough time has passed,
- immediate cancellation of pending proactive output when a new user message arrives.

The system should avoid producing repetitive “just checking in” style filler.

---

## CLI Requirements

The CLI must be custom-designed and informative. It should not be only:
- a single prompt line,
- a plain transcript dump,
- or a basic `input()` / `print()` loop.

### CLI Goals

The interface should make the system observable during development and usable during long sessions.

### Required Information Layout

Display as many of the following as is reasonable:

#### Header / Session Bar
- app name,
- current session ID,
- uptime,
- current mode,
- currently selected models,
- whether proactive mode is enabled.

#### State Panel
- workflow state,
- current agent phase,
- idle timer,
- next wake-up countdown,
- last user activity time,
- last assistant activity time,
- number of consecutive proactive turns.

#### Conversation Panel
- clearly rendered user and assistant messages,
- proactive messages visually distinct,
- timestamps,
- optional message IDs or turn numbers.

#### Decision Diagnostics Panel
At minimum show concise decision diagnostics such as:
- last decision = WAIT / SPEAK,
- decision timestamp,
- short reason summary,
- cooldown status,
- whether a rule gate blocked speaking.

#### Input Area
- a clear prompt region for user input,
- support for slash commands.

### Optional but Strongly Encouraged
- keyboard shortcuts,
- collapsible diagnostics,
- scrollback,
- event log pane,
- token / latency stats,
- debug mode.

### CLI Technical Recommendation
A rich terminal framework is acceptable and preferred if it improves structure, for example Textual or Rich-based composition. Keep the implementation maintainable.

---

## Suggested Slash Commands

Implement a practical set of commands such as:

- `/help`
- `/quit`
- `/clear`
- `/status`
- `/models`
- `/proactive on`
- `/proactive off`
- `/debug on`
- `/debug off`
- `/history`
- `/config`
- `/reset`
- `/poke` (force an immediate proactive decision cycle for testing)
- `/speak-now` (manual debug trigger for proactive generation)

These commands are not all mandatory, but the CLI must include enough operator control for testing and evaluation.

---

## agere Workflow Requirements

The implementation must use **agere** as a real workflow orchestration mechanism, not as a decorative dependency.

### Expected Workflow Shape

At minimum, the workflow should represent logic similar to:

1. Wait for user input or timer wake event.
2. If user input arrives:
   - cancel pending proactive activity,
   - append turn to session state,
   - generate main reply,
   - stream output to CLI,
   - update session state,
   - return to idle.
3. If timer wake event arrives:
   - check hard rule gates,
   - build compact decision context,
   - ask decision model whether to speak,
   - if WAIT, schedule next wake,
   - if SPEAK, generate proactive reply with main model,
   - render it distinctly,
   - update counters and timestamps,
   - return to idle.
4. If interruption occurs:
   - stop or suppress pending proactive output,
   - give priority to user input.

The workflow implementation should be modular enough that future additions such as voice I/O or memory modules can plug in without rewriting the core state machine.

---

## ProactiveAgent Integration Requirements

Use `leomariga/ProactiveAgent` for the scheduling / wake-sleep decision backbone where appropriate.

The code should show clear integration points for:
- wake-up interval policy,
- decision execution,
- sleep calculation,
- configurable response interval windows,
- custom decision engine using Ollama-backed `qwen3:1.7b`.

Do not reduce ProactiveAgent to a dead dependency. The repository should make it obvious how and where it contributes to runtime behavior.

---

## Ollama Integration Requirements

The project must run locally with Ollama.

### Required capabilities
- configure model names from settings,
- health check Ollama connectivity on startup,
- fail gracefully if a model is unavailable,
- cleanly separate decision-model and dialogue-model clients,
- support generation parameters by role,
- optionally support streaming for the dialogue model.

### Baseline model assignments
- Dialogue model: `qwen3:14b`
- Decision model: `qwen3:1.7b`

These should be defaults, not hardcoded constants that are difficult to change.

---

## Prompting Requirements

Create separate prompt templates for:

1. **Main assistant reply**
2. **Proactive decision judgment**
3. **Proactive message generation**
4. **Optional summarization or compact context extraction**

### Decision Prompt Expectations
The decision prompt should be optimized for structured output. Prefer short deterministic output, for example JSON or a strict tag format.

Example conceptual output:

```json
{
  "decision": "WAIT",
  "reason": "assistant already asked a question and cooldown not elapsed",
  "confidence": 0.81
}
```

The exact schema may differ, but it must be machine-parseable and resilient.

### Proactive Generation Prompt Expectations
The proactive generation prompt must constrain style to avoid:
- repetitive follow-up spam,
- unnatural self-interruptions,
- generic filler,
- claims of background awareness the system does not have.

---

## Session State Requirements

Track at least:
- turn history,
- timestamps for each turn,
- last user turn time,
- last assistant turn time,
- last proactive turn time,
- last decision result,
- consecutive proactive count,
- whether the assistant is awaiting user answer,
- whether a pending proactive attempt was cancelled,
- current workflow state.

This state must be inspectable in debug mode.

---

## Logging and Observability

The project must include useful logs.

### Required logs
- startup and model readiness,
- workflow transitions,
- decision cycles,
- decision outputs,
- proactive suppression reasons,
- generation failures,
- interruption handling,
- shutdown.

### Strongly encouraged
- structured logs,
- separate debug vs user-facing logs,
- optional session transcript export,
- optional event timeline export for evaluation.

---

## Error Handling Requirements

The system must fail predictably.

Handle at minimum:
- Ollama unavailable,
- missing model,
- malformed decision output,
- timeout from decision model,
- timeout from dialogue model,
- interrupted render,
- invalid config,
- user pressing Ctrl+C,
- workflow cancellation race conditions.

Where possible, recover and return to a usable idle state instead of crashing.

---

## Repository Structure Expectations

A clean repository structure is expected. A possible layout:

```text
project/
  app/
    cli/
    workflow/
    agents/
    adapters/
    prompts/
    state/
    services/
    config/
    utils/
  tests/
    unit/
    integration/
    e2e/
  scripts/
  docs/
  AGENTS.md
  README.md
  pyproject.toml
```

Exact names may vary, but separation of concerns must be obvious.

---

## Code Quality Requirements

- Use Python with clear typing.
- Prefer small focused modules over large files.
- Keep workflow logic explicit and testable.
- Avoid hidden global state.
- Use dataclasses or equivalent structured objects for session and decision data.
- Provide docstrings where they add value.
- Keep prompt templates versioned and easy to edit.
- Avoid tightly coupling CLI rendering to agent logic.

---

## Test Requirements

A serious testing baseline is required.

### Unit tests
Test at least:
- decision parsing,
- cooldown logic,
- duplicate suppression,
- state transitions,
- config loading,
- slash command handling.

### Integration tests
Test at least:
- decision model adapter with mocked Ollama,
- dialogue model adapter with mocked Ollama,
- proactive cycle from idle to decision to response,
- interruption on new user input,
- workflow recovery after model error.

### End-to-end tests
At minimum, include one or more scripted terminal-session style tests that validate:
- normal reply path,
- idle wait path,
- proactive speak path,
- proactive suppression path,
- user interruption path.

Mocked tests are acceptable for CI reliability, but at least one documented manual runbook for real local Ollama validation is required.

---

## Performance Expectations

This is a local system. Responsiveness matters.

### Soft targets
- decision cycle should feel lightweight compared with full reply generation,
- user input should interrupt pending proactive activity promptly,
- CLI should remain readable during long sessions,
- the agent should not accumulate runaway background tasks.

Do not over-optimize prematurely, but avoid obviously blocking architecture mistakes.

---

## Deliverables

The implementation is expected to produce:

1. A runnable local CLI application.
2. Working Ollama integration for both models.
3. agere-based workflow orchestration.
4. ProactiveAgent-backed periodic proactive decision loop.
5. Clear configuration and prompt files.
6. Tests.
7. Documentation sufficient for local setup and usage.

---

## Acceptance Targets

The project is accepted only if all of the following are true.

### A. Startup and Configuration
- The app starts locally from documented instructions.
- It verifies Ollama availability.
- It clearly reports missing models or configuration problems.

### B. Normal Dialogue
- The user can chat with the assistant in the CLI.
- The main model uses `qwen3:14b` by default.
- Assistant output is rendered clearly and reliably.

### C. Proactive Decision Pipeline
- After the assistant has already replied, the system can enter idle mode.
- During idle, the system periodically wakes without user input.
- The decision model uses `qwen3:1.7b` by default.
- The decision result is structurally parsed and visible in debug/status output.
- The system can both choose `WAIT` and choose `SPEAK` under different contexts.

### D. Proactive Output
- When the decision is `SPEAK`, the system generates a proactive follow-up with the main model.
- Proactive messages are visibly distinguished from normal replies.
- The system does not spam repeated follow-ups due to missing cooldown protection.

### E. Interruption and Control
- A new user message can interrupt or supersede pending proactive behavior.
- Slash commands provide basic runtime control.
- The user can disable proactive mode during a session.

### F. Workflow Integrity
- agere is actually used to organize workflow transitions.
- ProactiveAgent is actually used in the proactive scheduling / wake-sleep logic.
- The implementation is modular enough that the decision layer can be swapped later.

### G. CLI Quality
- The CLI is not a barebones prompt loop.
- It displays useful live session information.
- It remains readable over extended interaction.

### H. Reliability
- The program handles common failures without immediate unrecoverable crash.
- Tests exist for critical proactive logic.

If any of the above are missing, the project is not complete.

---

## Stretch Goals

These are optional and should not delay the core deliverable:

- voice input/output adapters,
- transcript export,
- persistence across sessions,
- summarization memory,
- richer Textual UI,
- configurable decision explanations,
- replayable event traces,
- benchmark scripts for proactive behavior evaluation.

---

## Evaluation Scenarios

The following scenarios should be manually demonstrated.

### Scenario 1: Standard Chat
User asks a normal question. Assistant answers. No proactive follow-up occurs during cooldown.

### Scenario 2: Useful Proactive Follow-up
User asks for help planning something, assistant answers partially, context naturally supports a follow-up, idle cycle triggers a `SPEAK`, and assistant offers a useful continuation.

### Scenario 3: Suppressed Follow-up
Assistant already asked a direct question. Idle cycle runs. Decision system or rules suppress another message.

### Scenario 4: User Interrupts
System is idle or preparing a proactive response. User types a new message. User input takes priority cleanly.

### Scenario 5: Debug Observability
Operator can inspect current state, last decision, cooldowns, and proactive counters from the CLI.

---

## Non-Goals

Do not spend core project time on:
- web frontend,
- cloud deployment,
- multi-user server mode,
- animated avatars,
- nonessential plugin ecosystems,
- premature generalization into a framework.

The focus is a robust local proactive CLI agent.

---

## Final Instruction to the Coding Agent

When implementation tradeoffs arise, prioritize:

1. correctness of proactive workflow,
2. clear separation of decision and dialogue roles,
3. observability in the CLI,
4. interruption safety,
5. maintainable modular structure.

Avoid fake completion. If a dependency cannot support the intended role directly, build a thin adapter and document the limitation clearly.

The final result should be something that a developer can run locally, inspect easily, and use to verify that proactive re-speaking works in a controlled and testable way.
