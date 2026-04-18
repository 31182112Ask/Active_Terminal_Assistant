# Context Showcase

This file shows how the trained wait-time policy behaves across different conversational contexts.

## 1. Explicit reminder on the next morning

- Scenario: `showcase_scheduled_morning`
- Language: `en`
- Local hour / weekday: `18:00`, `2`
- Description: The user gives a clear future time anchor and asks for a reminder only after that point.
- Expected target: `54000s (15.0h)`
- Model output: `54000s (15.0h)`
- Model action: `follow_up`
- Predicted bucket: `12h-18h`

```text
user: If I still have not sent the deck by tomorrow morning, follow up.
assistant: Understood. I will check back tomorrow morning if it is still missing.
```

## 2. Urgent production incident

- Scenario: `showcase_urgent_incident`
- Language: `en`
- Local hour / weekday: `10:00`, `1`
- Description: Urgency and explicit short delay should force a very short follow-up window.
- Expected target: `37s (37s)`
- Model output: `37s (37s)`
- Model action: `follow_up`
- Predicted bucket: `30s-45s`

```text
user: Prod is still down after the restart.
assistant: Keep checking the worker logs and the failing job queue.
assistant: If I do not hear back, I will check in again in 37 seconds.
```

## 3. Explicit do-not-disturb

- Scenario: `showcase_do_not_disturb`
- Language: `zh`
- Local hour / weekday: `14:00`, `3`
- Description: The user blocks any more proactive contact for the rest of the night.
- Expected target: `suppress`
- Model output: `suppress`
- Model action: `suppress`
- Predicted bucket: `suppress`

```text
user: 我接下来一直开会，今晚别再找我。
assistant: 明白，我先停在这里，等你之后再回来。
```

## 4. Need-space emotional context

- Scenario: `showcase_need_space`
- Language: `en`
- Local hour / weekday: `21:00`, `5`
- Description: The assistant should suppress proactive outreach when the user asks for space without a return time.
- Expected target: `suppress`
- Model output: `suppress`
- Model action: `suppress`
- Predicted bucket: `suppress`

```text
user: I'm overwhelmed and need some space.
assistant: Understood. I'll wait for you to restart the conversation.
```

## 5. Missing detail blocks progress

- Scenario: `showcase_clarification`
- Language: `zh`
- Local hour / weekday: `11:00`, `3`
- Description: The task is blocked by a missing detail, so the model should keep the thread alive but not too aggressively.
- Expected target: `7200s (2.0h)`
- Model output: `7200s (2.0h)`
- Model action: `follow_up`
- Predicted bucket: `1.5h-2h`

```text
user: 这个问题不算特别急，但我还缺一个字段。
assistant: 你希望我用哪个数据源？如果你暂时没回，我两小时后再问。
```

## 6. Slow external dependency

- Scenario: `showcase_vendor_status`
- Language: `zh`
- Local hour / weekday: `14:00`, `1`
- Description: Vendor or shipment updates usually deserve a longer wait before re-engagement.
- Expected target: `86400s (24.0h)`
- Model output: `86400s (24.0h)`
- Model action: `follow_up`
- Predicted bucket: `18h-24h`

```text
user: 这个供应商回复还没到。
assistant: 如果一天后还是没有消息，我再回来确认。
```

## 7. Short accountability check-in

- Scenario: `showcase_accountability`
- Language: `en`
- Local hour / weekday: `19:00`, `0`
- Description: The user explicitly asks for a short-term accountability reminder.
- Expected target: `83s (83s)`
- Model output: `83s (83s)`
- Model action: `follow_up`
- Predicted bucket: `1m-2m`

```text
user: If I have not done the workout in 83 seconds, check back on me.
assistant: Will do. I will check back in 83 seconds if I still have not heard from you.
```

## 8. Async deliverable with explicit deadline

- Scenario: `showcase_async_deliverable`
- Language: `zh`
- Local hour / weekday: `20:00`, `2`
- Description: The assistant should wait until the user-provided delivery window instead of following up too early.
- Expected target: `68400s (19.0h)`
- Model output: `68400s (19.0h)`
- Model action: `follow_up`
- Predicted bucket: `18h-24h`

```text
user: 我会忙到明天下午，如果方案还没发出来你再提醒我。
assistant: 好，我先不打扰，等到明天下午如果还没收到，我再主动跟进。
```

## 9. Clean conversation close

- Scenario: `showcase_polite_close`
- Language: `en`
- Local hour / weekday: `18:00`, `4`
- Description: The thread is finished and should not trigger a new proactive turn.
- Expected target: `suppress`
- Model output: `suppress`
- Model action: `suppress`
- Predicted bucket: `suppress`

```text
user: Perfect, that's all.
assistant: Glad it helped. I'll leave it here.
```

## 10. Same-day after-lunch reminder

- Scenario: `showcase_after_lunch`
- Language: `en`
- Local hour / weekday: `9:00`, `2`
- Description: An intra-day reminder should land at the requested daypart, not immediately.
- Expected target: `18000s (5.0h)`
- Model output: `18000s (5.0h)`
- Model action: `follow_up`
- Predicted bucket: `4h-6h`

```text
user: If I still have not finished the PRD by after lunch, follow up.
assistant: Okay. I'll check back after lunch if it is still pending.
```

