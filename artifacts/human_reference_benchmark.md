# Human Reference Benchmark

This report compares the current wait-policy model against a manually labeled human-reference set.

Important: this is a single-rater human-reference benchmark, not production user telemetry.

## Aggregate Metrics

- Total cases: `16`
- Follow-up cases: `12`
- Suppress cases: `4`
- Overall within-tolerance rate: `0.8750`
- Follow-up within-tolerance rate: `0.8333`
- Suppress agreement rate: `1.0000`
- Follow-up detection rate: `1.0000`
- Follow-up MAE: `1452.0s`
- Follow-up median absolute error: `0.0s`
- Follow-up Spearman rank correlation: `0.8998`

## Visualization

![Human reference comparison](human_reference_comparison.svg)

## Case Table

| # | Case | Category | Human | Model | Tolerance | Within tolerance |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Explicit 30-second reminder | `explicit_short` | `30s` | `30s` | `10s` | `True` |
| 2 | Urgent incident with explicit 45-second check | `urgent_short` | `45s` | `45s` | `15s` | `True` |
| 3 | Blocked clarification after 10 minutes | `blocked_followup` | `10m` | `3.0h` | `60s` | `False` |
| 4 | Driving home for 30 minutes | `explicit_medium` | `30m` | `30m` | `3m` | `True` |
| 5 | After-lunch daypart anchor | `daypart_anchor` | `4.0h` | `4.0h` | `30m` | `True` |
| 6 | Tomorrow-morning follow-up | `daypart_anchor` | `15.0h` | `15.0h` | `60m` | `True` |
| 7 | Tomorrow-evening vendor check | `daypart_anchor` | `35.0h` | `35.0h` | `3.0h` | `True` |
| 8 | Attachment reminder in 20 minutes | `explicit_medium` | `20m` | `20m` | `2m` | `True` |
| 9 | Two-hour review window | `explicit_long` | `2.0h` | `2.0h` | `30m` | `True` |
| 10 | Busy all afternoon | `soft_busy_daypart` | `4.0h` | `4.0h` | `60m` | `True` |
| 11 | Soft deep-work request | `soft_busy_open` | `2.0h` | `48s` | `60m` | `False` |
| 12 | Supplier follow-up in one day | `async_long` | `24.0h` | `24.0h` | `2.0h` | `True` |
| 13 | Need space | `suppress_space` | `suppress` | `suppress` | `n/a` | `True` |
| 14 | Do not ping again tonight | `suppress_dnd` | `suppress` | `suppress` | `n/a` | `True` |
| 15 | Clean conversation close | `suppress_close` | `suppress` | `suppress` | `n/a` | `True` |
| 16 | No follow-up unless asked | `suppress_dnd` | `suppress` | `suppress` | `n/a` | `True` |

## Dialogue Contexts

### 1. Explicit 30-second reminder

- Category: `explicit_short`
- Human reference: `30s`
- Model prediction: `30s`
- Rationale: The user sets an exact second-level reminder, so a human would wait exactly that long.

```text
user: If I go quiet, ping me again in 30 seconds.
assistant: Okay. If you stay silent, I will ping you again in 30 seconds.
```

### 2. Urgent incident with explicit 45-second check

- Category: `urgent_short`
- Human reference: `45s`
- Model prediction: `45s`
- Rationale: This is an urgent production context and the assistant already committed to a short check-back window.

```text
user: Prod is still down after the last restart.
assistant: Keep watching the queue. If I still do not hear from you, I will check back in 45 seconds.
```

### 3. Blocked clarification after 10 minutes

- Category: `blocked_followup`
- Human reference: `10m`
- Model prediction: `3.0h`
- Rationale: A human operator would keep the thread alive because progress is blocked, but without becoming too aggressive.

```text
user: I still owe you the missing field name.
assistant: If I do not hear back, I will ask again in 10 minutes so we can unblock the draft.
```

### 4. Driving home for 30 minutes

- Category: `explicit_medium`
- Human reference: `30m`
- Model prediction: `30m`
- Rationale: The user gives a concrete availability window, so the next proactive turn should wait until that window closes.

```text
user: I am driving home for 30 minutes. If the invoice is still pending, ping me after that.
assistant: Understood. I will leave you alone for 30 minutes and then check back if it is still pending.
```

### 5. After-lunch daypart anchor

- Category: `daypart_anchor`
- Human reference: `4.0h`
- Model prediction: `4.0h`
- Rationale: A human would not interrupt before the requested daypart arrives.

```text
user: If the PRD is still missing after lunch, follow up.
assistant: Okay. I will check back after lunch if it is still missing.
```

### 6. Tomorrow-morning follow-up

- Category: `daypart_anchor`
- Human reference: `15.0h`
- Model prediction: `15.0h`
- Rationale: The user gives an explicit next-morning boundary, so the human reference waits until that point.

```text
user: If I still have not sent the deck by tomorrow morning, follow up.
assistant: Understood. I will check back tomorrow morning if it is still missing.
```

### 7. Tomorrow-evening vendor check

- Category: `daypart_anchor`
- Human reference: `35.0h`
- Model prediction: `35.0h`
- Rationale: The user names a specific future daypart, so a human would wait until that deadline instead of checking early.

```text
user: If the vendor still has not replied by tomorrow evening, follow up.
assistant: Okay. I will check again tomorrow evening if there is still no reply.
```

### 8. Attachment reminder in 20 minutes

- Category: `explicit_medium`
- Human reference: `20m`
- Model prediction: `20m`
- Rationale: The delay is explicit and short enough that a human would follow it closely.

```text
user: I still need to attach the screenshot. If I disappear, ask again in 20 minutes.
assistant: Got it. I will ask again in 20 minutes if I have not heard from you.
```

### 9. Two-hour review window

- Category: `explicit_long`
- Human reference: `2.0h`
- Model prediction: `2.0h`
- Rationale: The user gives a concrete two-hour review window, so a human would avoid checking back early.

```text
user: I am heads-down on this review for 2 hours.
assistant: Okay. I will circle back in 2 hours if it is still pending.
```

### 10. Busy all afternoon

- Category: `soft_busy_daypart`
- Human reference: `4.0h`
- Model prediction: `4.0h`
- Rationale: A human reading this context would usually defer until the afternoon meeting block is over.

```text
user: I am in back-to-back meetings this afternoon.
assistant: Understood. I will follow up later this afternoon if the decision is still blocked.
```

### 11. Soft deep-work request

- Category: `soft_busy_open`
- Human reference: `2.0h`
- Model prediction: `48s`
- Rationale: Even without a numeric delay, a human would usually infer that a couple of hours of space is appropriate.

```text
user: I need a couple of hours of heads-down time.
assistant: No problem. I will give you room and circle back later if it is still pending.
```

### 12. Supplier follow-up in one day

- Category: `async_long`
- Human reference: `24.0h`
- Model prediction: `24.0h`
- Rationale: External async dependencies usually deserve a long wait before a proactive nudge.

```text
user: The supplier still has not replied.
assistant: If it is still quiet in 1 day, I will follow up again.
```

### 13. Need space

- Category: `suppress_space`
- Human reference: `suppress`
- Model prediction: `suppress`
- Rationale: A human should not proactively interrupt after the user explicitly asks for space.

```text
user: I am overwhelmed and need some space.
assistant: Understood. I will wait for you to restart the conversation.
```

### 14. Do not ping again tonight

- Category: `suppress_dnd`
- Human reference: `suppress`
- Model prediction: `suppress`
- Rationale: The user gives an explicit do-not-contact instruction for the rest of the night.

```text
user: I am going to sleep. Do not ping me again tonight.
assistant: Understood. I will leave it here for tonight.
```

### 15. Clean conversation close

- Category: `suppress_close`
- Human reference: `suppress`
- Model prediction: `suppress`
- Rationale: The thread is finished and should not trigger another proactive turn.

```text
user: Perfect, that is all.
assistant: Glad it helped. I will leave it here.
```

### 16. No follow-up unless asked

- Category: `suppress_dnd`
- Human reference: `suppress`
- Model prediction: `suppress`
- Rationale: The user explicitly forbids proactive outreach unless they restart the thread.

```text
user: Thanks. No need to follow up unless I ask.
assistant: Okay. I will stay quiet unless you come back.
```

