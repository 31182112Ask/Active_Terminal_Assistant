# Wait-Time Benchmark Report

## Summary

- Acceptance passed: `True`
- Selected trial: `large`
- Best epoch: `8`
- Suppress threshold: `0.42`

## Training Coverage

- Train size: `72000`
- Scenario mix: `{'urgent_incident': 7028, 'accountability_checkin': 7038, 'support_followup': 7746, 'polite_close': 6204, 'scheduled_reminder': 9977, 'async_deliverable': 8283, 'sensitive_space': 5479, 'do_not_disturb': 7019, 'clarification_needed': 6983, 'order_status': 6243}`

## Model Metrics

| Split | Practical | Adaptive-60s | Adaptive-300s | Short MAE(s) | P90 Error(s) | Follow-up F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test_iid | 0.969 | 0.962 | 0.984 | 1.5 | 520.0 | 0.996 |
| test_lexical | 0.975 | 0.965 | 0.976 | 0.0 | 630.2 | 1.000 |
| test_stress | 0.967 | 0.971 | 0.984 | 3.8 | 300.8 | 0.999 |
| overall | 0.970 | 0.966 | 0.982 | 1.8 | 483.7 | 0.998 |

## Keyword Baseline

| Split | Practical | Adaptive-60s | Adaptive-300s | Short MAE(s) | P90 Error(s) | Follow-up F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test_iid | 0.269 | 0.075 | 0.242 | 111792.2 | 842890.6 | 0.869 |
| test_lexical | 0.263 | 0.062 | 0.236 | 97627.2 | 820800.0 | 0.881 |
| test_stress | 0.264 | 0.071 | 0.249 | 156768.0 | 862812.8 | 0.874 |
| overall | 0.265 | 0.069 | 0.243 | 122062.5 | 842167.8 | 0.875 |

## Golden Cases

- Pass rate: `1.000`
- golden_dnd: target `suppress`, predicted `suppress`, pass `True`
- golden_scheduled: target `61200s`, predicted `61200s`, pass `True`
- golden_urgent: target `240s`, predicted `240s`, pass `True`
- golden_close: target `suppress`, predicted `suppress`, pass `True`
- golden_clarify: target `7200s`, predicted `7200s`, pass `True`
- golden_space: target `suppress`, predicted `suppress`, pass `True`
- golden_after_lunch: target `18000s`, predicted `18000s`, pass `True`
- golden_vendor: target `86400s`, predicted `86400s`, pass `True`

## Acceptance Checks

- overall_practical_score: `True`
- overall_followup_f1: `True`
- overall_adaptive_60s_rate: `True`
- overall_adaptive_300s_rate: `True`
- overall_short_mae_seconds: `True`
- overall_short_within_5s_rate: `True`
- lexical_adaptive_60s_rate: `True`
- stress_adaptive_60s_rate: `True`
- stress_false_early_rate: `True`
- golden_pass_rate: `True`
- beats_keyword_baseline: `True`
