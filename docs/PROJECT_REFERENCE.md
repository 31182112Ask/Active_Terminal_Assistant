# Project Reference

## Runtime Model Split

- Dialogue model: `qwen3:14b`
- Decision model: `qwen3:1.7b`

## Core Flow

1. 用户输入到达
2. `agere` job/handler 进入 `user_input_received`
3. 对话模型生成普通回复
4. 状态转入 `idle_waiting`
5. `ProactiveAgent` 的 `WakeUpScheduler` 计算下次唤醒时间
6. 唤醒后进入 `speak_wait_decision`
7. 规则门控 + 决策模型判断 `WAIT` / `SPEAK`
8. 若 `SPEAK`，对话模型生成主动补充内容
9. 用户若在此时发新消息，会取消主动输出并优先处理新输入

## 关键模块

- [`app/workflow/orchestrator.py`](/c:/Users/33835/Erika_project_v3/app/workflow/orchestrator.py)
  agere 工作流、事件调度、生成取消、中断处理
- [`app/services/proactive.py`](/c:/Users/33835/Erika_project_v3/app/services/proactive.py)
  ProactiveAgent `WakeUpScheduler` 桥接与睡眠策略
- [`app/services/rules.py`](/c:/Users/33835/Erika_project_v3/app/services/rules.py)
  冷却、重复抑制、用户 disengagement、连续主动轮次控制
- [`app/adapters/dialogue.py`](/c:/Users/33835/Erika_project_v3/app/adapters/dialogue.py)
  对话模型流式生成
- [`app/adapters/decision.py`](/c:/Users/33835/Erika_project_v3/app/adapters/decision.py)
  决策模型 JSON 解析
- [`app/cli/rendering.py`](/c:/Users/33835/Erika_project_v3/app/cli/rendering.py)
  Rich CLI 面板渲染

## 配置优先级

1. 代码默认值
2. `agent.toml`
3. 环境变量覆盖

## 日志

日志输出到：

- 控制台
- `logs/session-<session_id>.log`

## 一键启动链路

[`run_all.bat`](/c:/Users/33835/Erika_project_v3/run_all.bat) 会依次调用：

1. `pip install -e .[dev]`
2. [`scripts/prepare_runtime.py`](/c:/Users/33835/Erika_project_v3/scripts/prepare_runtime.py)
3. [`scripts/check_ollama.py`](/c:/Users/33835/Erika_project_v3/scripts/check_ollama.py)
4. [`scripts/run_cli.py`](/c:/Users/33835/Erika_project_v3/scripts/run_cli.py)
