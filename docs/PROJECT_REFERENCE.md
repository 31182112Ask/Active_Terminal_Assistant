# Project Reference

## Runtime Model Split

- Dialogue model: `qwen3:14b`
- Decision model: `qwen3:1.7b`

## Main Product Shape

- 默认入口是 GUI
- CLI 保留为开发和调试兜底入口
- `agere` 负责事件式工作流
- `ProactiveAgent` 负责主动唤醒节奏

## Core Flow

1. 用户在 GUI 或 CLI 中输入消息
2. `WorkflowOrchestrator` 把事件放进 `agere` job/handler
3. 对话模型生成正常回复
4. 系统转入 `idle_waiting`
5. `ProactiveSchedulerBridge` 使用 `WakeUpScheduler` 计算下次唤醒
6. 唤醒后进入 `speak_wait_decision`
7. 判定模型输出极简 JSON
8. 若结果是 `SPEAK`，主模型生成一个更短、更轻的主动补充
9. 用户若在此期间输入新消息，主动输出会被取消

## Short vs Long Proactive Windows

系统现在有两层主动节奏：

- `short`
  刚回答完后的一小段自然延续窗口，更适合补一句
- `long`
  更保守的后续主动窗口，更适合稍晚一点再轻补一条

相关状态会出现在：

- GUI 状态区
- GUI developer panel
- `SessionSnapshot`

## Key Modules

- [`app/workflow/orchestrator.py`](/c:/Users/33835/Erika_project_v3/app/workflow/orchestrator.py)
  核心引擎，供 GUI 与 CLI 复用
- [`app/gui/main.py`](/c:/Users/33835/Erika_project_v3/app/gui/main.py)
  Tk GUI 主界面
- [`app/gui/viewmodel.py`](/c:/Users/33835/Erika_project_v3/app/gui/viewmodel.py)
  GUI 展示层视图模型
- [`app/services/proactive.py`](/c:/Users/33835/Erika_project_v3/app/services/proactive.py)
  ProactiveAgent 调度桥接
- [`app/services/rules.py`](/c:/Users/33835/Erika_project_v3/app/services/rules.py)
  必要的最小护栏
- [`app/adapters/dialogue.py`](/c:/Users/33835/Erika_project_v3/app/adapters/dialogue.py)
  主模型流式生成
- [`app/adapters/decision.py`](/c:/Users/33835/Erika_project_v3/app/adapters/decision.py)
  判定模型 JSON 解析

## Startup

[`run_all.bat`](/c:/Users/33835/Erika_project_v3/run_all.bat) 会依次运行：

1. `pip install -e .[dev]`
2. [`scripts/prepare_runtime.py`](/c:/Users/33835/Erika_project_v3/scripts/prepare_runtime.py)
3. [`scripts/check_ollama.py`](/c:/Users/33835/Erika_project_v3/scripts/check_ollama.py)
4. [`scripts/run_gui.py`](/c:/Users/33835/Erika_project_v3/scripts/run_gui.py)
