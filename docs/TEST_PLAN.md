# Test Plan

## Automated

运行：

```powershell
.venv\Scripts\python.exe -m pytest
```

当前自动化测试包含：

- `tests/unit/test_decision_parser.py`
- `tests/unit/test_rules.py`
- `tests/unit/test_commands.py`
- `tests/integration/test_workflow_runtime.py`
- `tests/e2e/test_scripted_session.py`

## Manual Validation Runbook

### 1. 启动

```powershell
cmd /c run_all.bat
```

### 2. 标准聊天

- 输入一个普通问题
- 确认助手立即回复
- 观察状态进入 `idle_waiting`

### 3. 主动补充

- 输入一个适合继续帮助的话题，例如“帮我做旅行计划”
- 等待主动唤醒，或使用 `/poke`
- 确认决策面板显示 `SPEAK`
- 确认主动消息以 proactive 形式显示

### 4. 主动抑制

- 让助手刚问完一个直接问题
- 不输入新消息，等待一次唤醒
- 确认决策为 `WAIT` 或被规则阻止

### 5. 用户打断

- 使用 `/speak-now`
- 在主动输出进行中立刻输入新消息
- 确认主动输出被取消
- 确认新用户消息优先得到处理

### 6. 运行时控制

- `/proactive off`
- `/proactive on`
- `/debug off`
- `/debug on`
- `/status`
- `/history`
- `/reset`

## Known External Dependency Requirement

本地 Ollama 至少需要以下模型：

- `qwen3:14b`
- `qwen3:1.7b`

如果缺失，可执行：

```powershell
ollama pull qwen3:14b
ollama pull qwen3:1.7b
```
