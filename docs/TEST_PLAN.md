# Test Plan

## Automated

运行：

```powershell
.venv\Scripts\python.exe -m pytest
```

当前测试集包含：

- `tests/unit/test_decision_parser.py`
- `tests/unit/test_rules.py`
- `tests/unit/test_commands.py`
- `tests/unit/test_gui_viewmodel.py`
- `tests/integration/test_workflow_runtime.py`
- `tests/e2e/test_scripted_session.py`

## Manual Runbook

### 1. 启动 GUI

```powershell
cmd /c run_all.bat
```

### 2. 正常聊天

- 输入一个普通问题
- 确认助手正常回复
- 确认状态转入 `idle_waiting`

### 3. 短延续窗口

- 输入一个适合自然续一句的话题
- 观察 developer panel 中的 `Window`
- 确认系统在较短窗口下可能补一句更轻的延续

### 4. 长主动窗口

- 让会话停一会儿
- 使用 `/poke` 或等待自然唤醒
- 确认系统在更保守的窗口里判断 `WAIT` 或 `SPEAK`

### 5. 用户打断

- 使用 `/speak-now`
- 在主动输出时立刻输入新消息
- 确认主动输出被取消
- 确认新消息优先得到处理

### 6. GUI 控制

- 开关 `Proactive mode`
- 开关 `Developer panel`
- 点击 `Poke`
- 点击 `Speak Now`
- 点击 `Reset`
- 点击 `Export Transcript`

## External Runtime Requirement

本地 Ollama 至少需要：

```powershell
ollama pull qwen3:14b
ollama pull qwen3:1.7b
```
