# Local Proactive CLI Agent

本项目是一个本地运行的主动式终端助手，满足 `AGENTS.md` 中的核心要求：

- 全本地推理，基于 Ollama
- 双模型架构
- 对话模型：`qwen3:14b`
- 决策模型：`qwen3:1.7b`
- `agere` 负责显式工作流编排
- `ProactiveAgent` 负责主动唤醒与睡眠节奏
- 自定义 CLI 面板，支持多轮对话、状态观测、slash commands、主动补充发言

## 已实现能力

- 多轮文本对话
- 双模型分工
- 主动决策与主动补充回复
- 保守规则门控
- 用户打断主动输出
- 可观测 CLI 面板
- slash commands
- 一键启动脚本
- 单元测试、集成测试、脚本化 e2e 测试

## 目录结构

```text
app/
  cli/          # Rich CLI、输入控制、slash commands
  workflow/     # agere 工作流编排
  adapters/     # Ollama 对话/决策适配器
  prompts/      # 提示词模板
  state/        # 会话状态与快照
  services/     # 规则门控、上下文压缩、ProactiveAgent 桥接
  config/       # TOML + 环境变量配置
  utils/        # 日志与文本工具
tests/
  unit/
  integration/
  e2e/
scripts/
  run_cli.py
  check_ollama.py
  prepare_runtime.py
run_all.bat
agent.toml
```

## 环境要求

- Windows + PowerShell
- Python 3.12+
- 本地安装 Ollama，并可执行 `ollama`

## 一键启动

项目根目录下直接运行：

```powershell
cmd /c run_all.bat
```

`run_all.bat` 会做这些事：

1. 创建虚拟环境（如缺失）
2. 安装项目依赖
3. 尝试拉起 `ollama serve`
4. 自动拉取缺失模型
5. 校验模型是否可用
6. 启动 CLI

## 手动启动

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe scripts\prepare_runtime.py
.venv\Scripts\python.exe scripts\run_cli.py
```

## 常用命令

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
- `/poke`
- `/speak-now`

## 配置

默认配置文件是根目录的 [`agent.toml`](/c:/Users/33835/Erika_project_v3/agent.toml)。

可以配置：

- Ollama 地址
- 对话/决策模型
- 主动模式开关
- 冷却时间
- 最大连续主动轮次
- CLI 刷新频率
- 日志级别

也支持部分环境变量覆盖，例如：

- `LPA_DIALOGUE_MODEL`
- `LPA_DECISION_MODEL`
- `LPA_OLLAMA_BASE_URL`
- `LPA_PROACTIVE_ENABLED`
- `LPA_DEBUG`

## 测试

```powershell
.venv\Scripts\python.exe -m pytest
```

当前测试覆盖：

- 决策输出解析
- 规则门控与重复抑制
- slash command 处理
- 普通回复路径
- 主动补充路径
- 用户打断主动输出
- 错误恢复

## 运行前检查

单独检查 Ollama 与模型：

```powershell
.venv\Scripts\python.exe scripts\check_ollama.py
```

如果缺失模型，可手动拉取：

```powershell
ollama pull qwen3:14b
ollama pull qwen3:1.7b
```

## 文档

- [项目参考](/c:/Users/33835/Erika_project_v3/docs/PROJECT_REFERENCE.md)
- [测试与手动验证](/c:/Users/33835/Erika_project_v3/docs/TEST_PLAN.md)
