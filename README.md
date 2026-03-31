# Active Terminal Assistant

这是一个本地运行的主动式对话助手，当前版本已经从 CLI 原型演进到 GUI 主界面。

核心特性：

- 全本地运行，基于 Ollama
- 双模型架构
- 对话模型：`qwen3:14b`
- 判定模型：`qwen3:1.7b`
- `agere` 负责显式工作流调度
- `ProactiveAgent` 负责主动唤醒与睡眠节奏
- 默认 GUI 聊天界面，CLI 作为开发兜底保留
- 支持多轮对话、主动补充发言、用户打断、开发者面板

## 当前重点

这版项目的重点不再是“可观测 CLI”，而是：

- 更自然的 GUI 聊天体验
- 更短、更口语化的默认回复
- 短延续窗口和长主动窗口两层节奏
- 更轻量的判定模型职责
- 保留必要护栏，但减少外部规则对交流内容的主导

## 目录结构

```text
app/
  adapters/     # Ollama 对话/判定适配器
  cli/          # CLI 调试入口和 slash command
  config/       # 配置加载
  gui/          # Tk GUI 主界面与显示模型
  prompts/      # 提示词
  services/     # 主动调度、规则护栏、上下文压缩
  state/        # 会话状态
  utils/        # 日志与文本工具
  workflow/     # agere 工作流编排
tests/
  unit/
  integration/
  e2e/
scripts/
  run_gui.py
  run_cli.py
  prepare_runtime.py
  check_ollama.py
```

## 一键启动

```powershell
cmd /c run_all.bat
```

这会自动：

1. 创建虚拟环境
2. 安装依赖
3. 准备 Ollama 运行时
4. 检查本地模型
5. 启动 GUI

## 手动启动 GUI

```powershell
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe scripts\prepare_runtime.py
.venv\Scripts\python.exe scripts\run_gui.py
```

## CLI 调试入口

如果你仍然想用旧的终端调试入口：

```powershell
.venv\Scripts\python.exe scripts\run_cli.py
```

## GUI 功能

- 聊天主区
- 多行输入框
- Send / Cancel Output / Clear Chat
- Proactive mode 开关
- Debug mode 开关
- Developer panel 折叠区
- Poke / Speak Now / Reset
- Transcript 导出

## Slash Commands

GUI 输入框和 CLI 都支持：

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

默认配置文件是 [`agent.toml`](/c:/Users/33835/Erika_project_v3/agent.toml)。

现在主要可调项包括：

- 模型名称与 Ollama 地址
- GUI 窗口尺寸与轮询间隔
- 主动模式开关
- 短延续窗口 / 长主动窗口
- 冷却时间
- 重复抑制阈值
- 日志级别

## 测试

```powershell
.venv\Scripts\python.exe -m pytest
```

当前自动化覆盖：

- 判定 JSON 解析
- 规则护栏
- GUI 状态显示模型
- slash command 处理
- 正常回复路径
- 主动补充路径
- 用户打断主动输出
- 错误恢复

## Ollama 依赖

本地至少需要：

```powershell
ollama pull qwen3:14b
ollama pull qwen3:1.7b
```

单独检查运行时：

```powershell
.venv\Scripts\python.exe scripts\check_ollama.py
```

## 文档

- [项目参考](/c:/Users/33835/Erika_project_v3/docs/PROJECT_REFERENCE.md)
- [测试与手动验证](/c:/Users/33835/Erika_project_v3/docs/TEST_PLAN.md)
