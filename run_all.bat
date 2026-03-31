@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo [setup] creating virtual environment...
  py -3.12 -m venv .venv || goto :error
)

echo [setup] installing project dependencies...
call .venv\Scripts\python.exe -m pip install -e .[dev] || goto :error

echo [check] preparing Ollama runtime...
call .venv\Scripts\python.exe scripts\prepare_runtime.py || goto :error

echo [check] validating Ollama connectivity...
call .venv\Scripts\python.exe scripts\check_ollama.py || goto :error

echo [run] starting Active Terminal Assistant GUI...
call .venv\Scripts\python.exe scripts\run_gui.py
goto :eof

:error
echo.
echo [error] startup failed. See output above.
exit /b 1
