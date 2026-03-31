PYTHON ?= .venv/Scripts/python.exe

.PHONY: setup run test check

setup:
	$(PYTHON) -m pip install -e .[dev]

run:
	$(PYTHON) scripts/run_cli.py

test:
	$(PYTHON) -m pytest

check:
	$(PYTHON) scripts/check_ollama.py

