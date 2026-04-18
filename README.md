# Active Terminal Assistant

Active Terminal Assistant is a local Ollama chat CLI with a learned wait-policy model for proactive follow-up timing.

This repository contains:

- `ollama_wait_cli.py`: minimal multi-turn CLI built on `ollama serve`
- `wait_model.py`: lightweight wait-policy model, feature extraction, bundle loading, and prediction
- `run_training.py`: synthetic data generation, training, benchmark, and artifact export
- `main.ipynb`: notebook entry point for training and export
- `artifacts/`: trained weights, benchmark metrics, benchmark report, and context showcase
- `smoke_test_ollama_wait_cli.py`: end-to-end smoke test for CLI startup, chat, wait prediction, and process shutdown

## What The Model Does

The prediction model solves this task:

Given the current dialogue context after the assistant has already spoken, and assuming the user remains silent from now on, predict when the assistant should proactively speak again.

The model does not use an LLM for this decision. It is a lightweight neural model trained on structured dialogue data with explicit time-awareness and suppression logic.

## Current Stack

- LLM runtime: Ollama
- Chat model: `gemma4:e4b`
- Wait-policy model: local PyTorch model bundle in `artifacts/wait_policy_bundle.pt`
- CLI mode: multi-turn terminal chat
- Thinking mode: disabled by default through Ollama `think: false`

## Project Layout

```text
.
├─ artifacts/
│  ├─ benchmark_report.md
│  ├─ context_showcase.md
│  ├─ metrics.json
│  ├─ prediction_preview.json
│  ├─ wait_policy_bundle.pt
│  └─ wait_policy_state.pt
├─ main.ipynb
├─ ollama_wait_cli.py
├─ run_training.py
├─ smoke_test_ollama_wait_cli.py
└─ wait_model.py
```

## Requirements

- Windows PowerShell
- Python 3.12
- Ollama installed locally
- `gemma4:e4b` already pulled in Ollama
- NVIDIA GPU optional
  - the wait-policy model will use CUDA automatically if PyTorch with CUDA is installed

## Install

Create and activate a virtual environment if needed:

```powershell
python -m venv .ashley
.\.ashley\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

For CUDA on Windows with NVIDIA GPU, install the matching PyTorch build. Example for CUDA 12.8:

```powershell
python -m pip install --upgrade --force-reinstall torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

## Run The CLI

Start the local chat CLI:

```powershell
python ollama_wait_cli.py
```

What it does:

- starts a dedicated `ollama serve` process on a local port
- chats with `gemma4:e4b`
- runs the wait-policy model after each assistant reply
- prints the recommended proactive wait time if the user stays silent
- kills the spawned Ollama process on exit

Optional flags:

```powershell
python ollama_wait_cli.py --device cuda
python ollama_wait_cli.py --think
python ollama_wait_cli.py --model gemma4:e4b
```

Notes:

- `--think` enables Ollama thinking mode. Default is off.
- if you omit `--device`, the wait-policy model uses `cuda` when available, otherwise `cpu`

## Train Or Rebuild The Model

Run the training pipeline:

```powershell
python run_training.py
```

Or open and execute:

```text
main.ipynb
```

Training outputs are written into `artifacts/`.

## Benchmark Outputs

Main benchmark artifacts:

- `artifacts/metrics.json`
- `artifacts/benchmark_report.md`
- `artifacts/context_showcase.md`

These files record:

- acceptance metrics
- practical-score style evaluation
- stress and lexical checks
- representative context-level prediction examples

## Test

Run the end-to-end smoke test:

```powershell
python smoke_test_ollama_wait_cli.py
```

The smoke test verifies:

- Ollama server startup
- one real chat roundtrip
- wait-policy inference
- clean process shutdown
- port release after exit

## Notes

- The wait-policy benchmark is currently based mainly on synthetic data.
- The benchmark is useful for iteration, but real annotated dialogue data would be a stronger validation set.
- Model artifacts are included in this repository so the CLI can run directly without retraining.
