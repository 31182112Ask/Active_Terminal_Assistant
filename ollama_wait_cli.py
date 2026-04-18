from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from wait_model import ConversationExample, humanize_wait, load_model_bundle, predict_wait_time, resolve_device
except ModuleNotFoundError as exc:
    missing_name = getattr(exc, "name", None) or str(exc)
    raise SystemExit(
        "Missing dependency while importing wait_model "
        f"({missing_name}). Run this CLI with the project environment:\n"
        r"  .\.ashley\Scripts\python.exe ollama_wait_cli.py"
    ) from exc


DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_BUNDLE = Path("artifacts") / "wait_policy_bundle.pt"
THINK_TOKEN = "<|think|>"


def pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])

def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(host: str, port: int, timeout_seconds: float, process: subprocess.Popen[str]) -> None:
    deadline = time.time() + timeout_seconds
    url = f"http://{host}:{port}/api/tags"
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ollama serve exited early with code {process.returncode}")
        try:
            get_json(url, timeout=2.0)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"ollama serve did not become ready on {host}:{port}: {last_error}")


def normalize_system_prompt(system_prompt: str, think_enabled: bool) -> str:
    prompt = system_prompt.strip()
    if not prompt:
        return ""
    if think_enabled:
        return prompt
    return prompt.replace(THINK_TOKEN, "").strip()


def spawn_ollama_server(host: str, port: int, timeout_seconds: float) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{host}:{port}"
    process = subprocess.Popen(
        ["ollama", "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        text=True,
    )
    wait_for_server(host, port, timeout_seconds, process)
    print(f"[ollama] server pid {process.pid} on http://{host}:{port}", flush=True)
    return process


def kill_process_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            [
                "powershell",
                "-Command",
                f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($p) {{ Stop-Process -Id {pid} -Force }}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def find_listening_pids(port: int) -> list[int]:
    if os.name != "nt":
        return []
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    pids: set[int] = set()
    suffix = f":{port}"
    for line in result.stdout.splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_address = parts[1]
        pid_text = parts[-1]
        if local_address.endswith(suffix) and pid_text.isdigit():
            pids.add(int(pid_text))
    return sorted(pids)


def ensure_port_closed(port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        listening_pids = find_listening_pids(port)
        if not listening_pids:
            return True
        for pid in listening_pids:
            kill_process_tree(pid)
        time.sleep(0.5)
    return False


def build_prediction_example(messages: list[dict[str, str]]) -> ConversationExample:
    turns = [
        {"role": message["role"], "text": message["content"]}
        for message in messages
        if message["role"] in {"user", "assistant"}
    ]
    now = datetime.now().astimezone()
    return ConversationExample(
        turns=turns,
        hour_local=now.hour,
        weekday=now.weekday(),
    )


def print_wait_policy(prediction: dict[str, Any]) -> None:
    if prediction["suppress"]:
        print("wait_policy: suppress proactive follow-up", flush=True)
        return
    wait_seconds = int(round(float(prediction["wait_seconds"])))
    print(
        f"wait_policy: if the user stays silent, next proactive turn in {wait_seconds}s ({humanize_wait(wait_seconds)})",
        flush=True,
    )


def chat_once(
    host: str,
    port: int,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    think_enabled: bool,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "think": think_enabled,
        "stream": False,
    }
    response = post_json(f"http://{host}:{port}/api/chat", payload, timeout=timeout_seconds)
    message = response.get("message", {})
    content = message.get("content", "").strip()
    if not content:
        raise RuntimeError("Empty response from Ollama chat API")
    return content


def run_cli(args: argparse.Namespace) -> int:
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing wait-policy bundle: {bundle_path}")

    prediction_device = resolve_device(args.device)
    model, featurizer, metadata = load_model_bundle(str(bundle_path), device=prediction_device)
    suppress_threshold = float(metadata.get("suppress_threshold", 0.42))

    host = args.host
    port = args.port or pick_free_port(host)
    server_process: subprocess.Popen[str] | None = None
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": normalize_system_prompt(args.system_prompt, args.think)})

    try:
        server_process = spawn_ollama_server(host, port, args.startup_timeout)
        print(f"model: {args.model}", flush=True)
        print(f"wait_model_device: {prediction_device}", flush=True)
        print(f"thinking: {'enabled' if args.think else 'disabled'}", flush=True)
        print("type /exit to quit", flush=True)

        while True:
            try:
                user_text = input("you> ").strip()
            except EOFError:
                print("", flush=True)
                break
            except KeyboardInterrupt:
                print("\n[cli] interrupted", flush=True)
                break

            if not user_text:
                continue
            if user_text.lower() in {"/exit", "/quit"}:
                break

            messages.append({"role": "user", "content": user_text})
            assistant_text = chat_once(
                host,
                port,
                args.model,
                messages,
                timeout_seconds=args.request_timeout,
                think_enabled=args.think,
            )
            messages.append({"role": "assistant", "content": assistant_text})
            print(f"assistant> {assistant_text}", flush=True)

            example = build_prediction_example(messages)
            prediction = predict_wait_time(
                model,
                featurizer,
                example,
                device=prediction_device,
                suppress_threshold=suppress_threshold,
            )
            print_wait_policy(prediction)
    finally:
        if server_process is not None:
            print(f"[ollama] stopping server pid {server_process.pid}", flush=True)
            kill_process_tree(server_process.pid)
            try:
                server_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                kill_process_tree(server_process.pid)
            fully_released = ensure_port_closed(port, timeout_seconds=20.0)
            if fully_released:
                print("[ollama] server stopped", flush=True)
            else:
                print(f"[ollama] warning: port {port} was not confirmed closed before exit", flush=True)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Ollama multi-turn CLI with wait-policy prediction.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="Dedicated local port for the spawned ollama serve process.")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--device", default="auto", help="Prediction model device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--think", action="store_true", help="Enable Ollama thinking mode. Disabled by default.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
