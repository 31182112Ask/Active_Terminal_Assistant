from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path("ollama_wait_cli.py")


def pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def listening_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    pids: list[int] = []
    suffix = f":{port}"
    for line in result.stdout.splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[1].endswith(suffix) and parts[-1].isdigit():
            pids.append(int(parts[-1]))
    return sorted(set(pids))


def wait_for_port_close(port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not listening_pids(port):
            return True
        time.sleep(0.5)
    return False


def pid_exists_windows(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    line = result.stdout.strip()
    if not line:
        return False
    if "No tasks are running" in line:
        return False
    return line.startswith('"')


def main() -> int:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(SCRIPT_PATH)

    host = "127.0.0.1"
    port = pick_free_port(host)
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        "gemma4:e4b",
    ]
    scripted_input = "Reply with exactly OK.\n/exit\n"
    result = subprocess.run(
        command,
        input=scripted_input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        cwd=os.getcwd(),
        check=False,
    )

    stdout = result.stdout
    stderr = result.stderr
    print(stdout)
    if stderr.strip():
        print(stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"CLI exited with code {result.returncode}")
    if "assistant>" not in stdout:
        raise RuntimeError("CLI did not print an assistant response")
    if "wait_policy:" not in stdout:
        raise RuntimeError("CLI did not print wait-policy output")

    match = re.search(r"server pid (\d+)", stdout)
    if not match:
        raise RuntimeError("Could not parse ollama server pid from CLI output")
    server_pid = int(match.group(1))

    time.sleep(2.0)
    if pid_exists_windows(server_pid):
        raise RuntimeError(f"Ollama server process still exists after exit: pid={server_pid}")
    lingering = listening_pids(port)
    if not wait_for_port_close(port, timeout_seconds=20.0):
        raise RuntimeError(f"Ollama server port is still listening after exit: {host}:{port}, pids={lingering}")

    print(f"smoke_test: passed, pid={server_pid}, port={port} fully released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
