#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class GpuInfo:
    index: int
    memory_free_mib: int
    utilization_gpu: int


MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/public_models/Qwen/Qwen2.5-72B-Instruct")
VLLM_BIN = os.environ.get("VLLM_BIN", "/mnt/zelin/miniconda3/envs/judge_env/bin/vllm")
PORT = int(os.environ.get("PORT", "23333"))
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen72b")
GPU_MEMORY_UTILIZATION = os.environ.get("GPU_MEMORY_UTILIZATION", "0.83")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "4096")
MAX_NUM_SEQS = os.environ.get("MAX_NUM_SEQS", "2")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "60"))
MIN_FREE_MIB = int(os.environ.get("MIN_FREE_MIB", "42000"))
MAX_UTIL_PERCENT = int(os.environ.get("MAX_UTIL_PERCENT", "10"))

child_process: subprocess.Popen[str] | None = None


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def service_is_ready() -> bool:
    url = f"http://127.0.0.1:{PORT}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False

    for model in payload.get("data", []):
        if model.get("id") == SERVED_MODEL_NAME:
            return True
    return False


def query_gpus() -> list[GpuInfo]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    gpus = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        gpu_index, memory_free, utilization = [part.strip() for part in line.split(",")]
        gpus.append(
            GpuInfo(
                index=int(gpu_index),
                memory_free_mib=int(memory_free),
                utilization_gpu=int(utilization),
            )
        )
    return gpus


def pick_gpus(gpus: list[GpuInfo]) -> list[GpuInfo]:
    candidates = [
        gpu
        for gpu in gpus
        if gpu.memory_free_mib >= MIN_FREE_MIB and gpu.utilization_gpu <= MAX_UTIL_PERCENT
    ]
    candidates.sort(key=lambda gpu: (-gpu.memory_free_mib, gpu.index))
    return candidates[:TP_SIZE]


def start_service(selected_gpus: list[GpuInfo]) -> subprocess.Popen[str]:
    cuda_visible_devices = ",".join(str(gpu.index) for gpu in selected_gpus)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    command = [
        VLLM_BIN,
        "serve",
        "--model",
        MODEL_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        str(TP_SIZE),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--disable-log-requests",
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--max-model-len",
        MAX_MODEL_LEN,
        "--max_num_seqs",
        MAX_NUM_SEQS,
    ]

    log(
        "Starting 72B vLLM with GPUs "
        f"{cuda_visible_devices} and command: {' '.join(shlex.quote(part) for part in command)}"
    )
    return subprocess.Popen(command, env=env, text=True)


def handle_signal(signum: int, _frame: object) -> None:
    global child_process
    log(f"Received signal {signum}, shutting down watcher.")
    if child_process is not None and child_process.poll() is None:
        child_process.terminate()
        try:
            child_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child_process.kill()
    raise SystemExit(0)


def main() -> None:
    global child_process

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log(
        "Watcher started. Waiting for "
        f"{TP_SIZE} GPUs with memory_free>={MIN_FREE_MIB} MiB and utilization<={MAX_UTIL_PERCENT}%."
    )

    while True:
        if service_is_ready():
            log(f"Service {SERVED_MODEL_NAME} is already ready on port {PORT}.")
            time.sleep(POLL_SECONDS)
            continue

        try:
            gpus = query_gpus()
        except subprocess.CalledProcessError as exc:
            log(f"Failed to query GPUs: {exc}. Retrying in {POLL_SECONDS}s.")
            time.sleep(POLL_SECONDS)
            continue

        selected_gpus = pick_gpus(gpus)
        if len(selected_gpus) < TP_SIZE:
            gpu_summary = ", ".join(
                f"gpu{gpu.index}:free={gpu.memory_free_mib}MiB,util={gpu.utilization_gpu}%"
                for gpu in gpus
            )
            log(f"Not enough free GPUs yet. Current state: {gpu_summary}")
            time.sleep(POLL_SECONDS)
            continue

        child_process = start_service(selected_gpus)
        return_code = child_process.wait()
        log(f"72B vLLM process exited with code {return_code}. Returning to watch mode.")
        child_process = None
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
