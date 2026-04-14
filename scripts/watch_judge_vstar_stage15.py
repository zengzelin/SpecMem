#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a VStar raw folder and judge new files when the judge service is ready.")
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--api_base", default="http://127.0.0.1:23333/v1")
    parser.add_argument("--model_name", default="qwen72b")
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--judge_python", default="/mnt/zelin/miniconda3/envs/judge_env/bin/python3.11")
    parser.add_argument("--judge_script", default="/home/zelin/SpecMem/judge_code/judge_vstar.py")
    return parser.parse_args()


def infer_output_folder(input_folder: str) -> str:
    if "eval_results" in input_folder:
        return input_folder.replace("eval_results", "judge_results", 1)
    return os.path.join(os.path.dirname(input_folder), f"judge_{os.path.basename(os.path.normpath(input_folder))}")


def service_ready(api_base: str, model_name: str) -> bool:
    url = f"{api_base}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    return model_name in payload


def pending_files(input_folder: str, output_folder: str) -> list[str]:
    raw_files = sorted(
        file_name
        for file_name in os.listdir(input_folder)
        if file_name.startswith("vstar") and file_name.endswith(".jsonl")
    )
    pending = []
    for file_name in raw_files:
        judged_name = file_name.replace(".jsonl", "_acc.jsonl")
        judged_path = os.path.join(output_folder, judged_name)
        if not os.path.exists(judged_path) or os.path.getsize(judged_path) == 0:
            pending.append(file_name)
    return pending


def run_judge(args) -> int:
    command = [
        args.judge_python,
        args.judge_script,
        "--input_folder",
        args.input_folder,
        "--api_url",
        args.api_base,
        "--eval_model_name",
        args.model_name,
    ]
    log(f"Launching judge: {' '.join(command)}")
    return subprocess.call(command)


def main() -> int:
    args = parse_args()
    output_folder = infer_output_folder(args.input_folder)
    os.makedirs(output_folder, exist_ok=True)

    log(
        f"Watching {args.input_folder} and will write judge outputs to {output_folder}. "
        f"Polling every {args.poll_seconds}s."
    )

    while True:
        pending = pending_files(args.input_folder, output_folder)
        if not pending:
            log("No pending VStar raw files to judge.")
            time.sleep(args.poll_seconds)
            continue

        if not service_ready(args.api_base, args.model_name):
            log(f"Judge service {args.model_name} is not ready at {args.api_base}; pending files: {len(pending)}")
            time.sleep(args.poll_seconds)
            continue

        log(f"Judge service is ready. Pending files: {pending}")
        return_code = run_judge(args)
        log(f"Judge process exited with code {return_code}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(0)
