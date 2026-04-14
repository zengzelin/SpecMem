#!/usr/bin/env python3
import argparse
import signal
import sys
import time

import torch


stop_requested = False


def parse_args():
    parser = argparse.ArgumentParser(description="SpecMem GPU runner.")
    parser.add_argument("--gpu", type=int, required=True, help="CUDA device index")
    parser.add_argument(
        "--leave-free-mib",
        type=int,
        default=512,
        help="How much free memory to leave on the target GPU",
    )
    parser.add_argument(
        "--reserve-mib",
        type=int,
        default=None,
        help="Reserve a fixed amount of memory instead of auto-using current free memory",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=512,
        help="Allocation chunk size in MiB",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="How often to print a heartbeat",
    )
    return parser.parse_args()


def mib_to_bytes(value_mib):
    return int(value_mib) * 1024 * 1024


def handle_signal(signum, _frame):
    global stop_requested
    stop_requested = True
    print(f"[specmem] received signal {signum}, releasing GPU memory...", flush=True)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid gpu index {args.gpu}, available count={torch.cuda.device_count()}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_mib = free_bytes // (1024 * 1024)
    total_mib = total_bytes // (1024 * 1024)

    if args.reserve_mib is not None:
        reserve_mib = args.reserve_mib
    else:
        reserve_mib = max(0, free_mib - args.leave_free_mib)

    if reserve_mib <= 0:
        print(
            f"[specmem] gpu{args.gpu} total={total_mib}MiB free={free_mib}MiB, nothing to reserve",
            flush=True,
        )
        return

    print(
        f"[specmem] using gpu{args.gpu}: total={total_mib}MiB free={free_mib}MiB target={reserve_mib}MiB",
        flush=True,
    )

    chunks = []
    remaining_mib = reserve_mib
    chunk_mib = max(1, args.chunk_mib)

    try:
        while remaining_mib > 0 and not stop_requested:
            this_chunk_mib = min(chunk_mib, remaining_mib)
            chunk = torch.empty(mib_to_bytes(this_chunk_mib), dtype=torch.uint8, device=device)
            chunk.fill_(1)
            chunks.append(chunk)
            remaining_mib -= this_chunk_mib
            allocated_mib = reserve_mib - remaining_mib
            print(
                f"[specmem] allocated {allocated_mib}/{reserve_mib}MiB on gpu{args.gpu}",
                flush=True,
            )
            torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        print("[specmem] hit OOM before target, keeping successfully allocated chunks", flush=True)

    heartbeat = 0
    while not stop_requested:
        if chunks:
            chunks[0][0] = (chunks[0][0] + 1) % 255
            torch.cuda.synchronize(device)

        free_bytes_now, total_bytes_now = torch.cuda.mem_get_info(device)
        free_mib_now = free_bytes_now // (1024 * 1024)
        used_mib_now = (total_bytes_now - free_bytes_now) // (1024 * 1024)
        heartbeat += 1
        print(
            f"[specmem] heartbeat={heartbeat} gpu{args.gpu} used={used_mib_now}MiB free={free_mib_now}MiB chunks={len(chunks)}",
            flush=True,
        )
        time.sleep(max(1, args.heartbeat_seconds))

    del chunks
    torch.cuda.empty_cache()
    print(f"[specmem] released gpu{args.gpu}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[specmem] fatal error: {exc}", file=sys.stderr, flush=True)
        raise
