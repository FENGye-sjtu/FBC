"""Wait for spare GPU memory and run one BF16 job inside this workstream."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def select_devices(rows: list[tuple[int, int]], weight_mib: int,
                   reserve_mib: int, min_devices: int = 1,
                   min_each_free_mib: int = 0) -> list[tuple[int, int]]:
    selected: list[tuple[int, int]] = []
    capacity = 0
    for index, free in sorted(rows, key=lambda row: row[1], reverse=True):
        if free < max(reserve_mib + 2048, min_each_free_mib):
            continue
        selected.append((index, free))
        capacity += free - reserve_mib
        if capacity >= weight_mib and len(selected) >= min_devices:
            return selected
    return []


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--weight-mib", type=int, default=17000)
    parser.add_argument("--reserve-mib", type=int, default=1536)
    parser.add_argument("--timeout-hours", type=float, default=48)
    parser.add_argument("--min-devices", type=int, default=1)
    parser.add_argument("--min-each-free-mib", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root.resolve()
    authorized = (Path("/data_cinema/zqc/CVPR"), Path("/data_wyn/zqc/CVPR"))
    if not any(base == root or base in root.parents for base in authorized):
        parser.error("root must stay inside an authorized CVPR directory")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "queue_status.json"
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        while time.time() - started < args.timeout_hours * 3600:
            raw = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ], text=True)
            rows = [tuple(map(int, line.split(","))) for line in raw.splitlines() if line.strip()]
            selected = select_devices(rows, args.weight_mib, args.reserve_mib,
                                      args.min_devices, args.min_each_free_mib)
            status_path.write_text(json.dumps({
                "state": "waiting", "snapshot": rows, "command": command,
                "updated": time.time(),
            }, indent=2))
            if selected:
                budgets = {str(local): free - 256
                           for local, (_, free) in enumerate(selected)}
                env = dict(os.environ,
                           CUDA_VISIBLE_DEVICES=",".join(str(index) for index, _ in selected),
                           CVPR_GPU_BUDGETS_MIB=json.dumps(budgets))
                status_path.write_text(json.dumps({
                    "state": "running", "devices": selected, "command": command,
                    "started": time.time(),
                }, indent=2))
                code = subprocess.call(command, env=env)
                status_path.write_text(json.dumps({
                    "state": "completed" if code == 0 else "failed",
                    "exit_code": code, "devices": selected, "finished": time.time(),
                }, indent=2))
                raise SystemExit(code)
            time.sleep(15)
    raise SystemExit("timed out waiting for BF16 memory")


if __name__ == "__main__":
    main()
