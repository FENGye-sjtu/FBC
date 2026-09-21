"""Fit an unquantized model across spare memory on busy GPUs, without eviction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def select_devices(rows, weight_mib=17000, reserve_mib=1280):
    selected, capacity = [], 0
    for index, free in sorted(rows, key=lambda row: row[1], reverse=True):
        if free < reserve_mib + 2048:
            continue
        selected.append((index, free))
        capacity += free-reserve_mib
        if capacity >= weight_mib:
            return selected
    return []


def main():
    import fcntl
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--weight-mib', type=int, default=17000)
    p.add_argument('--timeout-hours', type=float, default=48)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    root = Path('/data_cinema/zqc/CVPR')
    lock = (root/'runtime'/f'{args.name}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = root/'runtime'/f'{args.name}.json'
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        p.error('Need child command')
    start = time.time()
    while time.time()-start < args.timeout_hours*3600:
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                       '--format=csv,noheader,nounits'],text=True)
        rows = [tuple(map(int,line.split(','))) for line in raw.strip().splitlines()]
        selected = select_devices(rows,args.weight_mib)
        path.write_text(json.dumps({'state':'waiting','snapshot':rows,'command':command}))
        if selected:
            # The model loader uses capped allocator budgets and forbids CPU fallback.
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(str(i) for i,_ in selected),
                       CVPR_GPU_BUDGETS_MIB=json.dumps({str(j):free-256 for j,(_,free) in enumerate(selected)}))
            env.pop('CVPR_GPU_BUDGET_MIB',None)
            path.write_text(json.dumps({'state':'running','devices':selected,'command':command,'started':time.time()}))
            code = subprocess.call(command,env=env)
            path.write_text(json.dumps({'state':'completed' if code==0 else 'failed',
                                        'exit_code':code,'devices':selected,'finished':time.time()}))
            raise SystemExit(code)
        time.sleep(15)
    raise SystemExit('Timed out waiting for combined BF16 memory')


if __name__ == '__main__':
    main()
