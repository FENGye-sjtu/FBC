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


def select_model_and_tool_devices(rows, weight_mib=17000, tool_gpu_free_mib=0,
                                  reserve_mib=1280):
    """Reserve a separate tool GPU only when both tools and model fit."""
    if tool_gpu_free_mib < 0:
        raise ValueError('Tool GPU free-memory requirement must be nonnegative')
    if not tool_gpu_free_mib:
        return select_devices(rows, weight_mib, reserve_mib), None
    # Preserve the largest spare-memory cards for the model. Tool selection has
    # no utilization gate, and may use a card too small to be a weight shard.
    for tool in sorted(rows, key=lambda row: (row[1], row[0])):
        if tool[1] < tool_gpu_free_mib or tool[1] <= 256:
            continue
        remaining = [row for row in rows if row[0] != tool[0]]
        selected = select_devices(remaining, weight_mib, reserve_mib)
        if selected:
            return selected, tool
    return [], None


def launch_environment(selected, tool, base_env):
    """Expose tools last while limiting model placement to weight GPUs only."""
    visible = selected + ([tool] if tool is not None else [])
    env = dict(base_env, CUDA_VISIBLE_DEVICES=','.join(str(i) for i, _ in visible),
               CVPR_GPU_BUDGETS_MIB=json.dumps(
                   {str(j): free-256 for j, (_, free) in enumerate(selected)}))
    env.pop('CVPR_GPU_BUDGET_MIB', None)
    env.pop('CVPR_TOOL_DEVICE', None)
    env.pop('CVPR_TOOL_GPU_BUDGET_MIB', None)
    if tool is not None:
        env['CVPR_TOOL_DEVICE'] = f'cuda:{len(selected)}'
        env['CVPR_TOOL_GPU_BUDGET_MIB'] = str(tool[1]-256)
    return env


def main():
    import fcntl
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--weight-mib', type=int, default=17000)
    p.add_argument('--tool-gpu-free-mib', type=int, default=0,
                   help='Minimum free MiB on a separate tool GPU; 0 keeps weight-only scheduling')
    p.add_argument('--timeout-hours', type=float, default=48)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.tool_gpu_free_mib < 0:
        p.error('--tool-gpu-free-mib must be nonnegative')
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
        selected, tool = select_model_and_tool_devices(rows, args.weight_mib, args.tool_gpu_free_mib)
        tool_status = {'tool_gpu':tool[0] if tool else None,
                       'tool_gpu_free_mib':tool[1] if tool else None,
                       'tool_gpu_budget_mib':tool[1]-256 if tool else None,
                       'tool_device':f'cuda:{len(selected)}' if tool else None,
                       'tool_gpu_required_free_mib':args.tool_gpu_free_mib}
        path.write_text(json.dumps({'state':'waiting','snapshot':rows,'command':command,**tool_status}))
        if selected:
            # The model loader uses capped allocator budgets and forbids CPU fallback.
            env = launch_environment(selected, tool, os.environ)
            path.write_text(json.dumps({'state':'running','devices':selected,'command':command,
                                        'started':time.time(),**tool_status}))
            code = subprocess.call(command,env=env)
            path.write_text(json.dumps({'state':'completed' if code==0 else 'failed',
                                        'exit_code':code,'devices':selected,'finished':time.time(),**tool_status}))
            raise SystemExit(code)
        time.sleep(15)
    raise SystemExit('Timed out waiting for combined BF16 memory')


if __name__ == '__main__':
    main()
