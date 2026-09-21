"""Share busy GPUs whenever the requested memory fits; never evict other jobs."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--min-free-mib', type=int, default=3500)
    p.add_argument('--timeout-hours', type=float, default=48)
    p.add_argument('--assets-ready', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    root = Path('/data_cinema/zqc/CVPR')
    status = root/'runtime'/f'{args.name}.json'
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        p.error('Need a command to launch')
    lock = (root/'runtime'/f'{args.name}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = time.time()
    while time.time()-start < args.timeout_hours*3600:
        ready = True
        if args.assets_ready:
            asset_path = root/'runtime/assets.json'
            try:
                assets = json.loads(asset_path.read_text()) if asset_path.exists() else {}
            except json.JSONDecodeError:
                assets = {}
            ready = len(assets) == 5 and all(a['state'] == 'ready' for a in assets.values())
            if ready:
                # A metadata-only snapshot is not sufficient for inference.
                for name in ('dino', 'qwen7b'):
                    folder = Path(assets[name]['result']['path'])
                    weights = list(folder.glob('*.safetensors')) + list(folder.glob('pytorch_model*.bin'))
                    ready = ready and bool(weights) and all(w.stat().st_size > 10000 for w in weights)
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        candidates = [row.split(',') for row in raw.strip().splitlines()]
        eligible = [int(i) for i, free, util in sorted(candidates, key=lambda r:int(r[1]), reverse=True)
                    if int(free) >= args.min_free_mib]
        status.write_text(json.dumps({'state':'waiting', 'assets_ready':ready, 'gpu_snapshot':raw,
                                     'elapsed_seconds':time.time()-start, 'command':command}, indent=2))
        if ready and eligible:
            # Cooperative lock with all queues in this CVPR project.
            gpu = eligible[0]
            gpu_lock = (root/'runtime'/f'gpu-{gpu}.lock').open('w')
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close(); time.sleep(30); continue
            # Recheck after acquiring the lock; external schedulers remain independent.
            fresh = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free,utilization.gpu',
                                             '--format=csv,noheader,nounits'], text=True).strip().split(',')
            if int(fresh[0]) < args.min_free_mib:
                gpu_lock.close(); time.sleep(30); continue
            import os
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       CVPR_GPU_BUDGET_MIB=str(max(1, int(fresh[0])-256)))
            status.write_text(json.dumps({'state':'running', 'gpu':gpu, 'command':command, 'started':time.time()}, indent=2))
            code = subprocess.call(command, env=env)
            status.write_text(json.dumps({'state':'completed' if code == 0 else 'failed', 'exit_code':code,
                                         'gpu':gpu, 'command':command, 'finished':time.time()}, indent=2))
            raise SystemExit(code)
        time.sleep(30)
    status.write_text(json.dumps({'state':'timed_out_waiting_for_resources', 'command':command}, indent=2))
    raise SystemExit(2)


if __name__ == '__main__':
    main()
