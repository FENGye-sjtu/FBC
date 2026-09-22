"""Serial file-queue worker, retaining frozen weights across development jobs."""
import argparse
import importlib
import json
from pathlib import Path
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--idle-seconds', type=int, default=1800)
    args = p.parse_args()
    root = args.directory.resolve()
    if not any(base in root.parents for base in [Path('/data_cinema/zqc/CVPR'), Path('/data_wyn/zqc/CVPR')]):
        p.error('Worker directory must be inside authorized CVPR roots')
    (root/'inbox').mkdir(parents=True, exist_ok=True)
    (root/'results').mkdir(exist_ok=True)
    shared, last_job, failures = {}, time.monotonic(), 0
    while time.monotonic()-last_job < args.idle_seconds:
        pending = [f for f in sorted((root/'inbox').glob('*.json')) if not (root/'results'/f.name).exists()]
        if not pending:
            time.sleep(3)
            continue
        job = pending[0]
        request = json.loads(job.read_text())
        (root/'active.json').write_text(json.dumps({'job':job.name,'started':time.time()}))
        print('START_JOB', job.name, flush=True)
        try:
            # Never reload during an active job. Existing manifests pin code hashes.
            for module in ('core','answers','prompts','qwen','online','online_repair',
                           'evidence_first','faithact','format_adapter','run'):
                loaded = importlib.import_module('trainingfree.'+module)
                importlib.reload(loaded)
            from .run import main as run
            run(request['args'], shared=shared)
            result = {'state':'completed','finished':time.time()}
        except (Exception, SystemExit) as exc:
            failures += 1
            result = {'state':'failed','error':str(exc),'traceback':traceback.format_exc(),'finished':time.time()}
            print(result['traceback'], flush=True)
            if 'vlm' in shared:
                shared['vlm'].torch.cuda.empty_cache()
        (root/'results'/job.name).write_text(json.dumps(result, indent=2))
        (root/'active.json').write_text(json.dumps({'state':'idle','last_job':job.name, **result}))
        print('END_JOB', job.name, result['state'], flush=True)
        last_job = time.monotonic()
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
