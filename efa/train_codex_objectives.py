"""Train reviewed error pairs with positive replay; never fabricate repair traces."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from efa.codex_objectives import load_correction_pairs, attach_reviewed_replay, backward_codex_objective
from efa.latent_backend import LatentBackend
from efa.reader_data import load_training_sources, shuffled_reads, set_training_image


def main():
    p = argparse.ArgumentParser()
    for name in ('model', 'adapter', 'resampler', 'root', 'replay-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--replay-source', action='append', default=[], metavar='ROOT')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--adapter-learning-rate', type=float, default=1e-6)
    p.add_argument('--negative-weight', type=float, default=1.0)
    p.add_argument('--train-adapter', action='store_true')
    p.add_argument('--engineering-only', action='store_true')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() or a.epochs < 1:
        raise ValueError('Use a fresh output and positive epoch budget')
    if any(not math.isfinite(v) or v <= 0 for v in (a.learning_rate, a.adapter_learning_rate)) or not math.isfinite(a.negative_weight) or a.negative_weight < 0:
        raise ValueError('Invalid optimizer/loss parameters')
    pairs, pair_manifest = load_correction_pairs(a.root, a.engineering_only)
    reads, replay_manifest = load_training_sources(a.replay_root, 'reviewed_episodes.jsonl',
        [(root, 'reviewed_episodes.jsonl') for root in a.replay_source], a.engineering_only)
    rows = attach_reviewed_replay(pairs, reads)
    steps = len(rows) * a.epochs
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    # A mismatch is not an initialization shortcut here: this run requires the
    # actual reader aligned to the selected starting adapter.
    backend = LatentBackend(a.model, rows[0]['image_path'], True, max_pixels=50176,
        adapter_path=a.adapter, resampler_path=a.resampler)
    backend.model.requires_grad_(False)
    backend.resampler.train()
    parameters = list(backend.resampler.parameters())
    groups = [dict(params=list(parameters), lr=a.learning_rate)]
    if a.train_adapter:
        lora = []
        for name, param in backend.model.named_parameters():
            if '.lora_' in name:
                param.requires_grad_(True)
                lora.append(param)
        if not lora:
            raise ValueError('No shared LoRA parameters found')
        backend.model.train()
        backend.visual.eval()
        backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        groups.append(dict(params=lora, lr=a.adapter_learning_rate))
        parameters += lora
    optimizer = torch.optim.AdamW(groups)
    out.mkdir(parents=True)
    manifest = dict(corrections=pair_manifest, replay=replay_manifest, steps=steps,
                    epochs=a.epochs, sampling='all_pairs_and_all_reviewed_reads_shuffled_each_epoch')
    (out / 'input_sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    start = time.monotonic()
    losses = []
    with (out / 'losses.jsonl').open('w', encoding='utf-8') as stream:
        for step, row in enumerate(shuffled_reads(rows, steps, a.seed), 1):
            set_training_image(backend, row)
            optimizer.zero_grad(set_to_none=True)
            record = backward_codex_objective(backend, row, a.negative_weight)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            record.update(step=step, id=row['id'])
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            losses.append(record['total'])
            print(json.dumps(record), flush=True)
    if a.train_adapter:
        adapter = out / 'adapter'
        backend.model.save_pretrained(adapter, safe_serialization=True)
        backend.encoder_config.update(adapter=str(adapter),
            adapter_sha256=hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest())
    torch.save(dict(resampler=backend.resampler.state_dict(), steps=steps,
                    encoder_config=backend.encoder_config), out / 'resampler.pt')
    metrics = dict(steps=steps, epochs=a.epochs, examples=len(rows),
        correction_pairs=len(pairs), positive_replay=len(reads),
        mean_loss=sum(losses) / len(losses), seconds=time.monotonic() - start,
        encoder_config=backend.encoder_config, config=vars(a), joint_adapter=a.train_adapter,
        objectives=['summed_positive_sequence_ce', 'whole_negative_sequence_unlikelihood'],
        actual_student_repairs=0, calibrated=False, input_sources=manifest,
        scope='codex_reviewed_source_value_discrimination_with_positive_replay',
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
