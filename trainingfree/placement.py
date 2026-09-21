"""Explicit BF16 module packing, avoiding auto-map reserve/fragmentation offload."""
import json
import math
from pathlib import Path
import struct


def module_bytes(model_path):
    root = Path(model_path)
    index = json.loads((root/'model.safetensors.index.json').read_text())
    sizes = {}
    for filename in sorted(set(index['weight_map'].values())):
        with (root/filename).open('rb') as stream:
            header_size = struct.unpack('<Q',stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        for name,entry in header.items():
            if name == '__metadata__':
                continue
            parts = name.split('.')
            group = '.'.join(parts[:3]) if name.startswith('model.layers.') else (
                'visual' if parts[0]=='visual' else '.'.join(parts[:-1]))
            sizes[group] = sizes.get(group,0) + math.prod(entry['shape'])*2
    return sizes


def pack_modules(sizes, capacities):
    remaining = {int(i):int(value) for i,value in capacities.items()}
    placement = {}
    for name,size in sorted(sizes.items(),key=lambda item:item[1],reverse=True):
        eligible = [i for i,free in remaining.items() if free>=size]
        if not eligible:
            raise ValueError(f'BF16 module {name} ({size/1024**2:.1f} MiB) does not fit GPU budgets')
        device = max(eligible,key=lambda i:remaining[i])
        placement[name] = device
        remaining[device] -= size
    # Do NOT add a root ('') entry: Accelerate's root placement hook would move
    # all submodule tensors onto one GPU, undoing this explicit sharding.
    # Cover the nonpersistent rotary buffer at its own module boundary instead.
    placement['model.rotary_emb'] = placement.get('model.embed_tokens',0)
    return placement
