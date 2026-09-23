"""Explicit BF16 module packing, avoiding auto-map reserve/fragmentation offload."""
import json
import math
from pathlib import Path
import struct


def module_bytes(model_path, actual_dtypes=False):
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
            if actual_dtypes:
                dtype_bytes = {'F32': 4, 'I32': 4, 'U32': 4, 'F16': 2,
                               'BF16': 2, 'I16': 2, 'U16': 2, 'I8': 1,
                               'U8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1}
                try:
                    unit = dtype_bytes[entry['dtype']]
                except KeyError as exc:
                    raise ValueError(f'Unsupported safetensors dtype in placement: {entry["dtype"]}') from exc
            else:
                unit = 2  # Preserve the BF16 packing used by prior experiments.
            sizes[group] = sizes.get(group,0) + math.prod(entry['shape'])*unit
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
