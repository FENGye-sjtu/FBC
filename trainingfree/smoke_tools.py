"""Exercise real FaithAct CLIP + GroundingDINO without reserving a GPU."""
import json
from pathlib import Path
import time


def main():
    import torch
    from PIL import Image
    from .faithact import FaithAct
    torch.set_num_threads(4)
    root = Path('/data_cinema/zqc/CVPR')
    start = time.monotonic()
    tools = FaithAct(None, root/'upstream/FaithAct', root/'models/clip-vit-large-patch14-336',
                    root/'models/grounding-dino-base', root/'models/faithact_poll/best.pt')
    image = Image.open('/data_cinema/zqc/VBC/data/benchmarks/vlmevalkit/images/MMStar/0.jpg').convert('RGB')
    description, result = tools.inspect(image, 'suitcase')
    record = {'description':description, 'evidence':result,
              'elapsed_seconds':time.monotonic()-start, 'kind':'tool_interface_smoke_not_benchmark'}
    (root/'runtime/tools_smoke.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
