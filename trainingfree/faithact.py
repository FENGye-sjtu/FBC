"""Compatibility runner derived from pinned FaithAct's Qwen inference logic.

Original MIT code is loaded from the separately pinned upstream checkout; no
weight substitution. This is a port, not a claim of exact paper reproduction.
"""
import ast
import importlib.util
import inspect
from pathlib import Path


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class FaithAct:
    def __init__(self, vlm, upstream, clip_path, dino_path, checkpoint, tool_device='cpu',
                 box_threshold=.25, text_threshold=.20):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPProcessor, CLIPConfig, CLIPModel
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file() or checkpoint.stat().st_size < 10000:
            raise FileNotFoundError('Real FaithAct CLIP weights required; Git LFS pointer is not a checkpoint')
        self.vlm, self.torch, self.device = vlm, torch, tool_device
        self.template = module_from_file('faithact_template', Path(upstream)/'template.py')
        cls = module_from_file('faithact_clip', Path(upstream)/'train_pope/model.py').CLIP_cls
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        class FullCheckpointCLIP(cls):
            def __init__(self):
                torch.nn.Module.__init__(self)
                # Official best.pt includes BOTH encoders and the classifier.
                # Construct the same architecture without downloading them twice.
                self.clip = CLIPModel(CLIPConfig.from_pretrained(clip_path))
                self.proj_dim = self.clip.text_projection.out_features
                self.head = torch.nn.Sequential(torch.nn.Linear(self.proj_dim*2, 512),
                                               torch.nn.GELU(), torch.nn.Dropout(.1),
                                               torch.nn.Linear(512, 2))
        self.clip = FullCheckpointCLIP()
        self.clip.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.clip.to(tool_device).eval()
        self.clip.requires_grad_(False)
        self.dino_processor = AutoProcessor.from_pretrained(dino_path)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path).to(tool_device).eval()
        self.dino.requires_grad_(False)
        self.box_threshold, self.text_threshold = box_threshold, text_threshold

    def inspect(self, image, noun):
        inputs = self.dino_processor(images=image, text=noun+'.', return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            outputs = self.dino(**inputs)
        fn = self.dino_processor.post_process_grounded_object_detection
        kwargs = {'input_ids': inputs.input_ids, 'text_threshold': self.text_threshold,
                  'target_sizes': [image.size[::-1]]}
        key = 'box_threshold' if 'box_threshold' in inspect.signature(fn).parameters else 'threshold'
        kwargs[key] = self.box_threshold
        result = fn(outputs, **kwargs)[0]
        scores = result['scores'].cpu().tolist()
        boxes = result['boxes'].cpu().tolist()
        inputs = self.clip_processor(text=[noun], images=image, return_tensors='pt', padding=True).to(self.device)
        with self.torch.inference_mode():
            # Preserve upstream sigmoid(logit_yes), NOT a softmax replacement.
            clip_prob = self.torch.sigmoid(self.clip(**inputs)[:, 1]).item()
        probability = .7*clip_prob + .3*max(scores, default=0.)
        description = noun+': '
        if probability > .6:
            description += f'Object exists with probability {probability}. '
        elif probability >= .4:
            description += f'Object may exist with probability {probability}. '
        if probability >= .4:
            description += ' '.join(f'{i+1}.bbox: {box}' for i, box in enumerate(boxes))
        return description, {'noun': noun, 'poll': clip_prob, 'ground': scores,
                             'probability': probability, 'boxes': boxes}

    def run(self, image, question, original):
        objects, extraction_errors = [], []
        # Match the official script's blank-line split, including its first/last exclusion.
        for step in original.split('\n\n')[1:-1]:
            raw = self.vlm.generate(None, self.template.TEMPLATE_NOUNS.format(step))
            try:
                parsed = ast.literal_eval(raw.strip())
                if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
                    raise ValueError('Expected list of object names')
                objects.extend(parsed)
            except (ValueError, SyntaxError) as exc:
                extraction_errors.append({'raw': raw, 'error': str(exc)})
        descriptions, evidence = [], []
        for noun in dict.fromkeys(objects):
            description, record = self.inspect(image, noun)
            descriptions.append(description)
            evidence.append(record)
        question = question.replace('Please answer directly with only the letter of the correct option and nothing else.', '')
        prompt = self.template.TEMPLATE_EOT.format(question, '\n'.join(descriptions)) if descriptions else question
        output = self.vlm.generate(image, prompt)
        return {'raw': output, 'original_cot': original, 'evidence': evidence,
                'extraction_errors': extraction_errors, 'status': 'faithact_port'}
