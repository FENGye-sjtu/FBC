"""One resident frozen Qwen2.5-VL. Each call starts with a fresh chat/cache."""
import math
import json
import os


def align_rope_delta(model, cache_position):
    """Place Qwen's persistent RoPE delta beside the dispatched cache index."""
    if cache_position is not None and model.rope_deltas is not None:
        model.rope_deltas = model.rope_deltas.to(cache_position.device)


class Qwen:
    def __init__(self, model_path, device='cuda:0', max_pixels=1003520, max_new_tokens=384,
                 quantization='none'):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.device = device
        self.signature = (str(model_path), device, max_pixels, quantization)
        self.max_new_tokens = max_new_tokens
        options = {}
        if device == 'auto':
            if quantization != 'none':
                raise ValueError('Multi-GPU replication uses unquantized BF16 weights')
            budgets = json.loads(os.environ['CVPR_GPU_BUDGETS_MIB'])
            # Launch-time free-memory budgets, never utilization-based scheduling.
            # Reserve 1 GiB per device for vision activations and generation caches.
            options['max_memory'] = {int(i): (int(mib)-1024)*1024**2 for i,mib in budgets.items()}
            options['max_memory']['cpu'] = 0  # No silent slow CPU/disk offload.
            from .placement import module_bytes, pack_modules
            options['device_map'] = pack_modules(module_bytes(model_path),
                {i:value for i,value in options['max_memory'].items() if isinstance(i,int)})
            for i,mib in budgets.items():
                total = torch.cuda.get_device_properties(int(i)).total_memory
                torch.cuda.set_per_process_memory_fraction(min(1.,int(mib)*1024**2/total),int(i))
        if device.startswith('cuda') and os.environ.get('CVPR_GPU_BUDGET_MIB'):
            budget = int(os.environ['CVPR_GPU_BUDGET_MIB']) * 1024**2
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1., budget/total), device)
        # Transformers 4.57 switches this checkpoint to the fast processor by
        # default. Pin the original slow path so new mixed-model runs remain
        # comparable to the earlier Qwen2.5 BF16/NF4 measurements.
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28,
                                                       max_pixels=max_pixels, use_fast=False)
        if quantization == 'nf4':
            from transformers import BitsAndBytesConfig
            options['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 if device.startswith('cuda') or device == 'auto' else torch.float32,
            attn_implementation='sdpa', device_map=options.pop('device_map',device), **options).eval()
        self.model.requires_grad_(False)
        if device == 'auto':
            self.device = str(self.model.get_input_embeddings().weight.device)
            if any(str(d) in ('cpu','disk') for d in self.model.hf_device_map.values()):
                raise ValueError('BF16 model did not fit entirely on selected GPUs')
        self.calls, self.generated_tokens, self.input_tokens = 0, 0, 0

    def inputs(self, image, prompt):
        from qwen_vl_utils import process_vision_info
        images_in = image if isinstance(image, (list, tuple)) else ([image] if image is not None else [])
        image_processor = self.processor.image_processor
        old_min, old_max = image_processor.min_pixels, image_processor.max_pixels
        budget = max(28*28, old_max // max(1,len(images_in)))
        content = [{'type': 'image', 'image': im, 'min_pixels':min(old_min,budget),
                    'max_pixels':budget} for im in images_in]
        content.append({'type': 'text', 'text': prompt})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        image_processor.min_pixels, image_processor.max_pixels = min(old_min,budget), budget
        try:
            return self.processor(text=[text], images=images, videos=videos,
                                  padding=True, return_tensors='pt').to(self.device)
        finally:
            image_processor.min_pixels, image_processor.max_pixels = old_min, old_max

    def generate(self, image, prompt, max_tokens=None, schema=None):
        inputs = self.inputs(image, prompt)
        # Qwen2.5-VL caches this tensor on the device used by the previous
        # generation. Every adapter call is a fresh conversation; retaining it
        # can mix cache_position on one shard with rope_deltas on another.
        self.model.rope_deltas = None
        kwargs = {}
        length = inputs.input_ids.shape[1]
        if schema is not None:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            from transformers import StoppingCriteria, StoppingCriteriaList
            kwargs['prefix_allowed_tokens_fn'] = build_transformers_prefix_allowed_tokens_fn(
                self.processor.tokenizer, JsonSchemaParser(schema))
            tokenizer = self.processor.tokenizer

            class CompleteJSON(StoppingCriteria):
                def __call__(self, input_ids, scores, **unused):
                    text = tokenizer.decode(input_ids[0, length:], skip_special_tokens=True)
                    try:
                        json.loads(text)
                        return True
                    except (ValueError, TypeError):
                        return False

            kwargs['stopping_criteria'] = StoppingCriteriaList([CompleteJSON()])
        handle = self.model.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
        # Accelerate moves cache_position in its dispatch hook, after ordinary
        # module pre-hooks run. Wrap the already-bound _old_forward so alignment
        # happens after dispatch and immediately before Qwen adds these tensors.
        dispatched_forward = getattr(self.model, '_old_forward', None)
        if dispatched_forward is not None:
            def aligned_forward(*args, **forward_kwargs):
                cache = forward_kwargs.get('cache_position')
                align_rope_delta(self.model, cache)
                return dispatched_forward(*args, **forward_kwargs)
            self.model._old_forward = aligned_forward
        try:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_tokens or self.max_new_tokens,
                                             do_sample=False, use_cache=True, **kwargs)
        finally:
            handle.remove()
            if dispatched_forward is not None:
                self.model._old_forward = dispatched_forward
        self.calls += 1
        self.input_tokens += length
        self.generated_tokens += output.shape[1] - length
        return self.processor.batch_decode(output[:, length:], skip_special_tokens=True)[0]

    def model_image_size(self, image):
        """Coordinates refer to the model's processed image, not source-file pixels."""
        key = (image.size, self.processor.image_processor.min_pixels,
               self.processor.image_processor.max_pixels)
        cache = getattr(self, '_grid_size_cache', {})
        if key not in cache:
            encoded = self.inputs(image, '')
            grid = encoded.image_grid_thw[0].tolist()
            patch = self.processor.image_processor.patch_size
            cache[key] = (int(grid[2])*patch, int(grid[1])*patch)
            self._grid_size_cache = cache
        return cache[key]

    def log_probability(self, image, prompt, completion):
        """Teacher-forced full completion log likelihood, no generated confidence."""
        torch = self.torch
        inputs = self.inputs(image, prompt)
        prefix_length = inputs.input_ids.shape[1]
        suffix = self.processor.tokenizer(completion, add_special_tokens=False,
                                          return_tensors='pt').input_ids.to(self.device)
        if suffix.shape[1] == 0:
            raise ValueError('Empty scoring completion')
        inputs['input_ids'] = torch.cat([inputs.input_ids, suffix], dim=1)
        inputs['attention_mask'] = torch.ones_like(inputs.input_ids)
        with torch.inference_mode():
            logits = self._last_logits(inputs, suffix.shape[1]+1)[:, :-1, :]
            score = logits.log_softmax(-1).gather(-1, suffix.unsqueeze(-1)).sum().item()
        self.calls += 1
        self.input_tokens += inputs.input_ids.shape[1]
        return score

    def option_log_probabilities(self, image, prompt, labels):
        """Score single-token options with one shared prefix forward pass."""
        encoded = {label: self.processor.tokenizer.encode(label, add_special_tokens=False)
                   for label in labels}
        if not all(len(ids) == 1 for ids in encoded.values()):
            return {label:self.log_probability(image, prompt, label) for label in labels}
        inputs = self.inputs(image, prompt)
        with self.torch.inference_mode():
            logits = self._last_logits(inputs, 1)[0, -1].log_softmax(-1)
            result = {label:logits[ids[0]].item() for label, ids in encoded.items()}
        self.calls += 1
        self.input_tokens += inputs.input_ids.shape[1]
        return result

    def _last_logits(self, inputs, count):
        # Transformers 4.51 Qwen does not expose logits_to_keep. Project only
        # needed final hidden states instead of allocating the full prefix vocabulary.
        self.model.rope_deltas = None
        handle = self.model.lm_head.register_forward_pre_hook(
            lambda module, args: (args[0][:, -count:, :],))
        try:
            return self.model(**inputs, use_cache=False).logits.float()
        finally:
            handle.remove()

    def yes_probability(self, image, prompt):
        yes_ids = self.processor.tokenizer.encode('Yes', add_special_tokens=False)
        no_ids = self.processor.tokenizer.encode('No', add_special_tokens=False)
        if len(yes_ids) == len(no_ids) == 1:
            inputs = self.inputs(image, prompt)
            with self.torch.inference_mode():
                logits = self._last_logits(inputs, 1)[0, -1]
                value = self.torch.softmax(logits[[yes_ids[0], no_ids[0]]], dim=0)[0].item()
            self.calls += 1
            self.input_tokens += inputs.input_ids.shape[1]
            return value
        yes = self.log_probability(image, prompt, 'Yes')
        no = self.log_probability(image, prompt, 'No')
        delta = max(-60., min(60., yes - no))
        return 1 / (1 + math.exp(-delta))
