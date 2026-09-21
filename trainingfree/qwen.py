"""One resident frozen Qwen2.5-VL. Each call starts with a fresh chat/cache."""
import math
import json


class Qwen:
    def __init__(self, model_path, device='cuda:0', max_pixels=1003520, max_new_tokens=384):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28,
                                                       max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,
            attn_implementation='sdpa', device_map=device).eval()
        self.model.requires_grad_(False)
        self.calls, self.generated_tokens, self.input_tokens = 0, 0, 0

    def inputs(self, image, prompt):
        from qwen_vl_utils import process_vision_info
        content = ([{'type': 'image', 'image': image}] if image is not None else [])
        content.append({'type': 'text', 'text': prompt})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        return self.processor(text=[text], images=images, videos=videos,
                              padding=True, return_tensors='pt').to(self.device)

    def generate(self, image, prompt, max_tokens=None, schema=None):
        inputs = self.inputs(image, prompt)
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
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens or self.max_new_tokens,
                                         do_sample=False, use_cache=True, **kwargs)
        self.calls += 1
        self.input_tokens += length
        self.generated_tokens += output.shape[1] - length
        return self.processor.batch_decode(output[:, length:], skip_special_tokens=True)[0]

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
            logits = self.model(**inputs, use_cache=False).logits[:, prefix_length-1:-1, :].float()
            score = logits.log_softmax(-1).gather(-1, suffix.unsqueeze(-1)).sum().item()
        self.calls += 1
        self.input_tokens += inputs.input_ids.shape[1]
        return score

    def yes_probability(self, image, prompt):
        yes = self.log_probability(image, prompt, 'Yes')
        no = self.log_probability(image, prompt, 'No')
        delta = max(-60., min(60., yes - no))
        return 1 / (1 + math.exp(-delta))
