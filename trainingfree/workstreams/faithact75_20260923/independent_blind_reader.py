"""Frozen Qwen3 reader used only for candidate-blind visual rereading.

Qwen2.5 remains the proposer, supporter and reasoner. This is an explicit
two-model ablation, not a single-model inference method or a trained module.
"""

from __future__ import annotations

import json
import os

from trainingfree.qwen import Qwen


BLIND_MARKER = "Inspect the image afresh and answer ONLY the neutral inspection question."


class Qwen3BlindReader:
    def __init__(self, model_path: str, device: str = "cuda:1",
                 max_pixels: int = 1003520):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=device,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        self.model.requires_grad_(False)
        self.input_tokens = 0
        self.generated_tokens = 0

    def generate(self, image, prompt: str, max_tokens: int = 128) -> str:
        images = image if isinstance(image, (list, tuple)) else [image]
        content = [{"type": "image", "image": item} for item in images if item is not None]
        content.append({"type": "text", "text": prompt + "\nReturn one valid JSON object only."})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
        self.input_tokens += inputs.input_ids.shape[-1]
        self.generated_tokens += output.shape[-1] - inputs.input_ids.shape[-1]
        raw = self.processor.batch_decode(
            output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()
        self.last_raw = raw
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            value = None
        if (not isinstance(value, dict)
                or not isinstance(value.get("observation"), str)
                or not isinstance(value.get("resolved"), bool)):
            # A malformed response is an unresolved inspection, never a fact.
            value = {"observation": "Independent inspection unavailable.",
                     "resolved": False}
        return json.dumps({"observation": value["observation"],
                           "resolved": value["resolved"]}, ensure_ascii=False)


class QwenWithIndependentBlindReader(Qwen):
    """Route only neutral blind rereads to Qwen3; all other calls stay Qwen2.5."""

    def __init__(self, model_path, device="cuda:0", max_pixels=1003520,
                 max_new_tokens=384, quantization="none"):
        super().__init__(model_path, device, max_pixels, max_new_tokens, quantization)
        reader_model = os.environ["CVPR_SECONDARY_MODEL"]
        self.independent_reader = Qwen3BlindReader(reader_model, "cuda:1", max_pixels)
        self.independent_reader_model = reader_model
        self.independent_reader_calls = 0

    def generate(self, image, prompt, max_tokens=None, schema=None):
        if prompt.startswith(BLIND_MARKER):
            self.independent_reader_calls += 1
            before = (self.independent_reader.input_tokens,
                      self.independent_reader.generated_tokens)
            result = self.independent_reader.generate(image, prompt, max_tokens or 128)
            self.calls += 1
            self.input_tokens += self.independent_reader.input_tokens - before[0]
            self.generated_tokens += self.independent_reader.generated_tokens - before[1]
            return result
        return super().generate(image, prompt, max_tokens=max_tokens, schema=schema)
