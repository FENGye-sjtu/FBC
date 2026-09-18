"""Real Qwen-VL backend with independent crop inputs and teacher-forced scoring.

This first executable backend uses native crop tokens. It is explicitly a
source-isolated crop baseline, NOT the paper's trained eight-token resampler.
No scores from this backend are described as calibrated unless a policy is fitted.
"""
import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
import time

from PIL import Image
from efa.core import Scores
from efa.regions import Box, complement, source_version
from efa.format import STEP_END, route_prompt, read_prompt


def parse_json(text):
    text=text.rstrip().removesuffix(STEP_END).rstrip()
    start=text.find("{")
    if start<0:
        raise ValueError("No JSON object")
    obj,end=json.JSONDecoder().raw_decode(text[start:])
    if text[start+end:].strip().strip('`').strip():
        raise ValueError("Uncovered text after JSON")
    return obj


@dataclass(frozen=True)
class Sources:
    target: tuple[Box,...]
    context: tuple[Box,...]
    valid: bool


class QwenBackend:
    representation="independently_encoded_native_crop_tokens"

    def __init__(self, model_path, image_path, quantized=False, max_pixels=200704, adapter_path=None):
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
        self.torch=torch
        self.image=Image.open(image_path).convert("RGB")
        self.image_digest=hashlib.sha256(self.image.tobytes()+str(self.image.size).encode()).hexdigest()
        self.encoder_config=dict(model=str(model_path),max_pixels=max_pixels,min_pixels=3136,quantized=quantized,
                                 representation=self.representation,crop_preprocessing="minimum_patch_resize_28_aspect_200")
        self.processor=AutoProcessor.from_pretrained(model_path,min_pixels=3136,max_pixels=max_pixels)
        options=dict(torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa")
        if quantized:
            options["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True)
        model_type=AutoConfig.from_pretrained(model_path).model_type
        implementations={'qwen2_vl':Qwen2VLForConditionalGeneration,'qwen2_5_vl':Qwen2_5_VLForConditionalGeneration}
        if model_type not in implementations:
            raise ValueError('Unsupported Qwen vision architecture: '+model_type)
        self.encoder_config['model_type']=model_type
        self.model=implementations[model_type].from_pretrained(model_path,**options).eval()
        if adapter_path:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,adapter_path).eval()
            self.encoder_config['adapter']=str(adapter_path)
            weights=Path(adapter_path)/'adapter_model.safetensors'
            self.encoder_config['adapter_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
        self.device=self.model.get_input_embeddings().weight.device
        self.calls=[]
        self.premises=[]
        self.audit_kind="visual"

    def begin(self,claim):
        self.audit_kind=claim.kind

    def describe_source(self,query,source):
        details=dict(target=[vars(b) for b in source.target],context=[vars(b) for b in source.context],
                     valid=source.valid,kind=self.audit_kind)
        config=dict(self.encoder_config,source=details)
        if self.audit_kind=="premise":
            config["premises"]=self.premises
        details["version"]=source_version(self.image_digest,query,source.target+source.context,config)
        details["image_sha256"]=self.image_digest
        details["encoder"]=self.encoder_config
        details["query"]=query
        details["parents"]=[p["key"] for p in self.premises] if self.audit_kind=="premise" else []
        return details

    def premise_prompt(self,query):
        return "Given ONLY these explicit premises: "+json.dumps(self.premises)+"\nAnswer: "+json.dumps(query)

    def inputs(self,prompt,boxes):
        images=[self.image.crop(tuple(vars(b).values())) for b in boxes]
        # Qwen's processor rejects sub-patch dimensions, which occur naturally
        # for chart lines and complement strips. Resize only the selected pixels;
        # never expand a deletion crop into a forbidden neighboring region.
        for i,im in enumerate(images):
            minimum=max(28,math.ceil(max(im.size)/200))
            size=(max(minimum,im.width),max(minimum,im.height))
            if size!=im.size:
                images[i]=im.resize(size,Image.Resampling.BICUBIC)
        content=[{"type":"image","image":im} for im in images]+[{"type":"text","text":prompt}]
        text=self.processor.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
        return self.processor(text=[text],images=images or None,return_tensors="pt").to(self.device)

    def generate(self,prompt,boxes,max_tokens=256,k=1,step_boundary=False,json_schema=None):
        inputs=self.inputs(prompt,boxes)
        start=time.monotonic()
        constraints={}
        if step_boundary or json_schema is not None:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn
            from efa.grammar import step_parser
            if not hasattr(self,'_grammar_tokens'):
                self._grammar_tokens=build_token_enforcer_tokenizer_data(self.processor.tokenizer)
            parser=step_parser() if step_boundary else JsonSchemaParser(json_schema)
            constraints['prefix_allowed_tokens_fn']=build_transformers_prefix_allowed_tokens_fn(self._grammar_tokens,parser)
        with self.torch.inference_mode():
            stop=dict(stop_strings=[STEP_END],tokenizer=self.processor.tokenizer) if step_boundary else {}
            output=self.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,
                num_beams=1 if k==1 else 6,num_return_sequences=k,use_cache=True,**stop,**constraints)
        values=self.processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        self.calls.append(dict(operation="generate",seconds=time.monotonic()-start,images=len(boxes),outputs=k,
                               constrained_serialization=bool(constraints)))
        return values

    def logp(self,prompt,boxes,value):
        torch=self.torch
        inputs=self.inputs(prompt,boxes)
        prefix=inputs.input_ids.shape[1]
        ids=self.processor.tokenizer.encode(value,add_special_tokens=False)
        ids.append(self.processor.tokenizer.eos_token_id)
        suffix=torch.tensor([ids],device=self.device,dtype=torch.long)
        inputs["input_ids"]=torch.cat([inputs.input_ids,suffix],dim=1)
        inputs["attention_mask"]=torch.ones_like(inputs.input_ids)
        start=time.monotonic()
        with torch.inference_mode():
            logits=self.model(**inputs,use_cache=False).logits[:,prefix-1:-1,:].float()
            result=logits.log_softmax(-1).gather(-1,suffix.unsqueeze(-1)).sum().item()
        self.calls.append(dict(operation="teacher_forcing",seconds=time.monotonic()-start,images=len(boxes),tokens=len(ids)))
        return result,len(ids)

    def route(self,query):
        if self.audit_kind=="premise":
            return Sources((),(),bool(self.premises))
        whole=Box(0,0,self.image.width,self.image.height)
        prompt=route_prompt(query,self.image.width,self.image.height)
        try:
            from efa.grammar import ROUTE_SCHEMA
            parsed=parse_json(self.generate(prompt,[whole],json_schema=ROUTE_SCHEMA)[0])
            boxes=tuple(Box(*(int(v) for v in b)) for b in parsed["boxes"])
            context=tuple(Box(*(int(v) for v in b)) for b in parsed.get("context_boxes",parsed["boxes"]))
            valid=bool(boxes and context and all(b.valid(self.image.width,self.image.height) for b in boxes+context))
            if not valid:
                raise ValueError("Invalid source boxes")
            return Sources(boxes,context,True)
        except (ValueError,KeyError,TypeError):
            return Sources((whole,),(whole,),False)

    def prompt(self,query,boxes):
        return read_prompt(query,[list(vars(b).values()) for b in boxes])

    def score(self,claim,source):
        if claim.kind!="visual":
            if claim.kind!="premise" or not self.premises:
                return Scores(-1e30,-1e30,None,None,-1e30,False)
            # Fresh text-only read of exactly the exposed ledger. The effective
            # verified bit is still inherited separately by the controller.
            prompt=self.premise_prompt(claim.query)
            score,n=self.logp(prompt,[],claim.value)
            binding,bn=self.logp("Given premises: "+json.dumps(self.premises)+
                "\nName the entity or entities involved in the deduction.",[],claim.subject)
            return Scores(binding/bn,score/n,None,None,score,source.valid)
        targets=list(source.target)
        context=list(dict.fromkeys(source.target+source.context))
        comp=complement(self.image.width,self.image.height,targets)
        full=list(dict.fromkeys(context+comp))
        # All views use the SAME neutral textual query/serialization, including
        # coordinates of each actual view; no prior main-trace cache is reused.
        local,n=self.logp(self.prompt(claim.query,targets),targets,claim.value)
        expanded,_=self.logp(self.prompt(claim.query,context),context,claim.value)
        b_prompt=("Identify the object in the FIRST crop relative to its original-frame anchors. "
                  "Return its category and spatial/ownership identity. Coordinates: "+
                  json.dumps([list(vars(b).values()) for b in context]))
        binding,bn=self.logp(b_prompt,context,claim.subject)
        deletion=None
        if comp:
            original,_=self.logp(self.prompt(claim.query,full),full,claim.value)
            deleted,_=self.logp(self.prompt(claim.query,comp),comp,claim.value)
            deletion=(original-deleted)/n
        # Disjoint complement pieces are geometry-only controls in v0. They are
        # not asserted to be semantically matched distractor objects.
        controls=sorted(comp,key=lambda b:(b.x2-b.x1)*(b.y2-b.y1),reverse=True)[:2]
        specificity=None
        if controls:
            scores=[self.logp(self.prompt(claim.query,[b]),[b],claim.value)[0]/n for b in controls]
            specificity=local/n-sum(scores)/len(scores)
        return Scores(binding/bn,min(local,expanded)/n,deletion,specificity,local,source.valid)

    def refine(self,query,source,failures,round_id):
        if self.audit_kind=="premise":
            return source
        if not source.valid:
            return self.route(query)
        ratio=(0.25,0.75,1.5)[round_id-1]
        context=tuple(b.expand(self.image.width,self.image.height,ratio) for b in source.context)
        if round_id==3:
            context=(Box(0,0,self.image.width,self.image.height),)
        return Sources(source.target,context,True)

    def propose(self,query,source,k):
        boxes=list(dict.fromkeys(source.target+source.context))
        prompt=self.premise_prompt(query) if self.audit_kind=="premise" else self.prompt(query,boxes)
        values=self.generate(prompt,boxes,max_tokens=32,k=k)
        return [v.strip() for v in values if v.strip()]

    def final_context(self,query,source):
        if self.audit_kind=="premise":
            return source
        return Sources(source.target,(Box(0,0,self.image.width,self.image.height),),source.valid)

    def rescore(self,query,source,values):
        boxes=list(dict.fromkeys(source.target+source.context))
        prompt=self.premise_prompt(query) if self.audit_kind=="premise" else self.prompt(query,boxes)
        return {v:self.logp(prompt,boxes,v)[0] for v in values}
