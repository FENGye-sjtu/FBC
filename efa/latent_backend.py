"""Experimental compact source reader sharing the generator's Qwen weights.

Crop features are encoded independently and cached before value-dependent reads.
The readout uses eight source tokens per crop with sequential text positions;
native Qwen spatial positions remain inside each independent visual encoder.
"""
import hashlib
import json
import math
from pathlib import Path
import time
from PIL import Image
from efa.qwen import QwenBackend
from efa.latent import SourceResampler


class LatentBackend(QwenBackend):
    representation='independent_crops_query_resampler_8_tokens'

    def __init__(self,*args,resampler_path=None,allow_untrained=False,**kwargs):
        super().__init__(*args,**kwargs)
        torch=self.torch
        base=self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
        self.visual=base.visual
        hidden=self.model.get_input_embeddings().embedding_dim
        self.resampler=SourceResampler(hidden,latent_tokens=8).to(self.device,dtype=torch.bfloat16)
        self.features={}
        if resampler_path:
            state=torch.load(resampler_path,map_location=self.device,weights_only=True)
            self.resampler.load_state_dict(state['resampler'])
            if state['steps']<1:
                raise ValueError('The source reader has not been trained')
            self.encoder_config['resampler_sha256']=hashlib.sha256(Path(resampler_path).read_bytes()).hexdigest()
        elif not allow_untrained:
            raise ValueError('Supply a trained resampler checkpoint; random weights cannot audit')
        self.resampler.eval()
        self.encoder_config['source_position_mode']='sequential_readout_independent_native_vision'

    def crop_features(self,box):
        key=tuple(vars(box).values())
        if key not in self.features:
            crop=self.image.crop(key)
            minimum=max(28,math.ceil(max(crop.size)/200))
            size=(max(minimum,crop.width),max(minimum,crop.height))
            if size!=crop.size:
                crop=crop.resize(size,Image.Resampling.BICUBIC)
            inputs=self.processor.image_processor(images=[crop],return_tensors='pt')
            start=time.monotonic()
            with self.torch.no_grad():
                features=self.visual(inputs['pixel_values'].to(self.device,dtype=self.visual.dtype),
                                     grid_thw=inputs['image_grid_thw'].to(self.device))
            self.features[key]=features.detach()
            self.calls.append(dict(operation='independent_source_encode',seconds=time.monotonic()-start,
                                   source_box=list(key),visual_tokens=features.shape[0]))
        return self.features[key]

    def latent_inputs(self,prompt,boxes):
        torch=self.torch
        embed=self.model.get_input_embeddings()
        query_ids=self.processor.tokenizer.encode(prompt,add_special_tokens=False)
        query_features=embed(torch.tensor(query_ids,device=self.device)).detach()
        source=[]
        for box in boxes:
            coords=torch.tensor([box.x1/self.image.width,box.y1/self.image.height,
                                 box.x2/self.image.width,box.y2/self.image.height],
                                device=self.device,dtype=query_features.dtype)
            source.append(self.resampler(self.crop_features(box),query_features,coords).squeeze(0))
        markers='\n'.join('<|vision_start|>'+'<|image_pad|>'*8+'<|vision_end|>' for _ in boxes)
        text=self.processor.apply_chat_template([{'role':'user','content':markers+'\n'+prompt}],
                                                 tokenize=False,add_generation_prompt=True)
        inputs=self.processor.tokenizer(text,return_tensors='pt').to(self.device)
        embeddings=embed(inputs.input_ids)
        mask=inputs.input_ids==self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        if int(mask.sum())!=8*len(boxes):
            raise ValueError('Source token serialization mismatch')
        if source:
            embeddings=embeddings.clone()
            embeddings[mask]=torch.cat(source,dim=0).to(embeddings.dtype)
        inputs['inputs_embeds']=embeddings
        return inputs

    def sequence_logp(self,prompt,boxes,value):
        torch=self.torch
        inputs=self.latent_inputs(prompt,boxes)
        prefix=inputs.input_ids.shape[1]
        ids=self.processor.tokenizer.encode(value,add_special_tokens=False)+[self.processor.tokenizer.eos_token_id]
        suffix=torch.tensor([ids],device=self.device)
        inputs['input_ids']=torch.cat([inputs.input_ids,suffix],dim=1)
        inputs['inputs_embeds']=torch.cat([inputs.inputs_embeds,self.model.get_input_embeddings()(suffix)],dim=1)
        inputs['attention_mask']=torch.ones_like(inputs.input_ids)
        logits=self.model(**inputs,use_cache=False).logits[:,prefix-1:-1,:].float()
        return logits.log_softmax(-1).gather(-1,suffix.unsqueeze(-1)).sum(),len(ids)

    def logp(self,prompt,boxes,value):
        if not boxes:
            return super().logp(prompt,boxes,value)
        start=time.monotonic()
        with self.torch.inference_mode():
            score,n=self.sequence_logp(prompt,boxes,value)
        self.calls.append(dict(operation='latent_teacher_forcing',seconds=time.monotonic()-start,
                               source_tokens=8*len(boxes),answer_tokens=n))
        return score.item(),n

    def propose(self,query,source,k):
        if self.audit_kind=='premise':
            return super().propose(query,source,k)
        boxes=list(dict.fromkeys(source.target+source.context))
        start=time.monotonic()
        with self.torch.inference_mode():
            inputs=self.latent_inputs(self.prompt(query,boxes),boxes)
            output=self.model.generate(**inputs,max_new_tokens=32,do_sample=False,num_beams=6,
                                       num_return_sequences=k,use_cache=True)
        values=self.processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        self.calls.append(dict(operation='latent_repair_propose',seconds=time.monotonic()-start,outputs=k))
        return [v.strip() for v in values if v.strip()]
