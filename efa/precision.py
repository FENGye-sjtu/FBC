"""Shared visual precision contract for SFT and compact-reader checkpoints."""


def configure_mixed_precision(config,tokenizer,model_args,init_kwargs,configure):
    configure(config,tokenizer,model_args,init_kwargs)
    quant=init_kwargs.get('quantization_config')
    if quant is None:
        raise ValueError('SFT quantization was not configured; use quantization_method=bnb and quantization_bit=4')
    if quant is not None:
        if config.model_type not in ('qwen2_vl','qwen2_5_vl'):
            raise ValueError('Visual precision adapter supports Qwen2/2.5-VL')
        # An explicit list replaces Transformers' default exclusions. Keep the
        # tied output projection as well as vision in floating point.
        quant.llm_int8_skip_modules=list(dict.fromkeys((quant.llm_int8_skip_modules or [])+['lm_head','visual']))


def check_reader_precision(checkpoint,current,allow_adapter_mismatch=False):
    trained=checkpoint.get('encoder_config')
    if trained is None:
        if not current.get('vision_quantized'):
            raise ValueError('Legacy reader used NF4 vision; retrain it or explicitly set quantize_vision=True for historical replay')
    elif trained.get('vision_quantized')!=current.get('vision_quantized'):
        raise ValueError('Reader checkpoint visual precision differs from the current encoder; retrain alignment')
    if trained and not allow_adapter_mismatch and trained.get('adapter_sha256')!=current.get('adapter_sha256'):
        raise ValueError('Reader checkpoint was trained with a different shared adapter')
