"""DeepEyesV2/LLaMA-Factory SFT entry with a BF16 visual encoder and NF4 language model."""
import json
from pathlib import Path
from efa.precision import configure_mixed_precision


def main():
    import llamafactory.model.patcher as patcher
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback
    original=patcher.configure_quantization
    def configure(config,tokenizer,model_args,init_kwargs):
        configure_mixed_precision(config,tokenizer,model_args,init_kwargs,original)
    patcher.configure_quantization=configure
    class PrecisionCheck(TrainerCallback):
        def on_train_begin(self,args,state,control,model=None,**kwargs):
            base=model.get_base_model() if hasattr(model,'get_base_model') else model
            visual_types=sorted({type(m).__name__ for m in base.visual.modules() if 'Linear' in type(m).__name__})
            language_types=sorted({type(m).__name__ for m in base.model.modules() if 'Linear' in type(m).__name__})
            if any('4bit' in name or '8bit' in name for name in visual_types):
                raise RuntimeError('The visual tower must remain unquantized')
            if 'Linear4bit' not in language_types:
                raise RuntimeError('Expected NF4 language weights in this SFT entry')
            report=dict(mode='bf16_vision_nf4_language',vision_linear_types=visual_types,
                language_linear_types=language_types,visual_dtype=str(base.visual.dtype))
            out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
            (out/'precision_policy.json').write_text(json.dumps(report,indent=2))
            print('VERIFIED_TRAINING_PRECISION',json.dumps(report),flush=True)
    run_exp(callbacks=[PrecisionCheck()])


if __name__=='__main__':main()
