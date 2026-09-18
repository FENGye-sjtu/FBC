from types import SimpleNamespace
import unittest
from efa.precision import check_reader_precision,configure_mixed_precision


class PrecisionTests(unittest.TestCase):
    def test_silent_upstream_quantization_omission_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'quantization was not configured'):
            configure_mixed_precision(None,None,None,{},lambda *args: None)

    def test_upstream_language_quantization_and_existing_exclusions_are_preserved(self):
        options={}
        def upstream(config,tokenizer,args,kwargs):
            kwargs['quantization_config']=SimpleNamespace(llm_int8_skip_modules=['lm_head'],load_in_4bit=True)
        configure_mixed_precision(SimpleNamespace(model_type='qwen2_vl'),None,None,options,upstream)
        self.assertTrue(options['quantization_config'].load_in_4bit)
        self.assertEqual(options['quantization_config'].llm_int8_skip_modules,['lm_head','visual'])
        def default_upstream(config,tokenizer,args,kwargs):
            kwargs['quantization_config']=SimpleNamespace(llm_int8_skip_modules=None,load_in_4bit=True)
        configure_mixed_precision(SimpleNamespace(model_type='qwen2_vl'),None,None,options,default_upstream)
        self.assertEqual(options['quantization_config'].llm_int8_skip_modules,['lm_head','visual'])

    def test_reader_cannot_silently_change_visual_feature_precision(self):
        with self.assertRaisesRegex(ValueError,'Legacy reader'):
            check_reader_precision({},dict(vision_quantized=False))
        check_reader_precision({},dict(vision_quantized=True))
        with self.assertRaisesRegex(ValueError,'differs'):
            check_reader_precision(dict(encoder_config=dict(vision_quantized=True)),dict(vision_quantized=False))
        check_reader_precision(dict(encoder_config=dict(vision_quantized=False)),dict(vision_quantized=False))

    def test_joint_reader_requires_its_saved_shared_adapter(self):
        trained=dict(vision_quantized=False,adapter_sha256='trained')
        check_reader_precision(dict(encoder_config=trained),dict(trained))
        with self.assertRaisesRegex(ValueError,'different shared adapter'):
            check_reader_precision(dict(encoder_config=trained),dict(vision_quantized=False,adapter_sha256='other'))
        # Deliberate training initialization may realign a previous reader.
        check_reader_precision(dict(encoder_config=trained),dict(vision_quantized=False,adapter_sha256='other'),True)
