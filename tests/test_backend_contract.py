import unittest
import tempfile
from pathlib import Path
from efa.core import Claim
from efa.qwen import QwenBackend, parse_json
from efa.format import STEP_END
from PIL import Image


class BackendTests(unittest.TestCase):
    def test_new_image_clears_cached_sources_and_parent_context(self):
        b=self.backend();b.features={'old':'pixels'};b.calls=['old call']
        old=b.image_digest
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'new.png'
            Image.new('RGB',(20,30),'blue').save(path)
            b.set_image(path)
        self.assertNotEqual(b.image_digest,old)
        self.assertEqual(b.image.size,(20,30))
        self.assertEqual((b.features,b.calls,b.premises),({},[],[]))

    def backend(self):
        b=QwenBackend.__new__(QwenBackend)
        b.image=Image.new('RGB',(100,100))
        b.premises=[dict(key='a@1',claim={'value':'2'},verified=False)]
        b.image_digest='pixels'
        b.encoder_config={'model':'fixture'}
        return b

    def test_premise_repair_never_reads_pixels(self):
        b=self.backend(); b.begin(Claim('sum','value','4',kind='premise'))
        source=b.route({'subject':'sum','predicate':'value'})
        reads=[]
        def generate(prompt,boxes,**kwargs):
            reads.append((prompt,boxes));return ['4']
        def logp(prompt,boxes,value):
            reads.append((prompt,boxes));return -1,1
        b.generate=generate;b.logp=logp
        q={'subject':'sum','predicate':'value'}
        b.propose(q,source,1)
        b.rescore(q,b.final_context(q,source),['4'])
        self.assertTrue(all('explicit premises' in p and boxes==[] for p,boxes in reads))
        self.assertEqual(b.describe_source(q,source)['parents'],['a@1'])

    def test_step_boundary_and_trailing_content(self):
        self.assertEqual(parse_json('{"atoms": []}'+STEP_END+'\n'),{'atoms':[]})
        with self.assertRaises(ValueError):
            parse_json('{"atoms": []}'+STEP_END+'unchecked claim')


if __name__=='__main__':
    unittest.main()
