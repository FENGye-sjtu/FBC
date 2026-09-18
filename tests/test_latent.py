import unittest
try:
    import torch
    from efa.latent import SourceResampler,unlikelihood
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Optional torch extra is unavailable')
class LatentTests(unittest.TestCase):
    def test_eight_tokens_and_trainable_path(self):
        torch.manual_seed(1)
        reader=SourceResampler(32,8,4)
        sources=torch.randn(19,32)
        query=torch.randn(5,32)
        output=reader(sources,query,torch.tensor([.1,.2,.3,.4]))
        self.assertEqual(tuple(output.shape),(1,8,32))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(reader.latents.grad).all())
        self.assertGreater(reader.latents.grad.abs().sum().item(),0)
        self.assertTrue(torch.equal(output,reader(sources,query,torch.tensor([.1,.2,.3,.4]))))

    def test_whole_string_unlikelihood_is_finite(self):
        x=torch.tensor([-100.,-1.,0.],requires_grad=True)
        loss=unlikelihood(x)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue((loss>=0).all())
        loss.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__=='__main__':
    unittest.main()
