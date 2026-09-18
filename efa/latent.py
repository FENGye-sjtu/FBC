"""Query-conditioned, independently acquired source latents for the paper reader.

The incumbent string is not an argument to SourceResampler.forward. Compression
weights require training; random initialization must not be used as a verifier.
"""
import torch
from torch import nn


class SourceResampler(nn.Module):
    def __init__(self,hidden_size,latent_tokens=8,heads=8):
        super().__init__()
        self.hidden_size=hidden_size
        self.latent_tokens=latent_tokens
        self.latents=nn.Parameter(torch.randn(latent_tokens,hidden_size)*0.02)
        self.coordinates=nn.Linear(4,hidden_size)
        self.query_norm=nn.LayerNorm(hidden_size)
        self.source_norm=nn.LayerNorm(hidden_size)
        self.attention=nn.MultiheadAttention(hidden_size,heads,batch_first=True)
        self.output=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,hidden_size))

    def forward(self,source_features,neutral_query_features,coordinates):
        # One crop per invocation. No cross-crop mixing before allowed-source
        # selection; no original-image thumbnail or main-trace hidden states.
        if source_features.ndim==2:
            source_features=source_features.unsqueeze(0)
        query=neutral_query_features.mean(dim=-2)
        if query.ndim==1:
            query=query.unsqueeze(0)
        coordinates=coordinates.reshape(-1,4)
        q=self.latents.unsqueeze(0)+query.unsqueeze(1)+self.coordinates(coordinates).unsqueeze(1)
        k=self.source_norm(source_features)
        attended,_=self.attention(self.query_norm(q),k,k,need_weights=False)
        return self.output(attended+q)


def unlikelihood(log_probability):
    """Whole-string unlikelihood, not normalized-log-score pseudo probability."""
    bounded=torch.clamp(log_probability,max=-torch.finfo(log_probability.dtype).eps)
    return -torch.log(-torch.expm1(bounded))
