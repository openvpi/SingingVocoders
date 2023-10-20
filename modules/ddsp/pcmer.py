import torch

from torch import nn
import math
from functools import partial
from einops import rearrange, repeat

# from local_attention import LocalAttention
import torch.nn.functional as F
#import fast_transformers.causal_product.causal_product_cuda








class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""
    
    def __init__(self, 
                num_layers,
                num_heads,
                dim_model,
                dim_keys,
                dim_values,
                residual_dropout,
                attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout

        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])
        
    #  METHODS  ########################################################################################################
    
    def forward(self, phone, mask=None):
        
        # apply all layers to the input
        for (i, layer) in enumerate(self._layers):
            phone = layer(phone, mask)
        # provide the final sequence
        return phone


# ==================================================================================================================== #
#  CLASS  _ E N C O D E R  L A Y E R                                                                                   #
# ==================================================================================================================== #


class _EncoderLayer(nn.Module):
    """One layer of the encoder.
    
    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """
    
    def __init__(self, parent: PCmer):
        """Creates a new instance of ``_EncoderLayer``.
        
        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()
        
        
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        
        # selfatt -> fastatt: performer!
        self.attn = Attention(dim = parent.dim_model,
                                  heads = parent.num_heads,dim_head=32
                                  # causal = False
                              )
        
    #  METHODS  ########################################################################################################

    def forward(self, phone, mask=None):
        
        # compute attention sub-layer
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        
        phone = phone + (self.conformer(phone))
        
        return phone 

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            #nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)




class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, conditiondim=None):
        super().__init__()
        if conditiondim is None:
            conditiondim = dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(conditiondim, hidden_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(hidden_dim, dim, ),
                                    )

    def forward(self, q, kv=None, mask=None):
        # b, c, h, w = x.shape
        if kv is None:
            kv = q
        # q, kv = map(
        #     lambda t: rearrange(t, "b c t -> b t c", ), (q, kv)
        # )

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=2)

        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.heads), (q, k, v)
        )

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        with torch.backends.cuda.sdp_kernel(#enable_math=False
                                            ):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = rearrange(out, "b h t c -> b t (h c) ", h=self.heads, )
        return self.to_out(out)
