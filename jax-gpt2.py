
# TODO: look into rngs
# TODO: figure out Transformer module and how to make a list of Blocks
# TODO: also check why HF implementation uses nn.Module --> it's because they import linen as nn; they are using linen instead of nnx

from dataclasses import dataclass
import jax
import jax.numpy as jnp     # instead of torch.Tensor equivalent
from flax import nnx        # pytorch equivalent
from functools import partial

class CausalSelfAttention(nnx.Module):
    def __init__(self, config, *, rngs:nnx.Rngs):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, rngs=rngs)
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, rngs=rngs)
        self.n_head, self.n_embd = config.n_head, config.n_embd
        # TODO: apply causal mask with a buffer

    def __call__(self, x):
        B, T, C = x.size()
        assert C == self.n_embd

        W_qkv = self.attn(x)
        q, k, v = jnp.split(W_qkv, 3, axis=-1) # NOTE: using jnp to manipulate matrices instead of torch.split
        # jnp.split(ary, indices_or_sections, axis=0) v/s ary.split(tensor, split_size_or_sections, dim=0)
        
        head_dim = self.n_embd // self.n_layer
        # Goal: to split the attention heads worth of information
        # using reshape to get the appropriate dimensions
        # transpose input params different
        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3) 
        # --> qkv matrices: (B,T,C) -> (B,T, n_head, head_dim) -> (B, n_head, T, head_dim)

        # Calculating attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        # TODO: apply mask
        attn_scores = F.softmax(attn_scores, dim=-1)
        y = attn_scores @ v
        y = y.transpose(0, 2, 1, 3).contiguous
        y = y.reshape(B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nnx.Module):
    # again, very literal translation from pyTorch to flax, just changed it to nnx
    def __init__(self, config, *, rngs:nnx.Rngs):
        super().__init__()
        self.c_fc = nnx.Linear(config.n_embd, config.n_embd * 4, rngs=rngs)
        self.gelu = nnx.gelu(x, approximate=True)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, rngs=rngs)
    
    def __call__(x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nnx.Module):
    # the only difference in this block is using nnx modules instead of nn modules --> this was very smooth
    def __init__(self, config, *, rngs:nnx.Rngs):
        super().__init__()
        self.ln_1 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x):
        x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x

class Transformer(nnx.Module):
    def __init__(self, config, *, rngs:nnx.Rngs):
        super().__init__()
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)   # flax requires a separate way to handle random initialized weights
        self.wpe = nnx.Embed(config.block_size, config.n_embd, rngs=rngs)
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]     # note that we don't have nn.ModuleList equivalent; using Python lists instead
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nnx.Module):
    def __init__(self, config, *, rngs:nnx.Rngs):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config, rngs=rngs) # flax nnx does not have nnx.ModuleDict equivalent --> making it into a class in order oto adhere to HF GPT2 implementation
        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, use_bias=False, rngs=rngs)

