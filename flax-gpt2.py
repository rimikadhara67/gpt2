
# TODO: look into rngs
# TODO: figure out Transformer module and how to make a list of Blocks
# TODO: also check why HF implementation uses nn.Module --> it's because they import linen as nn; they are using linen instead of nnx

from dataclasses import dataclass
import jax
import jax.numpy as jnp     # instead of torch.Tensor equivalent
from flax import nnx        # pytorch equivalent
from transformers import FlaxGPT2LMHeadModel, GPT2Config
from flax.core.frozen_dict import freeze, unfreeze
import math


class CausalSelfAttention(nnx.Module):
    def __init__(self, config, *, rngs:nnx.Rngs):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, rngs=rngs)
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, rngs=rngs)
        self.n_head, self.n_embd = config.n_head, config.n_embd
        # apply causal mask with a buffer
        mask = jnp.tril(jnp.ones((config.block_size, config.block_size))).reshape(1, 1, config.block_size, config.block_size) # jnp.tril and jnp.ones instead of torch --> simply creates the desired mask and reshapes it
        self.bias = nnx.Variable(mask, trainable=False)

    def __call__(self, x):
        B, T, C = x.shape
        assert C == self.n_embd

        W_qkv = self.c_attn(x)
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
        attn_scores = (q @ k.transpose((0, 1, 3, 2))) / math.sqrt(k.size(-1))
        attn_scores = jnp.where((self.bias.value[:, :, :T, :T]) == 0, -jnp.inf, attn_scores)       # wherever there is a mask, set it to -inf
        attn_scores = nnx.softmax(attn_scores, dim=-1)
        y = attn_scores @ v
        y = y.transpose(0, 2, 1, 3) # apparently don't need .contiguous here because jax arrays are always contiguous
        y = y.reshape(B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nnx.Module):
    # again, very literal translation from pyTorch to flax, just changed it to nnx
    def __init__(self, config, *, rngs:nnx.Rngs):
        super().__init__()
        self.c_fc = nnx.Linear(config.n_embd, config.n_embd * 4, rngs=rngs)
        # self.gelu = nnx.gelu(x, approximate=True) --> need to apply directly since gelu is not a Module in flax it is a function
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, rngs=rngs)

    def __call__(self, x):
        return self.c_proj(nnx.gelu(self.c_fc(x)))

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

    # TODO: write a __call__ function

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class FlaxGPT(nnx.Module):
    def __init__(self, config, *, rngs:nnx.Rngs):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config, rngs=rngs) # flax nnx does not have nnx.ModuleDict equivalent --> making it into a class in order oto adhere to HF GPT2 implementation
        self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, use_bias=False, rngs=rngs)

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)

        # Setup configuration
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print("Forcing vocab_size=50257, block_size=1024")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        if 'dropout' in override_args:
            print(f"Overriding dropout to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        print(f"Creating GPT model with config: {config}")
        model = cls(config, rngs=nnx.Rngs(0))  # nnx models still need an rngs for init

        # Load pretrained Hugging Face Flax model
        print(f"Loading pretrained Hugging Face FlaxGPT2LMHeadModel for {model_type}")
        hf_model = FlaxGPT2LMHeadModel.from_pretrained(model_type, dtype=jnp.float32)
        hf_params = hf_model.params


        return model, hf_params  # Return the **model architecture** and the **pretrained params**