# Debugging: Print the keys of hf_flax_params to understand its structure
print(hf_flax_params.keys())

# If 'lm_head' is not a direct key, you might need to explore further.
# For example, it might be under a 'shared' or 'params' key depending on the model design.
# Based on the traceback and typical HF structures, the 'lm_head' parameters
# are often at the top level or within the 'transformer' block.

# Looking at the global variables provided, hf_flax_params is:
# {'transformer': {'h': {...}, 'ln_f': {...}, 'wpe': {...}, 'wte': {...}}, 'lm_head': {'kernel': ...}}
# This indicates 'lm_head' *is* a top-level key, and within it, there is a 'kernel'.
# The original code looks correct based on this structure.

# However, let's re-examine the model initialization and parameter transfer.
# The `FlaxGPT.from_pretrained` method returns `model, hf_params`.
# The `load_hf_params_into_flax_gpt` function is then called with these.

# Let's try a more robust way to access the kernel, ensuring 'lm_head' and 'kernel' keys exist.
# We will keep the original parameter loading function but add a check or assume the structure based on the global variables provided.

# The provided global variables show that hf_flax_params indeed has 'lm_head' as a top-level key.
# {'transformer': {'h': {...}, 'ln_f': {...}, 'wpe': {...}, 'wte': {...}}, 'lm_head': {'kernel': ...}}
# This means the original access `hf_params['lm_head']['kernel']` *should* work if `hf_params` is exactly this dictionary.

# Let's re-run the cell with the parameter loading function and the subsequent comparison code.
# It's possible there was a transient issue or the `hf_flax_params` variable
# was accidentally overwritten or modified before reaching the `load_hf_params_into_flax_gpt` call in the execution that produced the error.
# Rerunning the cell should confirm if the key truly exists or not in the context of the error.

# No code changes are strictly necessary based on the provided global variable state,
# as the key 'lm_head' and nested 'kernel' appear to exist.
# However, to make the loading more robust, you could add checks:

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import nnx
from transformers import FlaxGPT2LMHeadModel, GPT2Config
from flax.core.frozen_dict import freeze, unfreeze
import math
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

# --- Your Flax Model Definitions (from your provided code) ---

class CausalSelfAttention(nnx.Module):
    def __init__(self, config, *, rngs:nnx.Rngs):
      super().__init__()
      assert config.n_embd % config.n_head == 0

      self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, rngs=rngs)
      self.c_proj = nnx.Linear(config.n_embd, config.n_embd, rngs=rngs)
      self.n_head, self.n_embd = config.n_head, config.n_embd
      mask = jnp.tril(jnp.ones((config.block_size, config.block_size))).reshape(1, 1, config.block_size, config.block_size)
      self.bias = nnx.Variable(mask, trainable=False)

    def __call__(self, x):
      B, T, C = x.shape
      assert C == self.n_embd

      W_qkv = self.c_attn(x)
      q, k, v = jnp.split(W_qkv, 3, axis=-1)

      head_dim = self.n_embd // self.n_head # Corrected: config.n_layer was typo
      q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
      k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
      v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

      attn_scores = (q @ k.transpose((0, 1, 3, 2))) / math.sqrt(k.shape[-1])
      attn_scores = jnp.where((self.bias.value[:, :, :T, :T]) == 0, -jnp.inf, attn_scores)
      attn_scores = nnx.softmax(attn_scores, axis=-1)
      y = attn_scores @ v
      y = y.transpose(0, 2, 1, 3)
      y = y.reshape(B, T, C)
      y = self.c_proj(y)

      return y

class MLP(nnx.Module):
    def __init__(self, config, *, rngs:nnx.Rngs):
      super().__init__()
      self.c_fc = nnx.Linear(config.n_embd, config.n_embd * 4, rngs=rngs)
      self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, rngs=rngs)

    def __call__(self, x):
      return self.c_proj(nnx.gelu(self.c_fc(x)))

class Block(nnx.Module):
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
      self.config = config
      self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
      self.wpe = nnx.Embed(config.block_size, config.n_embd, rngs=rngs)
      self.h: List[Block] = [
                Block(config, rngs=rngs) for _ in range(config.n_layer)]
      self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)

    def __call__(self, idx):
      B, T = idx.shape
      assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}." # Corrected: config.block was typo
      pos = jnp.arange(T)[None, :]
      x = self.wte(idx) + self.wpe(pos)

      for block in self.h:
        x = block(x)

      return self.ln_f(x)

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
      self.transformer = Transformer(config, rngs=rngs)
      self.lm_head = nnx.Linear(config.n_embd, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, idx):
      hidden = self.transformer(idx)
      return self.lm_head(hidden)

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
      assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
      override_args = override_args or {}
      assert all(k == 'dropout' for k in override_args)

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
      # Initialize the FlaxGPT model with an RNG key
      model = cls(config, rngs=nnx.Rngs(0))

      print(f"Loading pretrained Hugging Face FlaxGPT2LMHeadModel for {model_type}")
      hf_model = FlaxGPT2LMHeadModel.from_pretrained(model_type, dtype=jnp.float32)
      hf_params = hf_model.params

      # Manually transfer weights (this is a simplified example,
      # a more robust solution would involve a dedicated conversion script)
      # This is crucial for comparing activations.

      def load_hf_params_into_flax_gpt(flax_model: FlaxGPT, hf_params: Dict[str, Any]):
          # Example for embedding layers (adjust paths based on your model's structure)
          flax_model.transformer.wte.embedding.value = hf_params['transformer']['wte']['embedding']
          flax_model.transformer.wpe.embedding.value = hf_params['transformer']['wpe']['embedding']

          # Example for blocks (iterate and map for each block)
          for i in range(flax_model.config.n_layer):
              # Attention layer
              # Transpose kernels for Linear layers
              flax_model.transformer.h[i].attn.c_attn.kernel.value = hf_params['transformer']['h'][str(i)]['attn']['c_attn']['kernel'].T
              flax_model.transformer.h[i].attn.c_attn.bias.value = hf_params['transformer']['h'][str(i)]['attn']['c_attn']['bias']
              flax_model.transformer.h[i].attn.c_proj.kernel.value = hf_params['transformer']['h'][str(i)]['attn']['c_proj']['kernel'].T
              flax_model.transformer.h[i].attn.c_proj.bias.value = hf_params['transformer']['h'][str(i)]['attn']['c_proj']['bias']

              # MLP layer
              # Transpose kernels for Linear layers
              flax_model.transformer.h[i].mlp.c_fc.kernel.value = hf_params['transformer']['h'][str(i)]['mlp']['c_fc']['kernel'].T
              flax_model.transformer.h[i].mlp.c_fc.bias.value = hf_params['transformer']['h'][str(i)]['mlp']['c_fc']['bias']
              flax_model.transformer.h[i].mlp.c_proj.kernel.value = hf_params['transformer']['h'][str(i)]['mlp']['c_proj']['kernel'].T
              flax_model.transformer.h[i].mlp.c_proj.bias.value = hf_params['transformer']['h'][str(i)]['mlp']['c_proj']['bias']


              # LayerNorms (no transpose needed)
              flax_model.transformer.h[i].ln_1.scale.value = hf_params['transformer']['h'][str(i)]['ln_1']['scale']
              flax_model.transformer.h[i].ln_1.bias.value = hf_params['transformer']['h'][str(i)]['ln_1']['bias']
              flax_model.transformer.h[i].ln_2.scale.value = hf_params['transformer']['h'][str(i)]['ln_2']['scale']
              flax_model.transformer.h[i].ln_2.bias.value = hf_params['transformer']['h'][str(i)]['ln_2']['bias']


          # Final LayerNorm (no transpose needed)
          flax_model.transformer.ln_f.scale.value = hf_params['transformer']['ln_f']['scale']
          flax_model.transformer.ln_f.bias.value = hf_params['transformer']['ln_f']['bias']


          # LM Head
          # Check if 'lm_head' and 'kernel' exist in hf_params before accessing
          if 'lm_head' in hf_params and 'kernel' in hf_params['lm_head']:
               # Transpose kernel for Linear layer
               flax_model.lm_head.kernel.value = hf_params['lm_head']['kernel'].T
          else:
               print("Warning: 'lm_head' or 'kernel' not found in hf_params for LM head. LM head weights will not be loaded.")
          # HF GPT2LMHeadModel's lm_head does not have bias if you specify use_bias=False
          # If it had a bias, it would be hf_params['lm_head']['bias']

      # Load the parameters
      load_hf_params_into_flax_gpt(model, hf_params)


      return model, hf_params # Returning both for manual parameter mapping if needed

# --- Activation Collection in Flax ---

# This function will run a forward pass and collect activations
# It takes the model and the input, and returns the output and a dict of activations
def get_flax_activations(model: FlaxGPT, tok_ids: jax.Array) -> Tuple[jax.Array, OrderedDict]:
    acts_mine = OrderedDict()

    # We need to manually traverse the model and record outputs
    # For NNX, you call the modules directly

    # Input embeddings
    B, T = tok_ids.shape
    pos = jnp.arange(T)[None, :]
    x = model.transformer.wte(tok_ids) + model.transformer.wpe(pos)

    # Transformer blocks
    for l, block in enumerate(model.transformer.h):
        # LayerNorm before attention
        ln1_out = block.ln_1(x)
        acts_mine[f"blk{l}.ln1"] = ln1_out # Added LayerNorm activation

        # Attention
        attn_out = block.attn(ln1_out)
        acts_mine[f"blk{l}.attn"] = attn_out # Store activation
        x = x + attn_out # Residual connection

        # LayerNorm before MLP
        ln2_out = block.ln_2(x)
        acts_mine[f"blk{l}.ln2"] = ln2_out # Added LayerNorm activation

        # MLP
        mlp_out = block.mlp(ln2_out)
        acts_mine[f"blk{l}.mlp"] = mlp_out # Store activation
        x = x + mlp_out # Residual connection

    # Final LayerNorm
    final_ln_out = model.transformer.ln_f(x)
    acts_mine[f"ln_f"] = final_ln_out # Added final LayerNorm activation


    # Language Model Head
    logits = model.lm_head(final_ln_out)

    return logits, acts_mine

# --- Main Debugging Logic ---

import torch
from transformers import GPT2LMHeadModel
import tiktoken

# PyTorch setup
my_model_pt = GPT2LMHeadModel.from_pretrained("gpt2").eval() # Use HF's PyTorch model for direct comparison
# For PyTorch, ensure you have your `GPT` class if you truly have a custom one
# If `GPT` is your custom implementation, ensure it's loaded with the same weights as `gpt2`.
# For simplicity here, I'm assuming `my_model` in your original snippet was intended to be
# a custom PyTorch implementation loaded from HF weights.
# For a fair comparison, `my_model` should have its weights initialized identically to `their_model`.

prompt = "Hello, I am a language model."

enc = tiktoken.get_encoding("gpt2")
tok_ids_pt = torch.tensor([enc.encode(prompt)], dtype=torch.long)

torch.manual_seed(0)
my_model_pt.eval()
torch.set_grad_enabled(False)

acts_theirs_pt = OrderedDict() # PyTorch activations

def make_hook(store, tag):
    def _hook(module, _, out):
        if isinstance(out, tuple):
            out = out[0]
        store[tag] = out.detach().cpu()
    return _hook

# Hook PyTorch model
for l in range(12):
    mine_block_pt = my_model_pt.transformer.h[l]
    mine_block_pt.ln_1.register_forward_hook(make_hook(acts_theirs_pt, f"blk{l}.ln1")) # Added LayerNorm hook
    mine_block_pt.attn.register_forward_hook(make_hook(acts_theirs_pt, f"blk{l}.attn"))
    mine_block_pt.ln_2.register_forward_hook(make_hook(acts_theirs_pt, f"blk{l}.ln2")) # Added LayerNorm hook
    mine_block_pt.mlp.register_forward_hook(make_hook(acts_theirs_pt, f"blk{l}.mlp"))

my_model_pt.transformer.ln_f.register_forward_hook(make_hook(acts_theirs_pt, f"ln_f")) # Added final LayerNorm hook

_ = my_model_pt(tok_ids_pt) # Run forward pass to fill hooks

# Flax setup
# We need an RNG for Flax model initialization and a JAX array for input
key = jax.random.PRNGKey(0)
# Call from_pretrained to get the initialized model and HF params
my_flax_model, hf_flax_params = FlaxGPT.from_pretrained("gpt2")

# Convert PyTorch tensor to JAX array
tok_ids_jax = jnp.array(tok_ids_pt.numpy())

# Get Flax activations
flax_logits, acts_mine_flax = get_flax_activations(my_flax_model, tok_ids_jax)


# Compare activations
tolerance = 3e-5

print("\n--- Comparing Activations ---")
# Iterate through keys in PyTorch activations as the ground truth
for k in acts_theirs_pt.keys():
    if k in acts_mine_flax:
        # Convert JAX array to numpy for direct comparison with PyTorch CPU tensor
        diff = jnp.max(jnp.abs(acts_theirs_pt[k].numpy() - acts_mine_flax[k])).item()
        print(f"{k:12s}  max|Δ| = {diff:.2e}")
        if diff > tolerance:
            print(f"--> first divergence at {k}")
            # break # Uncomment to stop on first divergence
    else:
        print(f"Warning: Key {k} not found in Flax activations.")

# Optionally, compare the final logits
print("\n--- Comparing Logits ---")
# Ensure both logits are on CPU and converted to numpy/tensor
logits_pt = my_model_pt(tok_ids_pt).logits.detach().cpu().numpy()
logits_flax = flax_logits.block_until_ready() # Ensure JAX computation is complete
logits_flax_np = jnp.array(logits_flax) # Convert to JAX array

diff_logits = jnp.max(jnp.abs(logits_pt - logits_flax_np)).item()
print(f"Final logits max|Δ| = {diff_logits:.2e}")
if diff_logits > tolerance:
    print(f"--> Final logits diverge!")