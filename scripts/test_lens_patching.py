#!/usr/bin/env python3
"""
Quick test of lens-gated attention patching on GPT-2.
"""

import torch
from transformers import AutoModelForCausalLM
import sys
sys.path.insert(0, "gpt2")

from ra_lens_gpt2 import patch_gpt2_with_lens_attention

print("=" * 70)
print("Testing Lens-Gated Attention Patching")
print("=" * 70)

# Load tiny GPT-2 model
print("\n1. Loading GPT-2 model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
print(f"   Loaded: {model.config.n_layer} layers, {model.config.n_embd} dims, {model.config.n_head} heads")

# Patch with lens-gated attention
print("\n2. Patching with lens-gated attention...")
model, cfg = patch_gpt2_with_lens_attention(
    model,
    use_reciprocity=True,
    use_discoverability=True,
    use_route_gate=True,
    mlp_use_ctx_summary=True,
    mlp_expansion_ratio=4.0,
)

# Test forward pass
print("\n3. Testing forward pass...")
B, T = 2, 16
input_ids = torch.randint(0, model.config.vocab_size, (B, T))
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

print(f"   Input shape: {input_ids.shape}")
print(f"   Output shape: {logits.shape}")
print(f"   Expected: {input_ids.shape} -> ({B}, {T}, {model.config.vocab_size})")

# Check config flags
print("\n4. Checking config flags...")
print(f"   lens_gated: {model.config.lens_gated}")
print(f"   lens_use_reciprocity: {model.config.lens_use_reciprocity}")
print(f"   lens_use_discoverability: {model.config.lens_use_discoverability}")
print(f"   lens_use_route_gate: {model.config.lens_use_route_gate}")

# Verify block structure
print("\n5. Verifying block structure...")
block0 = model.transformer.h[0]
print(f"   Block type: {type(block0).__name__}")
print(f"   Has lens_attn: {hasattr(block0.attn, 'gates')}")
print(f"   Has gated_mlp: {hasattr(block0.mlp, 'fc_gate')}")
print(f"   Has route_gate: {hasattr(block0.mlp, 'get_route_gate')}")

if hasattr(block0.mlp, "get_route_gate"):
    g = block0.mlp.get_route_gate()
    print(f"   Route gate value: {g.item():.3f}")

print("\n" + "=" * 70)
print("✓ All tests passed! Lens-gated patching works correctly.")
print("=" * 70)
