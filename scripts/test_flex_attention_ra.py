#!/usr/bin/env python3
"""
Test if PyTorch FlexAttention can support RA pattern.
FlexAttention (PyTorch 2.5+) allows custom score modifications while keeping fusion.
"""
import torch
import torch.nn.functional as F

# Check if FlexAttention is available
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False
    print("⚠️  FlexAttention not available (requires PyTorch 2.5+)")
    print(f"   Current PyTorch version: {torch.__version__}")
    print()
    print("FlexAttention would enable:")
    print("  - Custom score modifications (RA, S^T, discoverability)")
    print("  - While keeping kernel fusion benefits")
    print("  - Potential 4.6x speedup over current implementation")
    print()
    print("To enable: pip install --upgrade torch>=2.5.0")
    exit(0)

def reciprocal_attention_score_mod(score, b, h, q_idx, kv_idx):
    """
    Score modification for Reciprocal Attention.

    FlexAttention calls this for each (q_idx, kv_idx) pair.
    We can add S^T and discoverability bias here while staying fused!

    Args:
        score: base attention score S[q_idx, kv_idx]
        b: batch index
        h: head index
        q_idx: query position
        kv_idx: key position

    Returns:
        modified score for softmax
    """
    # Example: Add reciprocity (S^T contribution)
    # In practice, would need to access S[kv_idx, q_idx] - not directly possible
    # This is the challenge: FlexAttention doesn't give access to full score matrix

    # Can only modify based on indices, not other scores
    # This is a fundamental limitation for RA

    return score  # Placeholder

if __name__ == "__main__":
    print("✅ FlexAttention is available!")
    print(f"   PyTorch version: {torch.__version__}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, T, D = 2, 12, 64, 64

    Q = torch.randn(B, H, T, D, device=device)
    K = torch.randn(B, H, T, D, device=device)
    V = torch.randn(B, H, T, D, device=device)

    # Test FlexAttention with custom score modification
    try:
        out = flex_attention(Q, K, V, score_mod=reciprocal_attention_score_mod)
        print("✅ FlexAttention works!")
        print()
        print("❌ BUT: RA requires S^T access")
        print("   - FlexAttention score_mod only gets (q_idx, kv_idx, score)")
        print("   - Cannot access S[kv_idx, q_idx] for reciprocity")
        print("   - Fundamental API limitation for RA pattern")
        print()
        print("Potential workarounds:")
        print("  1. Two-pass approach (not fused)")
        print("  2. Approximate S^T effect with local bias")
        print("  3. Request PyTorch API extension for matrix access")

    except Exception as e:
        print(f"❌ FlexAttention failed: {e}")

print()
print("=" * 70)
print("CONCLUSION:")
print("  FlexAttention: Not directly applicable to RA")
print("  RA needs full score matrix access (S^T)")
print("  FlexAttention only provides point-wise modification")
print()
print("Next option: Custom CUDA kernel or post-softmax approximation")
print("=" * 70)
