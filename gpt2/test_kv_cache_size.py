"""
Test script to measure KV cache memory usage for different model
configurations by running inference and tracking cache sizes.
"""
import torch
import sys
import argparse
from train_ra_mla import GPT, GPTConfig

def measure_kv_cache(checkpoint_path, sequence_length=1024, batch_size=1):
    """Load model and measure KV cache size during inference."""
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
    elif 'model_args' in checkpoint:
        config_dict = checkpoint['model_args']
    else:
        print("Error: Could not find config in checkpoint")
        return
    
    print(f"\nModel configuration:")
    for key, val in config_dict.items():
        print(f"  {key}: {val}")
    
    # Create model
    config = GPTConfig(**config_dict)
    model = GPT(config)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # Create dummy input
    print(f"\nGenerating with sequence_length={sequence_length}, batch_size={batch_size}")
    idx = torch.randint(0, config.vocab_size, (batch_size, sequence_length))
    
    # Count KV cache parameters
    print("\n" + "="*60)
    print("KV CACHE ANALYSIS")
    print("="*60)
    
    n_layer = config.n_layer
    n_head = config.n_head
    n_embd = config.n_embd
    head_dim = n_embd // n_head
    
    print(f"\nArchitecture:")
    print(f"  Layers: {n_layer}")
    print(f"  Heads: {n_head}")
    print(f"  Embedding dim: {n_embd}")
    print(f"  Head dim: {head_dim}")
    
    # Standard GPT-2 KV cache
    standard_k_cache = batch_size * n_layer * n_head * sequence_length * head_dim
    standard_v_cache = batch_size * n_layer * n_head * sequence_length * head_dim
    standard_total = standard_k_cache + standard_v_cache
    
    print(f"\nStandard GPT-2 KV cache (no optimization):")
    print(f"  K cache: {standard_k_cache:,} elements")
    print(f"  V cache: {standard_v_cache:,} elements")
    print(f"  Total: {standard_total:,} elements")
    print(f"  Memory (fp16): {standard_total * 2 / (1024**2):.2f} MB")
    print(f"  Memory (fp32): {standard_total * 4 / (1024**2):.2f} MB")
    
    # Check for KV pruning config
    kv_prune = getattr(config, 'kv_cache_prune', False)
    kv_keep_ratio = getattr(config, 'kv_prune_keep_ratio', 1.0)
    
    if kv_prune:
        pruned_tokens = int(sequence_length * kv_keep_ratio)
        pruned_k_cache = batch_size * n_layer * n_head * pruned_tokens * head_dim
        pruned_v_cache = batch_size * n_layer * n_head * pruned_tokens * head_dim
        pruned_total = pruned_k_cache + pruned_v_cache
        
        print(f"\nWith KV pruning (keep_ratio={kv_keep_ratio}):")
        print(f"  Kept tokens: {pruned_tokens} / {sequence_length}")
        print(f"  K cache: {pruned_k_cache:,} elements")
        print(f"  V cache: {pruned_v_cache:,} elements")
        print(f"  Total: {pruned_total:,} elements")
        print(f"  Memory (fp16): {pruned_total * 2 / (1024**2):.2f} MB")
        print(f"  Memory (fp32): {pruned_total * 4 / (1024**2):.2f} MB")
        print(f"  Reduction: {(1 - pruned_total/standard_total)*100:.1f}%")
    
    # Check for KVSplice compression
    kvsplice_enable = getattr(config, 'kvsplice_enable', False)
    kvsplice_k = getattr(config, 'kvsplice_k', head_dim)
    
    if kvsplice_enable:
        compressed_v_cache = batch_size * n_layer * n_head * sequence_length * kvsplice_k
        compressed_k_cache = standard_k_cache  # K not compressed
        compressed_total = compressed_k_cache + compressed_v_cache
        
        print(f"\nWith KVSplice compression (k={kvsplice_k}):")
        print(f"  V compressed: {head_dim} → {kvsplice_k} dims")
        print(f"  K cache: {compressed_k_cache:,} elements (uncompressed)")
        print(f"  V cache: {compressed_v_cache:,} elements (compressed)")
        print(f"  Total: {compressed_total:,} elements")
        print(f"  Memory (fp16): {compressed_total * 2 / (1024**2):.2f} MB")
        print(f"  Memory (fp32): {compressed_total * 4 / (1024**2):.2f} MB")
        print(f"  Reduction: {(1 - compressed_total/standard_total)*100:.1f}%")
    
    # Combined: pruning + compression
    if kv_prune and kvsplice_enable:
        combined_k = batch_size * n_layer * n_head * pruned_tokens * head_dim
        combined_v = batch_size * n_layer * n_head * pruned_tokens * kvsplice_k
        combined_total = combined_k + combined_v
        
        print(f"\nWith both pruning + compression:")
        print(f"  Tokens: {sequence_length} → {pruned_tokens}")
        print(f"  V dims: {head_dim} → {kvsplice_k}")
        print(f"  K cache: {combined_k:,} elements")
        print(f"  V cache: {combined_v:,} elements")
        print(f"  Total: {combined_total:,} elements")
        print(f"  Memory (fp16): {combined_total * 2 / (1024**2):.2f} MB")
        print(f"  Memory (fp32): {combined_total * 4 / (1024**2):.2f} MB")
        print(f"  Reduction: {(1 - combined_total/standard_total)*100:.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()
    
    measure_kv_cache(args.checkpoint, args.seq_len, args.batch_size)
