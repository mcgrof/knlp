#!/usr/bin/env python3
"""
Semantic-Aware KV Calibration.

Collects KV activations grouped by semantic bucket and trains
separate PCA projectors for each content type.

Usage:
    python scripts/calibrate_semantic_kv.py --model Qwen/Qwen2.5-7B --rank 96
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from scripts.semantic_buckets import BUCKET_NAMES, bucket_tokens, get_bucketer


def get_calibration_texts() -> Dict[str, List[str]]:
    """
    Get calibration texts for each semantic bucket.

    Returns dict mapping bucket names to lists of representative texts.
    """
    texts = {
        "narrative": [
            "The sun was setting over the mountains, casting long shadows across the valley. "
            "Sarah walked slowly along the path, her footsteps crunching on the gravel. "
            "She had been traveling for days, and the journey had taken its toll. "
            "But as she crested the hill, the view that greeted her made it all worthwhile. "
            "Below, nestled in the valley, was the village she had been searching for.",
            "In the depths of the ancient forest, creatures stirred as darkness fell. "
            "The old oak trees stood like silent sentinels, their branches reaching toward the sky. "
            "A fox darted between the shadows, hunting for its evening meal. "
            "The moon rose slowly, painting everything in silver light. "
            "This was a world unchanged for centuries, untouched by modern hands.",
        ],
        "dialogue": [
            '"What do you think we should do?" asked Maria, her voice trembling slightly.\n'
            '"I think we need to act quickly," replied John. "We don\'t have much time."\n'
            '"But what if we\'re wrong?" she pressed.\n'
            '"Then we learn from it and try again," he said with a reassuring smile.\n'
            '"You always know what to say," she admitted.',
            "User: Can you help me understand machine learning?\n"
            "Assistant: Of course! Machine learning is a subset of artificial intelligence.\n"
            "User: What's the difference between supervised and unsupervised learning?\n"
            "Assistant: Great question! Supervised learning uses labeled data, while "
            "unsupervised learning finds patterns in unlabeled data.\n"
            "User: Can you give me an example?\n"
            "Assistant: Sure! Email spam detection is supervised, clustering customers is unsupervised.",
        ],
        "code": [
            "def calculate_fibonacci(n: int) -> int:\n"
            '    """Calculate the nth Fibonacci number recursively."""\n'
            "    if n <= 1:\n"
            "        return n\n"
            "    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)\n\n"
            "class DataProcessor:\n"
            "    def __init__(self, data: list):\n"
            "        self.data = data\n"
            "        self.processed = False\n\n"
            "    def process(self) -> list:\n"
            "        result = [x * 2 for x in self.data]\n"
            "        self.processed = True\n"
            "        return result",
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class TransformerBlock(nn.Module):\n"
            "    def __init__(self, d_model, n_heads):\n"
            "        super().__init__()\n"
            "        self.attn = nn.MultiheadAttention(d_model, n_heads)\n"
            "        self.ffn = nn.Sequential(\n"
            "            nn.Linear(d_model, 4 * d_model),\n"
            "            nn.GELU(),\n"
            "            nn.Linear(4 * d_model, d_model)\n"
            "        )\n"
            "        self.norm1 = nn.LayerNorm(d_model)\n"
            "        self.norm2 = nn.LayerNorm(d_model)",
        ],
        "math": [
            "Let's solve the equation: 3x + 5 = 20\n"
            "Subtracting 5 from both sides: 3x = 15\n"
            "Dividing by 3: x = 5\n\n"
            "For a quadratic equation ax² + bx + c = 0, the solution is:\n"
            "x = (-b ± √(b² - 4ac)) / 2a\n\n"
            "If a = 1, b = -5, c = 6, then:\n"
            "x = (5 ± √(25 - 24)) / 2 = (5 ± 1) / 2\n"
            "So x = 3 or x = 2",
            "Calculate the integral of x² from 0 to 3:\n"
            "∫₀³ x² dx = [x³/3]₀³ = 27/3 - 0 = 9\n\n"
            "The derivative of sin(x²) using chain rule:\n"
            "d/dx[sin(x²)] = cos(x²) · 2x = 2x·cos(x²)\n\n"
            "Probability: P(A|B) = P(B|A)·P(A) / P(B)",
        ],
        "reasoning": [
            "Let's think through this problem step by step.\n\n"
            "First, we need to identify the key constraints. The problem states that "
            "we have limited resources and multiple objectives.\n\n"
            "Therefore, we must prioritize. If we consider the long-term impact, "
            "option A provides better sustainability.\n\n"
            "However, option B offers faster short-term results. This creates a trade-off.\n\n"
            "Given the stakeholder requirements, we can conclude that a hybrid approach "
            "combining elements of both A and B would be optimal.",
            "The argument presented has several logical flaws.\n\n"
            "Premise 1: All birds can fly.\n"
            "Premise 2: Penguins are birds.\n"
            "Conclusion: Therefore, penguins can fly.\n\n"
            "This is a valid syllogism, but the first premise is false. "
            "Not all birds can fly - penguins, ostriches, and kiwis are counterexamples.\n\n"
            "Thus, while the logical structure is sound, the conclusion is false "
            "because it relies on a false premise. This demonstrates why we must "
            "verify our assumptions before drawing conclusions.",
        ],
        "instructions": [
            "How to Set Up Your Development Environment\n\n"
            "Step 1: Install Python 3.10 or later from python.org\n"
            "Step 2: Open your terminal and verify installation: python --version\n"
            "Step 3: Create a virtual environment: python -m venv myenv\n"
            "Step 4: Activate the environment:\n"
            "  - Windows: myenv\\Scripts\\activate\n"
            "  - Mac/Linux: source myenv/bin/activate\n"
            "Step 5: Install dependencies: pip install -r requirements.txt\n"
            "Step 6: Run the application: python main.py\n\n"
            "Note: If you encounter permission errors, try running as administrator.",
            "Quick Start Guide:\n\n"
            "1. Download the application from our website\n"
            "2. Extract the ZIP file to your preferred location\n"
            "3. Double-click setup.exe to begin installation\n"
            "4. Follow the on-screen prompts\n"
            "5. Select your preferred settings when asked\n"
            "6. Click 'Install' and wait for completion\n"
            "7. Launch the application from your Start menu\n\n"
            "Important: Make sure you have at least 2GB free disk space.\n"
            "Warning: Do not interrupt the installation process.",
        ],
    }

    return texts


def collect_kv_activations(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    max_tokens: int = 512,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Collect K and V activations for given texts.

    Returns (k_activations, v_activations) dicts mapping layer_idx to tensors.
    """
    k_activations = defaultdict(list)
    v_activations = defaultdict(list)

    # Hook to capture KV
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, output):
            # For most HF models, attention output includes (attn_output, attn_weights, past_kv)
            # We need to capture key_states and value_states from the input
            pass

        return hook

    # Alternative: Run forward pass and extract from cache
    model.eval()

    for text in tqdm(texts, desc="Collecting activations"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

            # Extract KV from past_key_values
            if hasattr(outputs, "past_key_values") and outputs.past_key_values:
                for layer_idx, (k, v) in enumerate(outputs.past_key_values):
                    # k, v shape: [batch, n_heads, seq_len, head_dim]
                    # Flatten to [batch * n_heads * seq_len, head_dim]
                    B, H, T, D = k.shape
                    k_flat = k.reshape(-1, D).cpu()
                    v_flat = v.reshape(-1, D).cpu()

                    k_activations[layer_idx].append(k_flat)
                    v_activations[layer_idx].append(v_flat)

    # Concatenate all activations per layer
    k_concat = {}
    v_concat = {}

    for layer_idx in k_activations:
        k_concat[layer_idx] = torch.cat(k_activations[layer_idx], dim=0)
        v_concat[layer_idx] = torch.cat(v_activations[layer_idx], dim=0)

    return k_concat, v_concat


def compute_pca(
    data: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PCA projection matrix.

    Returns (U, mean) where U is [d_input, rank] projection matrix.
    """
    # Center data
    mean = data.mean(dim=0)
    centered = data - mean

    # SVD
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Take top-k components
        projection = Vh[:rank, :].T  # [d_input, rank]
    except Exception:
        # Fallback to covariance-based PCA
        cov = centered.T @ centered / (data.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # Sort by eigenvalue descending
        idx = eigenvalues.argsort(descending=True)
        projection = eigenvectors[:, idx[:rank]]

    return projection, mean


def calibrate_semantic_kv(
    model_name: str,
    rank: int,
    device: str = "cuda",
    output_path: str = None,
) -> Dict:
    """
    Run semantic-aware KV calibration.

    Trains separate PCA projectors for each semantic bucket.
    """
    print(f"Semantic KV Calibration")
    print(f"  Model: {model_name}")
    print(f"  Rank: {rank}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    head_dim = config.hidden_size // config.num_attention_heads

    print(f"  Layers: {num_layers}")
    print(f"  Head dim: {head_dim}")

    # Get calibration texts
    calibration_texts = get_calibration_texts()

    # Collect activations per bucket
    bucket_k_activations = {}
    bucket_v_activations = {}

    for bucket in BUCKET_NAMES:
        print(f"\nCollecting activations for bucket: {bucket}")
        texts = calibration_texts.get(bucket, [])

        if not texts:
            print(f"  Warning: No texts for bucket {bucket}")
            continue

        k_acts, v_acts = collect_kv_activations(model, tokenizer, texts, device=device)

        bucket_k_activations[bucket] = k_acts
        bucket_v_activations[bucket] = v_acts

        print(f"  Collected {sum(v.shape[0] for v in k_acts.values())} K/V vectors")

    # Train PCA projectors per bucket per layer
    print("\nTraining PCA projectors...")

    calibration_data = {
        "model": model_name,
        "rank": rank,
        "head_dim": head_dim,
        "n_layers": num_layers,
        "n_heads": config.num_attention_heads,
        "buckets": {},
    }

    for bucket in BUCKET_NAMES:
        if bucket not in bucket_k_activations:
            continue

        print(f"\n  Bucket: {bucket}")
        bucket_data = {"layers": []}

        k_acts = bucket_k_activations[bucket]
        v_acts = bucket_v_activations[bucket]

        for layer_idx in range(num_layers):
            if layer_idx not in k_acts:
                continue

            k_data = k_acts[layer_idx].float()
            v_data = v_acts[layer_idx].float()

            # Compute PCA for K and V
            K_U, K_mean = compute_pca(k_data, rank)
            V_U, V_mean = compute_pca(v_data, rank)

            layer_data = {
                "K": {"U": K_U, "mean": K_mean},
                "V": {"U": V_U, "mean": V_mean},
            }
            bucket_data["layers"].append(layer_data)

        calibration_data["buckets"][bucket] = bucket_data
        print(f"    Trained {len(bucket_data['layers'])} layer projectors")

    # Also train a "universal" projector using all data
    print("\nTraining universal (baseline) projector...")

    all_k = defaultdict(list)
    all_v = defaultdict(list)

    for bucket in bucket_k_activations:
        for layer_idx, k in bucket_k_activations[bucket].items():
            all_k[layer_idx].append(k)
            all_v[layer_idx].append(bucket_v_activations[bucket][layer_idx])

    universal_layers = []
    for layer_idx in range(num_layers):
        if layer_idx not in all_k:
            continue

        k_data = torch.cat(all_k[layer_idx], dim=0).float()
        v_data = torch.cat(all_v[layer_idx], dim=0).float()

        K_U, K_mean = compute_pca(k_data, rank)
        V_U, V_mean = compute_pca(v_data, rank)

        universal_layers.append(
            {
                "K": {"U": K_U, "mean": K_mean},
                "V": {"U": V_U, "mean": V_mean},
            }
        )

    calibration_data["universal"] = {"layers": universal_layers}
    print(f"  Trained {len(universal_layers)} universal layer projectors")

    # Save
    if output_path is None:
        model_short = model_name.replace("/", "-").lower()
        output_path = f"kv_semantic_calib_{model_short}_r{rank}.pt"

    torch.save(calibration_data, output_path)
    print(f"\nSaved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Rank: {rank} (compression: {head_dim/rank:.2f}x)")
    print(f"Buckets trained: {list(calibration_data['buckets'].keys())}")
    print(f"Output: {output_path}")

    return calibration_data


def main():
    parser = argparse.ArgumentParser(description="Semantic-aware KV calibration")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to calibrate",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=96,
        help="Target rank for projection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path",
    )
    args = parser.parse_args()

    calibrate_semantic_kv(
        model_name=args.model,
        rank=args.rank,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
