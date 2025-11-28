"""
GPT-2 Language Model implementation adapted from nanoGPT by Andrej Karpathy.
Original source: https://github.com/karpathy/nanoGPT

This implementation integrates with AdamWPrune for state-based pruning experiments.

References:
1) Original nanoGPT: https://github.com/karpathy/nanoGPT
2) GPT-2 paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
3) Official GPT-2 TensorFlow implementation: https://github.com/openai/gpt-2
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

def compute_fisher_metrics(
    attn_probs: torch.Tensor,
    layer_idx: int,
    n_samples: int = 64,
    topk: int = 8,
) -> dict:
    """
    Compute Fisher Information Matrix spectrum metrics for attention.

    The FIM eigenvalues reveal the curvature geometry of attention (SPDA paper).
    Lower eigmax and better conditioning indicate smoother optimization.

    Args:
        attn_probs: [B, H, T, T] attention probabilities (softmax outputs)
        layer_idx: Layer index for metric naming
        n_samples: Number of samples per head for eigenvalue computation
        topk: Number of top eigenvalues to log

    Returns:
        Dictionary of Fisher metrics for W&B logging
    """
    B, H, T, _ = attn_probs.shape
    device = attn_probs.device

    all_metrics = {}

    for h in range(H):
        # Extract single head: [B, T, T]
        attn_h = attn_probs[:, h]

        # Reshape to [B*T_q, T_k] - treat each query's distribution as sample
        p = attn_h.reshape(B * T, T)

        # Subsample queries to reduce cost (eigendecomp is O(T^3))
        if p.size(0) > n_samples:
            idx = torch.randperm(p.size(0), device=device)[:n_samples]
            p = p[idx]  # [N, T]
        N = p.size(0)

        # Compute average Fisher: F = mean_i (diag(p_i) - p_i p_i^T)
        F = torch.zeros(T, T, device=device)
        for i in range(N):
            pi = p[i]  # [T]
            F += torch.diag(pi) - torch.outer(pi, pi)
        F /= N

        # Compute eigenvalues (symmetric PSD matrix)
        try:
            eigvals = torch.linalg.eigvalsh(F)  # [T], sorted ascending
            eigvals = eigvals.to("cpu")

            eigmax = float(eigvals[-1])
            trace = float(eigvals.sum())
            cond = float(eigvals[-1] / (eigvals[0].abs() + 1e-8))

            # Aggregate metrics across heads
            prefix = f"fisher/layer{layer_idx}/head{h}"
            all_metrics[f"{prefix}/eigmax"] = eigmax
            all_metrics[f"{prefix}/trace"] = trace
            all_metrics[f"{prefix}/cond"] = cond

            # Top-k eigenvalues
            topk_vals = eigvals[-topk:]
            for i, v in enumerate(topk_vals):
                all_metrics[f"{prefix}/eig_top{i}"] = float(v)

            # Energy concentration in top-r modes
            total_energy = eigvals.sum()
            if total_energy > 1e-8:
                for r in [8, 16]:
                    if r <= len(eigvals):
                        energy_r = eigvals[-r:].sum()
                        all_metrics[f"{prefix}/energy_r{r}"] = float(
                            energy_r / total_energy
                        )

        except Exception as e:
            # Eigendecomposition can fail for ill-conditioned matrices
            print(
                f"  Warning: Fisher eigendecomp failed for layer {layer_idx} head {h}: {e}"
            )

    # Compute layer-level aggregates
    if all_metrics:
        eigmax_vals = [
            v for k, v in all_metrics.items() if "eigmax" in k and "layer" in k
        ]
        if eigmax_vals:
            all_metrics[f"fisher/layer{layer_idx}/eigmax_mean"] = sum(
                eigmax_vals
            ) / len(eigmax_vals)
            all_metrics[f"fisher/layer{layer_idx}/eigmax_max"] = max(eigmax_vals)

    return all_metrics


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention support"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, mechint_kv_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Apply mechint KV masks if provided (for circuit discovery)
        if mechint_kv_mask is not None:
            k, v, _ = mechint_kv_mask(k, v)

        # Flash Attention: efficient fused attention kernel
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """MLP with GELU activation"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    # Model architecture
    block_size: int = 1024  # maximum sequence length
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    )
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension
    dropout: float = 0.0  # dropout rate
    bias: bool = True  # use bias in Linear and LayerNorm

    # Model size presets
    @classmethod
    def from_name(cls, name: str):
        """Create config from model name (gpt2-tiny, gpt2, gpt2-medium, gpt2-large, gpt2-xl)"""
        # GPT-2 model configurations
        configs = {
            "gpt2-tiny": dict(
                n_layer=6, n_head=8, n_embd=512
            ),  # ~22M params (for double-attn comparison)
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }
        if name not in configs:
            raise ValueError(
                f"Unknown model name: {name}. Choose from {list(configs.keys())}"
            )
        return cls(**configs[name])


class GPT2(nn.Module):
    """GPT Language Model"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: standard practice in modern LMs
        # Shares token embedding matrix with output projection to reduce parameters
        # and improve generalization (Press & Wolf 2016: https://arxiv.org/pdf/1608.05859)
        self.transformer.wte.weight = self.lm_head.weight
        assert self.transformer.wte.weight is self.lm_head.weight, "Weight tying failed"

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with proper weight decay handling.
        This method separates parameters into those that should have weight decay
        and those that shouldn't (biases and layer norms).
        """
        # separate out all parameters to those that will and won't experience weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # Special handling for weight-tied parameters
        # When weight tying is enabled, lm_head.weight shares the same tensor as transformer.wte.weight
        # Remove lm_head.weight from decay set if it doesn't exist as a separate parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Remove parameters that don't exist due to weight tying
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}

        # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # use fused AdamW if available
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"Using {'fused' if use_fused else 'regular'} AdamW")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )
        return optimizer

    @torch.no_grad()
    def compute_fisher_metrics(
        self,
        x: torch.Tensor,
        layer_indices: list = None,
        n_samples: int = 64,
        topk: int = 8,
    ) -> dict:
        """
        Compute Fisher Information Matrix spectrum metrics for selected layers.

        Args:
            x: Input tensor [B, T]
            layer_indices: Which layers to analyze (default: [0, n_layers//2, -1])
            n_samples: Samples per head for eigenvalue computation
            topk: Number of top eigenvalues to log

        Returns:
            Dictionary of Fisher metrics for W&B logging
        """
        if compute_fisher_metrics is None:
            return {}

        if layer_indices is None:
            n = len(self.transformer.h)
            layer_indices = [0, n // 2, n - 1]

        B, T = x.shape
        device = x.device

        # Forward pass to capture attention probabilities
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(torch.arange(T, device=device))
        h = self.transformer.drop(tok_emb + pos_emb)

        all_metrics = {}

        for i, block in enumerate(self.transformer.h):
            if i in layer_indices:
                attn_probs = self._get_attn_probs(block.attn, block.ln_1(h))
                if attn_probs is not None:
                    metrics = compute_fisher_metrics(
                        attn_probs, i, n_samples=n_samples, topk=topk
                    )
                    all_metrics.update(metrics)

            h = block(h)

        return all_metrics

    def _get_attn_probs(
        self, attn: "CausalSelfAttention", x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Extract attention probabilities from CausalSelfAttention layer."""
        B, T, C = x.shape
        device = x.device

        # Get Q, K, V
        q, k, v = attn.c_attn(x).split(self.config.n_embd, dim=2)

        # Reshape to [B, H, T, head_dim]
        q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)
        k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )
        att = att.masked_fill(causal_mask, float("-inf"))

        # Softmax to get probabilities
        att = F.softmax(att, dim=-1)

        return att  # [B, H, T, T]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text by sampling from the model.
        idx is (B, T) array of indices in the current context
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
