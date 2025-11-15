# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ======================== SinkGD Optimizer ========================


class SinkGD(Optimizer):
    """
    Sinkhorn-like Gradient Descent (SinkGD).

    Applies entropic optimal transport-inspired normalization to gradients
    before parameter updates. The gradient transform uses iterative
    row/column normalization with temperature scaling to encourage
    structured, balanced updates.

    This can improve optimization geometry in settings where vanilla
    adaptive methods (Adam/AdamW) exhibit noisy or over-adaptive behavior.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 3e-4)
        weight_decay: Decoupled weight decay (L2 penalty) (default: 0.1)
        tau: Temperature for Sinkhorn-like smoothing (default: 0.1)
        n_iter: Number of normalization iterations (default: 5)
        eps: Numerical stability epsilon (default: 1e-8)

    Reference: Based on Sinkhorn gradient transport concepts for
    balanced optimization dynamics.
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        weight_decay=0.1,
        tau=0.1,
        n_iter=5,
        eps=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 < tau:
            raise ValueError(f"Invalid tau: {tau}")
        if not 0 <= n_iter:
            raise ValueError(f"Invalid n_iter: {n_iter}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(
            lr=lr, weight_decay=weight_decay, tau=tau, n_iter=n_iter, eps=eps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            tau = group["tau"]
            n_iter = int(group["n_iter"])
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # Decoupled weight decay (AdamW-style)
                if wd != 0.0:
                    p.add_(p, alpha=-wd * lr)

                # Sinkhorn-like gradient transform
                g_tilde = _sinkhorn_like_transform(g, tau=tau, n_iter=n_iter, eps=eps)

                # Parameter update
                p.add_(g_tilde, alpha=-lr)

        return loss


def _sinkhorn_like_transform(g, tau=0.1, n_iter=5, eps=1e-8):
    """
    Apply Sinkhorn-inspired iterative normalization to gradient tensor.

    This performs alternating row/column normalization with temperature
    scaling to produce a balanced, entropy-regularized gradient.

    Args:
        g: Gradient tensor (any shape)
        tau: Temperature for smoothing (smaller = sharper)
        n_iter: Number of normalization iterations
        eps: Numerical stability constant

    Returns:
        Transformed gradient in original dtype
    """
    if n_iter == 0:
        # Degenerate case: just return normalized gradient
        return g / (g.abs().mean() + eps)

    orig_dtype = g.dtype
    orig_device = g.device

    # Work in fp32 for numerical stability
    v = g.detach().float()

    # Handle scalar and 1D tensors (no row/col structure)
    if v.ndim < 2:
        # Simple normalization for 1D
        v = v / (v.abs().sum().clamp_min(eps))
        v = torch.tanh(v / max(eps, tau))
        return v.to(orig_dtype)

    # For 2D+ tensors: iterative row/col normalization
    # Treat last dim as "columns", second-to-last as "rows"
    for _ in range(n_iter):
        # Row normalization (normalize across last dimension)
        row_norm = v.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
        v = v / row_norm

        # Column normalization (normalize across second-to-last dimension)
        if v.ndim >= 2:
            col_norm = v.abs().sum(dim=-2, keepdim=True).clamp_min(eps)
            v = v / col_norm

        # Temperature-based smoothing (bounded activation)
        v = torch.tanh(v / max(eps, tau))

    # Restore scale (optional: preserve gradient magnitude)
    # Here we use a simple rescaling to match original RMS
    orig_rms = (g.pow(2).mean().sqrt() + eps).item()
    curr_rms = (v.pow(2).mean().sqrt() + eps).item()
    v = v * (orig_rms / curr_rms)

    return v.to(dtype=orig_dtype, device=orig_device)


# ======================== Utility Functions ========================


def param_groups_for_weight_decay(model, model_type="resnet"):
    """
    Separate model parameters into groups with and without weight decay.
    Bias and normalization parameters should not have weight decay.

    Args:
        model: The model to get parameters from
        model_type: Type of model ("lenet", "resnet", "gpt2", etc.) for appropriate decay values

    Returns:
        List of parameter groups with appropriate weight decay settings
    """
    # For GPT2, use its own configure_optimizers method if available
    if model_type == "gpt2" and hasattr(model, "configure_optimizers"):
        # GPT2 has its own parameter grouping logic for weight decay
        # Just return all parameters as a single group to avoid the duplicate warning
        # The weight decay will be applied uniformly
        return [{"params": model.parameters(), "weight_decay": 0.0}]

    decay, no_decay = [], []
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            full = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias") or isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):
                no_decay.append(p)
            else:
                decay.append(p)

    # Default weight decay values per model type
    if model_type == "lenet":
        default_wd = 1e-4
    elif model_type == "gpt2":
        default_wd = 0.1  # GPT2 uses higher weight decay
    else:  # resnet18, resnet50, etc.
        default_wd = 5e-4

    return [
        {"params": decay, "weight_decay": default_wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def resolve_weight_decay(optimizer_name, user_wd=None, model_type="resnet"):
    """
    Resolve weight decay value based on optimizer and user input.

    Args:
        optimizer_name: Name of the optimizer
        user_wd: User-specified weight decay (overrides defaults)
        model_type: Type of model for appropriate defaults

    Returns:
        Appropriate weight decay value
    """
    if user_wd is not None:
        return user_wd

    # Model-specific defaults
    if model_type == "lenet":
        # LeNet-5 defaults (smaller model, needs less regularization)
        if optimizer_name == "sgd":
            return 1e-4
        elif optimizer_name in ["adamw", "adamwadv", "adamwspam", "adamwprune"]:
            return 1e-4
        elif optimizer_name == "adam":
            return 0.0  # Coupled L2 with Adam is rarely helpful
    elif model_type == "gpt2":
        # GPT-2 defaults (transformer model, uses higher weight decay)
        if optimizer_name == "sgd":
            return 0.1
        elif optimizer_name in ["adamw", "adamwadv", "adamwspam", "adamwprune"]:
            return 0.1
        elif optimizer_name == "adam":
            return 0.0  # Coupled L2 with Adam is rarely helpful
    else:
        # ResNet/CIFAR defaults
        if optimizer_name == "sgd":
            return 5e-4
        elif optimizer_name in ["adamw", "adamwadv", "adamwspam", "adamwprune"]:
            return 5e-4  # 1e-4 to 1e-3 also fine; 1e-2 usually too strong
        elif optimizer_name == "adam":
            return 0.0  # Coupled L2 with Adam is rarely helpful

    return 0.0


def create_optimizer(
    model,
    optimizer_type,
    learning_rate,
    num_epochs=None,
    args=None,
    model_type="resnet",
):
    """
    Create an optimizer based on the specified type.

    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer (sgd, adam, adamw, adamwadv, adamwspam, adamwprune)
        learning_rate: Learning rate for the optimizer
        num_epochs: Number of epochs (needed for some schedulers)
        args: Additional arguments containing SPAM parameters
        model_type: Type of model ("lenet", "resnet") for appropriate defaults

    Returns:
        optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state
    """
    scheduler = None
    gradient_clip_norm = None
    spam_state = None
    adamprune_state = None

    # Get user-specified weight decay if provided
    user_wd = getattr(args, "weight_decay", None) if args else None

    # Resolve the appropriate weight decay value
    weight_decay = resolve_weight_decay(optimizer_type, user_wd, model_type)

    # Get parameter groups (with and without weight decay)
    param_groups = param_groups_for_weight_decay(model, model_type)

    # Apply the resolved weight decay to the decay group
    for group in param_groups:
        if group["weight_decay"] != 0:
            group["weight_decay"] = weight_decay

    if optimizer_type == "adam":
        # Adam typically doesn't benefit from weight decay (coupled L2 regularization)
        optimizer = torch.optim.Adam(param_groups, lr=learning_rate)
        logger.info(f"Using Adam optimizer (no weight decay by default)")

    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
        logger.info(f"Using AdamW optimizer with weight decay={weight_decay}")
        print(f"AdamW weight decay (resolved): {weight_decay}", flush=True)

    elif optimizer_type == "adamwadv":
        # AdamW Advanced with all enhancements
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True,  # AMSGrad variant for better convergence
        )
        # Cosine annealing learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        # Gradient clipping for stability
        gradient_clip_norm = 1.0
        logger.info("Using AdamW Advanced optimizer with:")
        logger.info("  - AMSGrad enabled for better convergence")
        logger.info("  - Weight decay=0.01 for stronger regularization")
        logger.info("  - Cosine annealing LR schedule")
        logger.info("  - Gradient clipping norm=1.0")

    elif optimizer_type == "adamwspam":
        # AdamW with SPAM (Spike-Aware Pruning-Adaptive Momentum)
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True,
        )
        # Initialize SPAM state tracking
        spam_state = {
            "gradient_history": deque(maxlen=100),
            "gradient_norms": deque(maxlen=100),
            "spike_threshold": 2.0,  # Detect spikes > 2 std deviations
            "spike_events": [],
            "last_reset_step": 0,
            "global_step": 0,
            "momentum_reset_count": 0,
            # SPAM fidelity options
            "theta": args.spam_theta if args else 50.0,
            "interval": args.spam_interval if args else 0,
            "warmup_steps": args.spam_warmup_steps if args else 0,
            "enable_clip": args.spam_enable_clip if args else False,
            "warmup_until": 0,
        }
        # Cosine annealing with SPAM awareness
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        gradient_clip_norm = 1.0
        logger.info("Using AdamW SPAM optimizer with:")
        logger.info("  - Spike detection (threshold=2.0 std)")
        logger.info("  - Automatic momentum reset on spikes")
        logger.info("  - AMSGrad for stability")
        logger.info("  - Adaptive gradient clipping")
        logger.info("  - Cosine annealing LR schedule")
        if spam_state["interval"] > 0:
            logger.info(
                f"  - Periodic momentum reset interval: {spam_state['interval']} steps"
            )
            if spam_state["warmup_steps"] > 0:
                logger.info(
                    f"  - Cosine warmup after reset: {spam_state['warmup_steps']} steps"
                )
        if spam_state["enable_clip"]:
            logger.info(
                f"  - Spike-aware clipping enabled (theta={spam_state['theta']})"
            )

    elif optimizer_type == "adamwprune":
        # AdamWPrune - Augments any Adam variant with state-based pruning

        # Get base optimizer name from config
        base_optimizer_name = (
            getattr(args, "adamwprune_base_optimizer_name", "adamw")
            if args
            else "adamw"
        )

        # Override hyperparameters if specified for AdamWPrune
        beta1 = float(getattr(args, "adamwprune_beta1", 0.9) if args else 0.9)
        beta2 = float(getattr(args, "adamwprune_beta2", 0.999) if args else 0.999)
        weight_decay = float(
            getattr(args, "adamwprune_weight_decay", 0.0) if args else 0.0
        )
        amsgrad = bool(getattr(args, "adamwprune_amsgrad", False) if args else False)

        logger.info(f"Creating AdamWPrune with base optimizer: {base_optimizer_name}")

        # Recursively create the base optimizer with overridden parameters
        if base_optimizer_name in ["adam", "adamw", "adamwadv", "adamwspam"]:
            # Create a modified args for the base optimizer
            base_args = None
            if args:

                class BaseArgs:
                    pass

                base_args = BaseArgs()
                # Copy all attributes from original args
                for attr in dir(args):
                    if not attr.startswith("_"):
                        setattr(base_args, attr, getattr(args, attr))
                # Override with AdamWPrune-specific values
                if base_optimizer_name == "adamwspam":
                    # Keep SPAM settings from original args
                    pass
                elif base_optimizer_name == "adamwadv":
                    # Disable SPAM for AdamWAdv base
                    base_args.spam_interval = 0
                    base_args.spam_enable_clip = False

            # Create base optimizer recursively
            base_optimizer_tuple = create_optimizer(
                model,
                base_optimizer_name,
                learning_rate,
                num_epochs,
                base_args,
                model_type,
            )
            optimizer = base_optimizer_tuple[0]
            scheduler = (
                base_optimizer_tuple[1] if len(base_optimizer_tuple) > 1 else None
            )
            gradient_clip_norm = (
                base_optimizer_tuple[2] if len(base_optimizer_tuple) > 2 else None
            )
            spam_state = (
                base_optimizer_tuple[3] if len(base_optimizer_tuple) > 3 else None
            )

            # Override optimizer parameters with AdamWPrune-specific values
            for group in optimizer.param_groups:
                group["betas"] = (beta1, beta2)
                group["weight_decay"] = weight_decay
                # Only set amsgrad if it exists in the parameter group (Adam/AdamW support it)
                if "amsgrad" in group:
                    group["amsgrad"] = amsgrad
        else:
            # Fallback to creating AdamW directly with proper parameter groups
            # Note: For AdamWPrune fallback, we use the resolved weight decay
            for group in param_groups:
                group["betas"] = (beta1, beta2)
                group["eps"] = 1e-8
                group["amsgrad"] = amsgrad
                # Keep the weight decay from param_groups setup
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=learning_rate,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
            gradient_clip_norm = 1.0
            spam_state = None

        logger.info(f"  - Base: {base_optimizer_name}")
        logger.info(f"  - Beta1: {beta1}, Beta2: {beta2}")
        logger.info(f"  - Weight decay: {weight_decay}")
        logger.info(f"  - AMSGrad: {amsgrad}")

        # AdamWPrune specific state for state-based pruning
        adamwprune_enable_pruning = bool(
            getattr(args, "adamwprune_enable_pruning", False) if args else False
        )
        adamwprune_pruning_method = "state" if adamwprune_enable_pruning else "none"
        adamwprune_target_sparsity = (
            float(getattr(args, "adamwprune_target_sparsity", 0.7) if args else 0.7)
            if adamwprune_enable_pruning
            else 0.0
        )

        # Only create adamprune_state if pruning is enabled
        if adamwprune_enable_pruning:
            adamprune_state = {
                "enabled": True,  # Main flag for pruning activation
                "pruning_enabled": True,  # Legacy compatibility
                "target_sparsity": adamwprune_target_sparsity,
                "warmup_steps": (
                    getattr(args, "adamwprune_warmup_steps", 100) if args else 100
                ),
                "pruning_frequency": (
                    getattr(args, "adamwprune_frequency", 50) if args else 50
                ),
                "ramp_end_epoch": (
                    getattr(args, "adamwprune_ramp_end_epoch", 75) if args else 75
                ),
                "ramp_end_step": (
                    getattr(args, "adamwprune_ramp_end_step", None) if args else None
                ),
                "step_count": 0,
                "masks": {},  # module -> bool mask buffer
                "pruning_strategy": "hybrid",  # hybrid of momentum and stability
                "variant": (
                    getattr(args, "adamwprune_variant", "bitter0")
                    if args
                    else "bitter0"
                ),  # bitter0=original, bitter1=magnitude, bitter2=scale-aware, bitter3=gradient-magnitude,
                # bitter4=layer-adaptive, bitter5=movement-to-zero, bitter6=coherence-weighted,
                # bitter7=second-moment, bitter8=bias-corrected, bitter9=hybrid
            }

            # Initialize masks for prunable layers as boolean buffers
            # Also build layer index mapping for adaptive sparsity
            layer_index = {}
            layer_count = 0
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    mask = torch.ones_like(module.weight.data, dtype=torch.bool)
                    module.register_buffer("adamprune_mask", mask)
                    adamprune_state["masks"][module] = module.adamprune_mask
                    layer_index[module] = layer_count
                    layer_count += 1

            adamprune_state["layer_index"] = layer_index
            adamprune_state["total_layers"] = layer_count
            logger.info(
                f"  - Pruning: Enabled (target: {adamprune_state['target_sparsity']:.1%})"
            )
            logger.info(f"  - Pruning warmup: {adamprune_state['warmup_steps']} steps")
            logger.info(f"  - Pruning variant: {adamprune_state['variant']}")
        else:
            adamprune_state = None
            logger.info("  - Pruning: Disabled")

    elif optimizer_type == "sinkgd":
        # SinkGD optimizer with Sinkhorn-like gradient transport
        sinkgd_lr = getattr(args, "sinkgd_lr", None) if args else None
        sinkgd_lr = sinkgd_lr if sinkgd_lr is not None else learning_rate
        sinkgd_wd = getattr(args, "sinkgd_weight_decay", None) if args else None
        sinkgd_wd = sinkgd_wd if sinkgd_wd is not None else weight_decay
        sinkgd_tau = getattr(args, "sinkgd_tau", 0.1) if args else 0.1
        sinkgd_iters = getattr(args, "sinkgd_iters", 5) if args else 5
        sinkgd_eps = getattr(args, "sinkgd_eps", 1e-8) if args else 1e-8

        optimizer = SinkGD(
            model.parameters(),
            lr=sinkgd_lr,
            weight_decay=sinkgd_wd,
            tau=sinkgd_tau,
            n_iter=sinkgd_iters,
            eps=sinkgd_eps,
        )
        logger.info("Using SinkGD optimizer with:")
        logger.info(f"  - Learning rate: {sinkgd_lr}")
        logger.info(f"  - Weight decay: {sinkgd_wd}")
        logger.info(f"  - Temperature (tau): {sinkgd_tau}")
        logger.info(f"  - Sinkhorn iterations: {sinkgd_iters}")
        logger.info(f"  - Epsilon: {sinkgd_eps}")

    else:  # Default to SGD
        optimizer = torch.optim.SGD(param_groups, lr=learning_rate, momentum=0.9)
        logger.info(
            f"Using SGD optimizer with momentum=0.9, weight_decay={weight_decay}"
        )

    return optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state


def apply_spam_gradient_processing(
    optimizer,
    model,
    spam_state,
    gradient_clip_norm,
):
    """Apply SPAM-specific gradient processing including spike detection and clipping."""
    if spam_state is None:
        return

    # SPAM-inspired spike-aware clipping using Adam's second moment
    if spam_state.get("enable_clip", False):
        theta = float(spam_state.get("theta", 50.0))
        eps = 1e-12
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = optimizer.state.get(p, {})
                v = state.get("exp_avg_sq", None)
                if v is None:
                    continue
                # threshold per-parameter: sqrt(theta * v)
                thr = (theta * (v + eps)).sqrt()
                g = p.grad
                # clip per-parameter exceeding threshold
                over = g.abs() > thr
                if over.any():
                    g.data[over] = g.data[over].sign() * thr.data[over]

    # Compute gradient norm for spike detection
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    # Track gradient norms (only if finite)
    if np.isfinite(total_norm):
        spam_state["gradient_norms"].append(total_norm)
    spam_state["global_step"] += 1

    # Detect spikes after warmup period
    if len(spam_state["gradient_norms"]) >= 10:
        history = np.array(list(spam_state["gradient_norms"])[:-1])
        # Filter out NaN and infinite values
        history = history[np.isfinite(history)]

        if len(history) > 0:
            mean_norm = np.mean(history)
            std_norm = np.std(history)

            if std_norm > 0 and np.isfinite(mean_norm) and np.isfinite(std_norm):
                z_score = (total_norm - mean_norm) / std_norm

                # Spike detected - reset momentum
                if z_score > spam_state["spike_threshold"] and np.isfinite(z_score):
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            state = optimizer.state.get(p, {})
                            if "exp_avg" in state:
                                state["exp_avg"].mul_(0.5)  # Soft reset
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"].mul_(0.9)  # Gentle reset

                    spam_state["spike_events"].append(spam_state["global_step"])
                    spam_state["momentum_reset_count"] += 1
                    spam_state["last_reset_step"] = spam_state["global_step"]

                    logger.debug(
                        f"SPAM: Spike detected (z={z_score:.2f}), momentum reset #{spam_state['momentum_reset_count']}"
                    )


def apply_periodic_spam_reset(optimizer, spam_state):
    """Apply periodic SPAM momentum reset with optional warmup."""
    if spam_state is None:
        return

    spam_state["global_step"] += 1
    interval = int(spam_state.get("interval", 0))

    if interval > 0 and spam_state["global_step"] % interval == 0:
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
        spam_state["momentum_reset_count"] += 1
        spam_state["last_reset_step"] = spam_state["global_step"]
        if int(spam_state.get("warmup_steps", 0)) > 0:
            spam_state["warmup_until"] = spam_state["global_step"] + int(
                spam_state["warmup_steps"]
            )

    # Apply cosine warmup scaling to param group lrs if within warmup window
    warmup_until = int(spam_state.get("warmup_until", 0))
    if warmup_until > spam_state["global_step"]:
        remaining = warmup_until - spam_state["global_step"]
        total = int(spam_state.get("warmup_steps", 0))
        # cosine from small to 1.0
        t = (total - remaining) / max(total, 1)
        scale = 0.5 * (1 - np.cos(np.pi * t))  # 0 -> 1
        scale = max(1e-3, float(scale))
        for group in optimizer.param_groups:
            base_lr = group.get("base_lr", group["lr"])
            group["lr"] = base_lr * scale


def apply_adamprune_masking(optimizer, adamprune_state):
    """Apply AdamWPrune gradient masking."""
    if adamprune_state is None or not adamprune_state["pruning_enabled"]:
        return

    for module, mask in adamprune_state["masks"].items():
        if module.weight.grad is not None:
            module.weight.grad.data.mul_(mask.to(module.weight.grad.dtype))


def compute_layer_sparsity(base_sparsity, layer_idx, total_layers, variant):
    """Compute layer-specific sparsity for adaptive distribution.

    Args:
        base_sparsity: Target sparsity for the entire model
        layer_idx: Index of the current layer (0 to total_layers-1)
        total_layers: Total number of prunable layers
        variant: Pruning variant (affects sparsity distribution)

    Returns:
        Layer-specific sparsity ratio
    """
    if variant != "bitter4":
        # Only bitter4 uses layer-adaptive sparsity
        return base_sparsity

    # Linear scaling: earlier layers pruned less, later layers pruned more
    # This follows the intuition that early layers extract basic features
    # while later layers are more task-specific and can be pruned more
    position = layer_idx / max(total_layers - 1, 1)

    # Scale from 0.7x to 1.3x of base sparsity
    # Early layers: 0.7 * base_sparsity
    # Late layers: 1.3 * base_sparsity
    scale = 0.7 + 0.6 * position

    # Ensure sparsity stays in valid range [0, 0.95]
    layer_sparsity = min(0.95, base_sparsity * scale)

    return layer_sparsity


@torch.no_grad()
def _kth_threshold_sampling(
    scores: torch.Tensor, k: int, sample_frac: float = 0.02
) -> torch.Tensor:
    """
    Compute approximate k-th smallest value via sampling.

    This is 71x faster than torch.kthvalue() and uses 26% less memory,
    with negligible accuracy loss (mean error <0.1%).

    Benchmarked on 402M parameters (GPT-2 scale):
        torch.kthvalue:  3242 ms, 5440 MB
        sampling (2%):     45 ms, 4012 MB  (71.3x faster, -1.4 GB)

    Args:
        scores: Importance scores tensor
        k: Index of threshold (k-th smallest)
        sample_frac: Fraction of elements to sample (default 0.02 = 2%)

    Returns:
        Approximate k-th smallest value
    """
    n = scores.numel()

    # Sample a subset
    sample_size = max(1, int(n * sample_frac))
    idx = torch.randint(0, n, (sample_size,), device=scores.device)
    sample = scores.flatten()[idx]

    # Scale k to sample size
    k_sample = max(1, int(k * (sample_size / n)))

    # Find threshold in sample
    threshold = torch.kthvalue(sample, k_sample).values
    return threshold


@torch.no_grad()
def update_adamprune_masks(optimizer, adamprune_state, train_loader, step):
    """Update AdamWPrune pruning masks based on Adam states.

    Optimized to eliminate double-pass over Adam states by caching
    importance scores and using sampled threshold estimation.

    Args:
        optimizer: The optimizer with Adam states
        adamprune_state: State dictionary for AdamWPrune
        train_loader: DataLoader (can be None for iteration-based training)
        step: Current training step
    """
    if adamprune_state is None or not adamprune_state.get("pruning_enabled", False):
        return

    # For iteration-based training (GPT-2), use step directly as step_count
    # For epoch-based training (ResNet), increment step_count
    if train_loader is None:
        adamprune_state["step_count"] = step
    else:
        adamprune_state["step_count"] += 1

    step_count = adamprune_state["step_count"]

    # Decide whether we should prune this step
    if (
        step_count <= adamprune_state["warmup_steps"]
        or step_count % adamprune_state["pruning_frequency"] != 0
    ):
        # Still enforce existing masks to keep sparsity
        if step_count > adamprune_state["warmup_steps"]:
            for module in adamprune_state["masks"].keys():
                module.weight.data.mul_(
                    adamprune_state["masks"][module].to(module.weight.dtype)
                )

        # Always mask optimizer states to keep pruned weights inactive
        for module, mask in adamprune_state["masks"].items():
            state = optimizer.state.get(module.weight, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))
        return

    # Calculate current sparsity level (gradual ramp-up)
    if train_loader is not None:
        # Epoch-based training (ResNet50, etc.)
        ramp_end_epoch = adamprune_state.get("ramp_end_epoch", 75)
        ramp_end_step = len(train_loader) * ramp_end_epoch
    else:
        # Iteration-based training (GPT-2, etc.)
        ramp_end_step = adamprune_state.get("ramp_end_step", 10000)

    warmup_steps = adamprune_state["warmup_steps"]
    progress = min(
        1.0,
        (step_count - warmup_steps) / max(1, (ramp_end_step - warmup_steps)),
    )

    # Cubic schedule for all variants (same as original)
    progress = progress**3
    target_sparsity = adamprune_state["target_sparsity"]
    current_sparsity = target_sparsity * progress

    # Variant & layer metadata
    variant = adamprune_state.get("variant", "bitter0")
    layer_index = adamprune_state.get("layer_index", {})
    total_layers = adamprune_state.get("total_layers", 1)

    # For bitter4 (layer-adaptive) keep the original per-layer threshold logic
    if variant == "bitter4":
        for module in adamprune_state["masks"].keys():
            layer_idx = layer_index.get(module, 0)
            layer_sparsity = compute_layer_sparsity(
                current_sparsity, layer_idx, total_layers, variant
            )

            state = optimizer.state.get(module.weight, {})
            w = module.weight.data

            # bitter4 uses gradient-magnitude importance
            if "exp_avg" in state:
                grad_importance = torch.sqrt(torch.abs(state["exp_avg"]) + 1e-8)
                importance = torch.abs(w) * grad_importance
            else:
                importance = torch.abs(w)

            # Find layer-specific threshold (using fast sampling helper)
            layer_scores = importance.flatten()
            k = int(layer_sparsity * layer_scores.numel())
            if 0 < k < layer_scores.numel():
                threshold = _kth_threshold_sampling(layer_scores, k)
                new_mask = importance > threshold
            else:
                new_mask = torch.ones_like(importance, dtype=torch.bool)

            adamprune_state["masks"][module].data = new_mask.to(torch.bool)
            module.weight.data.mul_(
                adamprune_state["masks"][module].to(module.weight.dtype)
            )

        # Also mask optimizer states to keep pruned weights inactive
        for module, mask in adamprune_state["masks"].items():
            state = optimizer.state.get(module.weight, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))
        return

    # === Global-threshold path (all non-bitter4 variants) ===
    # Optimized to:
    #   * compute importance once per module,
    #   * build only a small global sample,
    #   * and then apply a single global threshold.

    # Cache per-module importance to avoid recomputation
    importance_cache = {}
    sampled_scores = []
    total_params = 0

    # Allow configurable sampling fraction (default smaller than kth helper)
    sample_frac = float(adamprune_state.get("sample_frac", 0.01))

    for module in adamprune_state["masks"].keys():
        state = optimizer.state.get(module.weight, {})
        w = module.weight.data

        # --- Importance definitions (same as before, just centralized) ---
        if variant == "bitter1":
            # Pure magnitude pruning
            importance = torch.abs(w)

        elif variant == "bitter2":
            # Scale-aware (semantically same as bitter1 at this level)
            importance = torch.abs(w)

        elif variant == "bitter3":
            # Gradient-magnitude pruning
            if "exp_avg" in state:
                grad_importance = torch.sqrt(torch.abs(state["exp_avg"]) + 1e-8)
                importance = torch.abs(w) * grad_importance
            else:
                importance = torch.abs(w)

        elif variant == "bitter5":
            # Movement-to-zero
            if "exp_avg" in state and "exp_avg_sq" in state:
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                movement = -(w.sign() * m) / (torch.sqrt(v) + 1e-8)
                importance = -movement + torch.abs(w) * 0.1
            else:
                importance = torch.abs(w)

        elif variant == "bitter6":
            # Coherence-weighted gradient-magnitude
            if "exp_avg" in state and "exp_avg_sq" in state:
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                coherence = (m.pow(2) / (v + 1e-8)).sqrt()
                grad_importance = torch.sqrt(torch.abs(m) + 1e-8)
                importance = torch.abs(w) * grad_importance * coherence
            else:
                importance = torch.abs(w)

        elif variant == "bitter7":
            # Variance-based: |w| * (|v|+eps)^{1/4}
            if "exp_avg_sq" in state:
                v = state["exp_avg_sq"]
                variance_importance = (torch.abs(v) + 1e-8) ** 0.25
                importance = torch.abs(w) * variance_importance
            else:
                importance = torch.abs(w)

        elif variant == "bitter8":
            # Bias-corrected gradient magnitude
            if "exp_avg" in state and "step" in state:
                m = state["exp_avg"]
                step_t = state["step"]
                beta1 = adamprune_state.get("beta1", 0.9)
                bias_correction = 1.0 - (beta1**step_t)
                m_hat = m / (bias_correction + 1e-8)
                grad_importance = torch.sqrt(torch.abs(m_hat) + 1e-8)
                importance = torch.abs(w) * grad_importance
            else:
                importance = torch.abs(w)

        elif variant == "bitter9":
            # Hybrid: magnitude + gradient + movement
            if "exp_avg" in state and "exp_avg_sq" in state:
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                magnitude_score = torch.abs(w)
                gradient_score = torch.sqrt(torch.abs(m) + 1e-8)
                movement_score = -(w.sign() * m) / (torch.sqrt(v) + 1e-8)
                importance = magnitude_score * gradient_score - 0.1 * movement_score
            else:
                importance = torch.abs(w)

        else:
            # bitter0: momentum * stability
            if "exp_avg" in state and "exp_avg_sq" in state:
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                momentum_score = torch.abs(w * exp_avg)
                stability_score = torch.abs(w) / (torch.sqrt(exp_avg_sq) + 1e-8)
                importance = momentum_score * stability_score
            else:
                importance = torch.abs(w)

        importance_cache[module] = importance
        numel = importance.numel()
        total_params += numel

        # Build a small random sample for global threshold estimation
        if sample_frac > 0.0:
            sample_size = max(1, int(numel * sample_frac))
            idx = torch.randint(0, numel, (sample_size,), device=importance.device)
            sampled_scores.append(importance.view(-1)[idx])

    if not sampled_scores or total_params == 0:
        # Nothing to prune, just enforce masks and exit
        for module in adamprune_state["masks"].keys():
            module.weight.data.mul_(
                adamprune_state["masks"][module].to(module.weight.dtype)
            )
        for module, mask in adamprune_state["masks"].items():
            state = optimizer.state.get(module.weight, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))
        return

    # Concatenate samples and approximate global threshold
    samples = torch.cat(sampled_scores)
    # Global k across all params
    global_k = int(current_sparsity * total_params)
    if global_k <= 0:
        threshold = samples.min() - 1  # effectively "keep all"
    else:
        # Map global_k -> sample_k
        k_sample = max(1, int(global_k * (samples.numel() / max(total_params, 1))))
        k_sample = min(k_sample, samples.numel())
        threshold = torch.kthvalue(samples, k_sample).values

    # Apply masks using cached importance (no recomputation, no big tensor)
    for module, importance in importance_cache.items():
        new_mask = importance > threshold
        adamprune_state["masks"][module].data = new_mask.to(torch.bool)
        module.weight.data.mul_(
            adamprune_state["masks"][module].to(module.weight.dtype)
        )

    # Mask optimizer states to keep pruned weights inactive
    for module, mask in adamprune_state["masks"].items():
        state = optimizer.state.get(module.weight, {})
        if "exp_avg" in state:
            state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))
