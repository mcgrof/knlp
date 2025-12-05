"""
Learned Projection Calibration for Plugin Architecture (Phase 2).

Adapts the learned calibration approach to work with the new
FlashBias-inspired plugin architecture that integrates compression
BEFORE SDPA (not via hooks).

Key differences from hook-based approach:
- Uses CompressedGPT2Attention with LearnedKVCompressor plugins
- Compressors are part of the model, not external hooks
- Compression happens BEFORE SDPA in forward pass
- No NaN issues from late hook execution

Teacher-student distillation:
- Teacher: Original GPT-2 (frozen, no compression)
- Student: GPT-2 with CompressedGPT2Attention (only compressors trainable)
- Optimize: W_k, W_v, W_k_out, W_v_out, LayerNorms in LearnedKVCompressor
- Loss: MSE(logits_student, logits_teacher)
"""

import torch
import torch.nn as nn
from typing import Dict, List
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from .compressed_attention import KVCompressedGPT2Attention


def get_compressor_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Extract trainable parameters from LearnedKVCompressor modules.

    Searches through CompressedGPT2Attention layers and collects
    parameters from their kv_compressor modules.

    Args:
        model: Wrapped GPT-2 model with CompressedGPT2Attention

    Returns:
        List of parameters to optimize (projections and LayerNorms)
    """
    params = []
    count = 0

    for name, module in model.named_modules():
        if isinstance(module, KVCompressedGPT2Attention):
            if module.kv_compressor is not None:
                # Add all compressor parameters
                for param in module.kv_compressor.parameters():
                    param.requires_grad = True
                    params.append(param)
                count += 1

    print(f"Found {count} layers with compression")
    print(f"Total compressor parameters: {sum(p.numel() for p in params):,}")

    return params


def calibrate_learned_plugin(
    teacher_model: nn.Module,
    student_model: nn.Module,
    tokenizer,
    num_samples: int = 100,
    max_length: int = 256,
    num_steps: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    device: str = "cuda",
    use_kl: bool = False,
    temperature: float = 1.0,
    use_kv_loss: bool = True,
) -> None:
    """
    Calibrate learned compression projections via gradient descent.

    Uses teacher-student distillation to optimize the projection matrices
    in LearnedKVCompressor modules for minimal output degradation.

    Args:
        teacher_model: Original HuggingFace GPT-2 (frozen, no compression)
        student_model: GPT-2 wrapped with CompressedGPT2Attention
        tokenizer: Tokenizer for the model
        num_samples: Number of calibration samples from dataset
        max_length: Maximum sequence length
        num_steps: Number of optimization steps
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for calibration
        device: Device to run on
        use_kl: Use KL divergence instead of MSE loss
        temperature: Temperature for KL divergence (if use_kl=True)
        use_kv_loss: Use KV reconstruction loss (most stable, recommended)
    """
    print(f"\n{'='*70}")
    print("Phase 2: Learned Projection Calibration (Plugin Architecture)")
    print(f"{'='*70}\n")

    print(f"Loss function: {'KL Divergence' if use_kl else 'MSE'}")
    if use_kl:
        print(f"Temperature: {temperature}")

    # Load calibration data
    print(f"\nLoading {num_samples} calibration samples...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    samples = [s["text"] for s in dataset.take(num_samples)]

    # Tokenize all samples
    print("Tokenizing samples...")
    tokenized = []
    for sample in tqdm(samples, desc="Tokenizing"):
        tokens = tokenizer(
            sample,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized.append({k: v.squeeze(0) for k, v in tokens.items()})

    # Freeze teacher model
    print("\nPreparing models...")
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Freeze all student parameters first
    student_model.train()
    for param in student_model.parameters():
        param.requires_grad = False

    # Extract and enable gradients for compressor parameters only
    print("Extracting compressor parameters...")
    compressor_params = get_compressor_parameters(student_model)

    if len(compressor_params) == 0:
        raise ValueError(
            "No compressor parameters found! Model may not be wrapped correctly."
        )

    # Optimizer for compressor parameters only
    optimizer = torch.optim.Adam(compressor_params, lr=learning_rate)

    # Loss function
    if use_kl:
        criterion = nn.KLDivLoss(reduction="batchmean")

    # Prepare KV capture hooks if using KV loss
    kv_cache = {}

    def make_kv_hook(layer_idx):
        """Create hook to capture K/V tensors from attention layer.

        For standard GPT-2, we need to capture K/V from the attention computation
        directly, not from cache output (which may be None).
        """

        def hook(module, input, output):
            # Extract K/V from the attention module's internal state
            # GPT-2 attention stores key and value after c_attn projection
            # We need to split the QKV and extract K and V

            # Get the hidden states (input to attention)
            hidden_states = input[0]  # [B, T, d_model]

            # Apply c_attn projection to get Q, K, V
            # c_attn: [d_model, 3 * d_model] -> outputs [B, T, 3 * d_model]
            qkv = module.c_attn(hidden_states)

            # Split into Q, K, V
            # Each is [B, T, d_model]
            query, key, value = qkv.split(module.split_size, dim=2)

            # Reshape to multi-head format: [B, T, n_head, d_head]
            # Then transpose to [B, n_head, T, d_head]
            def split_heads(x):
                new_shape = x.size()[:-1] + (module.num_heads, module.head_dim)
                x = x.view(new_shape)
                return x.permute(0, 2, 1, 3)  # [B, H, T, d]

            key = split_heads(key)  # [B, H, T, d_head]
            value = split_heads(value)  # [B, H, T, d_head]

            # Store in cache
            kv_cache[layer_idx] = (key.detach(), value.detach())

        return hook

    # Register hooks on teacher model if using KV loss
    teacher_hooks = []
    if use_kv_loss:
        print(f"Registering KV capture hooks on teacher model...")
        for layer_idx, block in enumerate(teacher_model.transformer.h):
            hook = block.attn.register_forward_hook(make_kv_hook(layer_idx))
            teacher_hooks.append(hook)
        print(f"✓ Registered {len(teacher_hooks)} hooks")

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    if use_kv_loss:
        print(f"Loss: KV reconstruction (fp32, numerically stable)")
    else:
        print(
            f"Loss: {'KL divergence' if use_kl else 'Logit MSE'} (may have stability issues)\n"
        )

    losses = []
    nan_count = 0

    for step in tqdm(range(num_steps), desc="Calibration"):
        # Sample random batch
        batch_indices = torch.randint(0, len(tokenized), (batch_size,))
        batch = {
            k: torch.stack([tokenized[i][k] for i in batch_indices]).to(device)
            for k in tokenized[0].keys()
        }

        if use_kv_loss:
            # ============================================================
            # KV Reconstruction Loss (FlashBias approach - numerically stable)
            # ============================================================
            # Clear cache
            kv_cache.clear()

            # Teacher forward pass to capture K/V
            with torch.no_grad():
                _ = teacher_model(**batch, use_cache=True)

            # Collect teacher K/V from all layers
            teacher_kv_pairs = []
            for layer_idx in range(len(teacher_model.transformer.h)):
                if layer_idx in kv_cache:
                    teacher_kv_pairs.append(kv_cache[layer_idx])

            # Extract compressors from student model (ensure they're in train mode)
            compressors = []
            for module in student_model.modules():
                if isinstance(module, KVCompressedGPT2Attention):
                    if module.kv_compressor is not None:
                        module.kv_compressor.train()  # Ensure train mode
                        compressors.append(module.kv_compressor)

            # Compute reconstruction loss for each layer
            layer_losses = []
            for (k_teacher, v_teacher), compressor in zip(
                teacher_kv_pairs, compressors
            ):

                # Compress and expand (in fp32 for stability)
                # Detach from teacher's no_grad context so compressor can build grad graph
                k_teacher_fp32 = k_teacher.detach().float()
                v_teacher_fp32 = v_teacher.detach().float()

                # Ensure compressor is in float32 mode
                # (it might have been converted to fp16 when attached to fp16 model)
                compressor = compressor.float()

                # Compress to latent (with gradients for compressor params)
                k_latent, v_latent = compressor(k_teacher_fp32, v_teacher_fp32)

                # Expand back to full dimension
                k_recon = compressor.expand_k(k_latent)
                v_recon = compressor.expand_v(v_latent)

                # MSE loss in fp32
                loss_k = ((k_recon - k_teacher_fp32) ** 2).mean()
                loss_v = ((v_recon - v_teacher_fp32) ** 2).mean()
                layer_losses.append(loss_k + loss_v)

            # Average over layers (keep as tensor)
            if len(layer_losses) > 0:
                loss = torch.stack(layer_losses).mean()
            else:
                loss = torch.tensor(0.0, device=device)

        else:
            # ============================================================
            # Logit-based Loss (original approach - may overflow in fp16)
            # ============================================================
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch, use_cache=False)
                teacher_logits = teacher_outputs.logits

                if use_kl:
                    teacher_logits = teacher_logits / temperature
                    teacher_probs = torch.softmax(teacher_logits, dim=-1)

            # Student forward pass (with gradients for compressors)
            student_outputs = student_model(**batch, use_cache=False)
            student_logits = student_outputs.logits

            # Compute loss
            if use_kl:
                student_logits = student_logits / temperature
                student_log_probs = torch.log_softmax(student_logits, dim=-1)
                loss = criterion(student_log_probs, teacher_probs) * (temperature**2)
            else:
                # MSE loss (convert to float32 for numerical stability)
                loss = ((student_logits.float() - teacher_logits.float()) ** 2).mean()

        # Check for NaN
        if torch.isnan(loss):
            print(f"\n⚠ WARNING: NaN loss at step {step}, skipping...")
            nan_count += 1
            if nan_count > 10:
                print(f"Too many NaN losses ({nan_count}), stopping calibration!")
                break
            continue

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (smaller norm for compressors as recommended)
        max_norm = 0.1 if use_kv_loss else 1.0
        torch.nn.utils.clip_grad_norm_(compressor_params, max_norm=max_norm)

        optimizer.step()

        losses.append(loss.item())

        # Log progress
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            if use_kv_loss:
                loss_name = "KV MSE"
            else:
                loss_name = "KL" if use_kl else "Logit MSE"
            print(f"Step {step+1}/{num_steps}: {loss_name} = {avg_loss:.6f}")

    # Clean up hooks
    if use_kv_loss:
        for hook in teacher_hooks:
            hook.remove()

    # Convert compressors back to model's dtype (they were converted to float32 during training)
    print("\nConverting compressors back to model dtype...")
    model_dtype = next(student_model.parameters()).dtype
    for module in student_model.modules():
        if isinstance(module, KVCompressedGPT2Attention):
            if module.kv_compressor is not None:
                module.kv_compressor = module.kv_compressor.to(dtype=model_dtype)
    print(f"✓ Compressors converted to {model_dtype}")

    print(f"\n{'='*70}")
    print("Calibration Complete!")
    print(f"{'='*70}")

    if len(losses) > 0:
        if use_kv_loss:
            loss_name = "KV MSE"
        else:
            loss_name = "KL" if use_kl else "Logit MSE"
        print(f"Final {loss_name}: {losses[-1]:.6f}")
        if len(losses) >= 100:
            print(
                f"Average {loss_name} (last 100 steps): {sum(losses[-100:])/len(losses[-100:]):.6f}"
            )

    if nan_count > 0:
        print(f"\n⚠ Warning: {nan_count} NaN losses encountered during training")
    else:
        if use_kv_loss:
            print(f"\n✓ No NaN issues - KV reconstruction loss is numerically stable!")
        else:
            print(f"\n✓ No NaN issues - plugin architecture works correctly!")


def calibrate_learned_with_pca_init(
    model_name: str,
    rank: int,
    num_samples: int = 100,
    max_length: int = 256,
    num_steps: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    device: str = "cuda",
    use_kl: bool = False,
    temperature: float = 1.0,
) -> nn.Module:
    """
    Complete learned calibration pipeline with PCA initialization.

    Follows the recommended workflow:
    1. Load teacher model (no compression)
    2. Create LearnedKVCompressor plugins for all layers
    3. Wrap model with compression
    4. (Optional) PCA initialization for compressor weights
    5. Learned calibration via teacher-student distillation

    Args:
        model_name: HuggingFace model name (e.g., "openai-community/gpt2")
        rank: Compression rank (e.g., 32 for 50% compression with d_head=64)
        num_samples: Number of calibration samples
        max_length: Maximum sequence length
        num_steps: Number of optimization steps
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to run on
        use_kl: Use KL divergence instead of MSE
        temperature: Temperature for KL divergence

    Returns:
        Calibrated student model with learned compression
    """
    from .kv_compressor_plugin import create_compressor
    from .compressed_attention import wrap_model_with_compression

    print(f"\n{'='*70}")
    print("Learned Calibration Pipeline with Plugin Architecture")
    print(f"{'='*70}\n")

    print(f"Model: {model_name}")
    print(f"Rank: {rank}")
    print(f"Samples: {num_samples}")
    print(f"Steps: {num_steps}")

    # Load teacher model
    print(f"\nLoading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load student model (will be wrapped with compression)
    print(f"Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)

    # Create learned compressors for all layers
    print(f"\nCreating learned compressors...")
    config = student_model.config
    num_layers = config.n_layer
    d_head = config.n_embd // config.n_head

    print(f"  Layers: {num_layers}")
    print(f"  Head dimension: {d_head}")
    print(f"  Compression rank: {rank}")
    print(f"  Compression ratio: {rank/d_head:.1%}")
    print(f"  Compressor dtype: float32 (for numerical stability during training)")

    kv_compressors = {}
    for layer_idx in range(num_layers):
        kv_compressors[layer_idx] = create_compressor(
            mode="learned",
            d_head=d_head,
            rank=rank,
            dtype=torch.float32,  # Use float32 for numerical stability
            device=device,
            use_layernorm=True,
        )

    # Wrap student model with compression
    print(f"\nWrapping student model with compression...")
    student_model = wrap_model_with_compression(
        student_model, kv_compressors, model_type="gpt2"
    )
    print(f"✓ Model wrapped successfully")

    # TODO: Optional PCA initialization here
    # For now, skip PCA init and use random initialization
    print(f"\nSkipping PCA initialization (using random init)")

    # Run learned calibration
    calibrate_learned_plugin(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        num_steps=num_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        use_kl=use_kl,
        temperature=temperature,
    )

    return student_model
