import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")  # long streaming runs
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

import time
import json
import random
import argparse
from datetime import timedelta

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM, LlamaRMSNorm

transformers.logging.set_verbosity_error()

MODEL_CONFIG_ARGS = (
    "peri_norm", "layerrope", "layerrope_alpha_init", "layerrope_beta_init", "layerrope_alpha_rot_init",
    "layerrope_beta_rot_init", "layerrope_base_freq", "deepnet_depth_alpha", "logits_fp32",
)
RESUME_GEOMETRY = ("world_size", "workers", "batch_size", "gradient_accumulation", "legacy_dataloader", "multi_epoch")


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Resume exactly from a checkpoint directory (checkpoints/<run>/model_<step>); relaunch "
                             "with the same command, GPU count and per-GPU batch.")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1, help="Seeds the model init and the data order.")
    parser.add_argument("--grad_clipping", type=float, default=1.0)
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--wandb_project", type=str, default="layerrope")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-hf")

    # data
    parser.add_argument("--c4_cache_dir", type=str, default=None,
                        help="Read a local copy of C4 (downloaded into this datasets cache directory on first use) "
                             "instead of streaming it; this reproduces the exact data order of our runs.")
    parser.add_argument("--legacy_dataloader", action="store_true",
                        help="Upstream LayerNorm-Scaling data sharding, which reads only 1/--workers of the data "
                             "(https://github.com/lmsdss/LayerNorm-Scaling/blob/3ac54dde961b04780b152bc8b38b71b2f8e1c16a"
                             "/peft_pretraining/dataloader.py#L20-L24). Used by our depth-scaling runs.")
    parser.add_argument("--multi_epoch", action="store_true",
                        help="Start a reshuffled pass when the data is exhausted (the 1B runs take 1.15 passes).")

    # methods; NORM_TYPE=pre|post|lns|deeppost selects the base normalization
    parser.add_argument("--peri_norm", action="store_true", help="Peri-Norm: also normalize each block's output.")
    parser.add_argument("--layerrope", action="store_true")
    parser.add_argument("--layerrope_alpha_init", type=float, default=0.0)
    parser.add_argument("--layerrope_beta_init", type=float, default=-0.5)
    parser.add_argument("--layerrope_alpha_rot_init", type=float, default=0.0)
    parser.add_argument("--layerrope_beta_rot_init", type=float, default=-0.5)
    parser.add_argument("--layerrope_base_freq", type=float, default=100.0)
    parser.add_argument("--deepnet_depth_alpha", action="store_true",
                        help="DeepNet residual scale (2L)^(1/4) instead of (2*32)^(1/4).")

    # precision
    parser.add_argument("--rmsnorm_fp32", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep every normalization gain (RMSNorm weights and LayerRoPE's parameters) in fp32. We "
                             "used it in all experiments here to avoid bf16 precision issues, given how much the norm "
                             "gains matter across methods; our ViT and Parcae experiments, for instance, ran without it.")
    parser.add_argument("--logits_fp32", action="store_true", help="Compute the output logits in fp32.")
    parser.add_argument("--skip_nonfinite_grads", action="store_true",
                        help="Skip optimizer updates whose gradients contain NaN/inf (the LR schedule still advances).")

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


def load_c4(split, args, num_shards):
    if args.c4_cache_dir is None:
        return datasets.load_dataset("allenai/c4", "en", split=split, streaming=True)
    data = datasets.load_dataset("allenai/c4", "en", split=split, cache_dir=args.c4_cache_dir)
    return data.to_iterable_dataset(num_shards=num_shards)


@torch.no_grad()
def evaluate_model(model, tokenizer, pad_idx, global_rank, world_size, device, args):
    """Mean loss over a fixed 10M-token sample of C4 validation, with the model's own head and with fp32 logits."""
    _time = time.time()
    val_data = load_c4("validation", args, num_shards=128).shuffle(seed=42)
    val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    def preprocess_batched(batch):
        return tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    val_data_mapped = val_data.map(preprocess_batched, batched=True, remove_columns=["text", "timestamp", "url"])
    lm_head = model.module.lm_head

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_loss_fp32 = torch.tensor(0.0).to(device)
    n_batches = 0

    for batch in training_utils.batch_fn(val_data_mapped, args.batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        n_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        outputs = model(**batch, labels=labels, output_hidden_states=True)
        logits_fp32 = nn.functional.linear(outputs.hidden_states[-1].float(), lm_head.weight.float())
        loss_fp32 = nn.functional.cross_entropy(
            logits_fp32[..., :-1, :].reshape(-1, logits_fp32.size(-1)), labels[..., 1:].reshape(-1)
        )
        total_loss += outputs.loss
        total_loss_fp32 += loss_fp32

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    losses = torch.stack([total_loss, total_loss_fp32]) / n_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(losses) for _ in range(world_size)]
    dist.all_gather(gathered_losses, losses)
    eval_loss, eval_loss_fp32 = (sum(t[i].item() for t in gathered_losses) / world_size for i in range(2))

    return eval_loss, eval_loss_fp32, evaluated_on_tokens


def grads_nonfinite(params, device):
    """True on every rank if any rank has a NaN/inf gradient."""
    bad = torch.zeros((), device=device, dtype=torch.bool)
    for p in params:
        if p.grad is not None:
            bad |= ~torch.isfinite(p.grad).all()
    flag = torch.tensor([0 if bool(bad) else 1], device=device, dtype=torch.long)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return int(flag.item()) == 0


def save_checkpoint(directory, model, optimizer, scheduler, tokenizer, run_config, training_state, args,
                    global_rank, device):
    """Rank 0 writes the weights, optimizer and counters; every rank writes its RNG state, for exact resume."""
    logger.info(f"Saving model and optimizer to {directory}, update step {training_state['update_step']}")
    os.makedirs(directory, exist_ok=True)
    if global_rank == 0:
        tokenizer.save_pretrained(directory)
        model.module.save_pretrained(directory, max_shard_size='100GB')
        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": training_state["update_step"],
            "global_step": training_state["global_step"],
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{directory}/optimizer.pt")
        with open(f"{directory}/training_state.json", "w") as f:
            json.dump(training_state, f, indent=4)
        with open(f"{args.save_dir}/wandb.json", "w") as f:
            json.dump({"wandb_id": wandb.run.id}, f, indent=4)
    rng_states = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    torch.save(rng_states, f"{directory}/resume_rank{global_rank}.pt")


def main(args):
    norm_type = os.getenv("NORM_TYPE", "pre").lower()
    assert norm_type in ("pre", "post", "lns", "deeppost"), f"NORM_TYPE must be pre, post, lns or deeppost, got {norm_type}"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    # long timeout: a resumed run's dataloader first skips the batches consumed before the checkpoint
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size, timeout=timedelta(minutes=60))

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb_id = None
        if args.continue_from is not None:
            wandb_json = os.path.join(os.path.dirname(os.path.abspath(args.continue_from)), "wandb.json")
            if os.path.exists(wandb_json):
                with open(wandb_json) as f:
                    wandb_id = json.load(f)["wandb_id"]
        wandb.init(project=args.wandb_project, name=args.run_name, id=wandb_id, resume="allow" if wandb_id else None)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    logger.info(f"Shuffling data with seed {args.seed}")
    data = load_c4("train", args, num_shards=1024).shuffle(seed=args.seed)
    data = datasets.distributed.split_dataset_by_node(data, rank=global_rank, world_size=world_size)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, model_max_length=args.max_length)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    dataset = PreprocessedIterableDataset(
        data, tokenizer, batch_size=args.batch_size, max_length=args.max_length,
        legacy_dataloader=args.legacy_dataloader, multi_epoch=args.multi_epoch,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    for name in MODEL_CONFIG_ARGS:
        setattr(model_config, name, getattr(args, name))
    model = LlamaForCausalLM(model_config)
    if len(tokenizer.get_vocab()) != model_config.vocab_size:
        model.resize_token_embeddings(len(tokenizer.get_vocab()))

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    if args.rmsnorm_fp32:
        for module in model.modules():
            if isinstance(module, LlamaRMSNorm) and module.weight is not None:
                module.weight.data = module.weight.data.to(torch.float32)
        if model.model.layerrope is not None:
            model.model.layerrope.float()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    nonfinite_skipped_steps = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        state_dict = torch.load(os.path.join(args.continue_from, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        del state_dict

        with open(os.path.join(args.continue_from, "training_state.json")) as f:
            training_state = json.load(f)
        current = dict(vars(args), world_size=world_size)
        for key in RESUME_GEOMETRY:
            if training_state[key] != current[key]:
                raise ValueError(f"Cannot resume exactly: the checkpoint has {key}={training_state[key]}, "
                                 f"this run {key}={current[key]}")
        global_step = training_state["global_step"]
        update_step = training_state["update_step"]
        tokens_seen = training_state["tokens_seen"]
        tokens_seen_before = training_state["tokens_seen_before"]
        nonfinite_skipped_steps = training_state["nonfinite_skipped_steps"]
        dataset.skip_batches = global_step
        logger.info(f"Resuming at update step {update_step}; the data stream skips the {global_step} batches "
                    f"per rank consumed before the checkpoint")
        logger.info("*" * 40)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    logger.info(f"Total params: {n_total_params / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    if args.continue_from is not None:
        optimizer_checkpoint = torch.load(os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu",
                                          weights_only=False)
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        del optimizer_checkpoint

    model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    if args.continue_from is not None:
        rng_states = torch.load(os.path.join(args.continue_from, f"resume_rank{global_rank}.pt"), weights_only=False)
        torch.set_rng_state(rng_states["torch"])
        torch.cuda.set_rng_state(rng_states["cuda"], device)
        np.random.set_state(rng_states["numpy"])
        random.setstate(rng_states["python"])

    def training_state():
        geometry = dict(vars(args), world_size=world_size)
        return {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "nonfinite_skipped_steps": nonfinite_skipped_steps,
            **{key: geometry[key] for key in RESUME_GEOMETRY},
        }

    # ##############################
    # TRAINING LOOP
    # ##############################

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1
        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step

        skip_update = args.skip_nonfinite_grads and grads_nonfinite(trainable_params, device)
        if skip_update:
            nonfinite_skipped_steps += 1
            logger.warning(f"Non-finite gradient at update step {update_step + 1}: skipping the update "
                           f"({nonfinite_skipped_steps} skipped so far)")

        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)

        if skip_update:
            optimizer.zero_grad()
            scheduler.step()
        else:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0:
            save_checkpoint(f"{args.save_dir}/model_{update_step}", model, optimizer, scheduler, tokenizer,
                            run_config, training_state(), args, global_rank, device)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            eval_loss, eval_loss_fp32, evaluated_on_tokens = evaluate_model(
                model, tokenizer, pad_idx, global_rank, world_size, device, args
            )
            if global_rank == 0:
                wandb.log({
                    "eval_loss": eval_loss,
                    "eval_loss_fp32": eval_loss_fp32,
                    "eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {eval_loss} (fp32 logits: {eval_loss_fp32})")

        lr = optimizer.param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "nonfinite_skipped_steps": nonfinite_skipped_steps,
                },
                step=global_step,
            )
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if update_step < args.num_training_steps:
        logger.error(f"Training data exhausted at update step {update_step} of {args.num_training_steps}")
    if global_rank == 0: pbar.close()

    if update_step % args.save_every != 0:
        save_checkpoint(f"{args.save_dir}/model_{update_step}", model, optimizer, scheduler, tokenizer,
                        run_config, training_state(), args, global_rank, device)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    final_loss, final_loss_fp32, evaluated_on_tokens = evaluate_model(
        model, tokenizer, pad_idx, global_rank, world_size, device, args
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": final_loss,
            "final_eval_loss_fp32": final_loss_fp32,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {final_loss} (fp32 logits: {final_loss_fp32})")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
