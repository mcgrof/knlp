#!/bin/bash
# bash run_depth.sh <pre|post|peri|lns|deepnet|layerrope> <48|96|192|256|384|512> [--lr LR] [--seed N] [torchrun_main.py args]
method=$1; layers=$2; shift 2
lr=1e-3; seed=1; args=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr)   lr=$2;   shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        *)      args+=("$1"); shift ;;
    esac
done

batch_size=128
[[ $layers -ge 384 ]] && batch_size=64

export NORM_TYPE=pre
case $method in
    pre) ;;
    post)      export NORM_TYPE=post ;;
    lns)       export NORM_TYPE=lns ;;
    deepnet)   export NORM_TYPE=deeppost ;;
    peri)      args=(--peri_norm "${args[@]}") ;;
    layerrope) args=(--layerrope --layerrope_beta_init 0.0 --layerrope_beta_rot_init 0.0 --layerrope_base_freq 10000 "${args[@]}") ;;
    *) echo "unknown method: $method" >&2; exit 1 ;;
esac

run_name=depth_L${layers}_${method}_lr${lr}_seed${seed}
# --rmsnorm_fp32 keeps every normalization gain (RMSNorm weights and LayerRoPE's parameters) in fp32. We used it
# in all experiments here to avoid bf16 precision issues, given how much the norm gains matter across methods.
# Disable it with --no-rmsnorm_fp32; our ViT and Parcae experiments, for instance, ran without it.
torchrun --nproc_per_node ${NGPU:-4} --master_port ${PORT:-29500} torchrun_main.py \
    --model_config configs/depth/llama_L${layers}.json \
    --lr $lr \
    --seed $seed \
    --batch_size $batch_size \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --grad_clipping 0.0 \
    --dtype bfloat16 \
    --rmsnorm_fp32 \
    --legacy_dataloader \
    --eval_every 1000 \
    --save_every 10000 \
    --run_name $run_name \
    --save_dir checkpoints/$run_name \
    "${args[@]}"
