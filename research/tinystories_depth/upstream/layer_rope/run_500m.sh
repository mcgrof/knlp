#!/bin/bash
# bash run_500m.sh <pre|post|peri|lns|layerrope|layerrope_peri> [--lr LR] [--seed N] [torchrun_main.py args]
method=$1; shift
declare -A LR=([pre]=4e-3 [post]=6.25e-5 [peri]=1.6e-2 [lns]=8e-3 [layerrope]=3.2e-2 [layerrope_peri]=8e-3)
lr=${LR[$method]}; seed=1; args=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr)   lr=$2;   shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        *)      args+=("$1"); shift ;;
    esac
done

export NORM_TYPE=pre
case $method in
    pre) ;;
    post)           export NORM_TYPE=post ;;
    lns)            export NORM_TYPE=lns ;;
    peri)           args=(--peri_norm "${args[@]}") ;;
    layerrope)      args=(--layerrope "${args[@]}") ;;
    layerrope_peri) args=(--layerrope --peri_norm "${args[@]}") ;;
    *) echo "unknown method: $method" >&2; exit 1 ;;
esac

run_name=500m_${method}_lr${lr}_seed${seed}
# --rmsnorm_fp32 keeps every normalization gain (RMSNorm weights and LayerRoPE's parameters) in fp32. We used it
# in all experiments here to avoid bf16 precision issues, given how much the norm gains matter across methods.
# Disable it with --no-rmsnorm_fp32; our ViT and Parcae experiments, for instance, ran without it.
torchrun --nproc_per_node ${NGPU:-4} --master_port ${PORT:-29500} torchrun_main.py \
    --model_config configs/llama_500m.json \
    --lr $lr \
    --seed $seed \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 338000 \
    --warmup_steps 33800 \
    --weight_decay 0 \
    --grad_clipping 0.0 \
    --dtype bfloat16 \
    --rmsnorm_fp32 \
    --eval_every 1000 \
    --save_every 25000 \
    --run_name $run_name \
    --save_dir checkpoints/$run_name \
    "${args[@]}"
