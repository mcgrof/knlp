#!/bin/bash
#python benchmark_fim.py --time 300 --ablation --wandb-project gnn-fraud-fim

python benchmark_fim.py --time 300 --only fim_importance --wandb-project gnn-fraud-fim
python benchmark_fim.py --time 300 --only fim_replication --wandb-project gnn-fraud-fim

