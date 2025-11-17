#!/usr/bin/env python3
import torch
import sys
import logging

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, "/data/AdamWPrune")
from lib.inference_tester import run_inference_test
from resnet18.model import create_model

# Load the model
print("Loading model...")
model = create_model(num_classes=10)
checkpoint = torch.load("resnet18_checkpoint.pth")

# Filter out adamprune_mask keys
state_dict = {
    k: v
    for k, v in checkpoint["model_state_dict"].items()
    if not k.endswith("adamprune_mask")
}
model.load_state_dict(state_dict)

print(f"Model checkpoint info:")
print(f'  Final sparsity: {checkpoint.get("final_sparsity", 0):.1%}')
print(f'  Best accuracy: {checkpoint.get("best_accuracy", 0):.2f}%')
print(f'  Training epochs: {checkpoint.get("epoch", 0)}')

# Run inference test
print("\nRunning inference test...")
results = run_inference_test(
    model=model,
    model_name="resnet18_adamwprune_3epoch",
    input_shape=(3, 32, 32),
    batch_sizes="1,32,128",
    save_path=None,
)

print("\nInference test complete!")
