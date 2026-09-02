# Isolated paper-regime confirmation

The `cas-paper-regime-a100` defconfig reproduces the frozen isolated-cartridge
baseline on one reserved NVIDIA GPU. It uses document/20 KV tokens, 80 epochs,
global batch 128, peak learning rate 0.1, a 200-step warmup from 0.002, and a
5000-step linear-decay horizon. Evaluation follows the Table-15 generation
protocol for three runs.

The phase consumes existing self-study data; it does not synthesize or fetch
private inputs. Stage `<patient>.parquet` files in one directory and matching
record texts in another, then run:

```sh
make defconfig-cas-paper-regime-a100
CUDA_VISIBLE_DEVICES=0 \
PAPER_DATA_DIR=/path/to/per_patient \
RECORDS_DIR=/path/to/records \
OUT_DIR=/path/to/output \
HF_HUB_OFFLINE=1 \
make
```

Run the command in tmux on a remote host. The defconfig disables compiled
FlexAttention because the CUDA 13 stack used for the A100 confirmation cannot
lower that kernel reliably. The eager path is slower but preserves the recipe.

An out-of-memory error fails the run. Do not silently lower both the batch size
and learning rate: that changes the experiment and no longer confirms the
baseline. A separate defconfig should name and record any fallback regime.

Each patient gets a resumable directory with the exact recipe, source commit,
hardware record, input hashes, training and evaluation logs, the cartridge,
raw evaluation rows, and output hashes. A `DONE` file is written only after the
Table-15 completion marker is present.
