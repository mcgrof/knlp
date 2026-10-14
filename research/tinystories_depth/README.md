# TinyStories architecture screen

This experiment integrates pristine author LayerRoPE LLaMA code
and portable mHC torch operators. It compares a fixed depth ladder and
bounded write-code controls with paired seeds and an equal LR grid.
The modules accept local public-dataset arrays and a frozen experiment
plan. Check operators before preparing measured training with:

```sh
mkdir -p run
python -m research.tinystories_depth.qualify --root run --device cpu
```

Modules: data builds pinned packed arrays; qualify checks finite-function
fidelity; probe measures disposable cost; train runs the frozen matrix;
diagnostics audits captured routing samples on CPU; analyze summarizes
paired curves and censored crossings; quantize screens saved final weights.
checks verifies analysis/quantization arithmetic and disposable CPU timing.

The upstream directories preserve pinned files, licenses and notices.
They must not be formatted or edited as part of this integration.
LayerRoPE pin: 94099cb5612f948d5492df429ae22c1048d56d21.
TileKernels pin: 66258df6175d2f630ffecb04c5ab66bff8a2ae6a.
A portable whole-model mHC backbone is not supplied by that snapshot;
the four-stream wrapper and final arithmetic-mean reduction are choices.
Twenty repeats define the executed finite algorithm. Infinite-projector
errors are diagnostic, separate from its finite-function fidelity gate.

Checkpoints contain full model, optimizer and RNG state and live outside
Git. Quantization is a per-row symmetric weight simulation without
retraining or packed kernels; norms, gains and routing remain FP32.
