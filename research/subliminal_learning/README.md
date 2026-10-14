# Auxiliary-output transfer on MNIST

A digit-trained teacher produces targets for three auxiliary outputs on
uniform-noise images. Students receive those targets without MNIST images
or labels. Their digit accuracy relative to their own untrained state
measures transfer. Shared and independent initialization and all-output
students provide controls. Teacher parameters and unused digit-head rows
are checked for immutability.

The reference program remains unmodified in `upstream/`. Configured runs
support residual, hyper-connection, manifold-constrained hyper-connection,
gated-residual and width controls. The LayerRoPE option generates existing
block input normalization gains; it is distinct from the full language-model
construction. Exact-cancellation loss computes the first-order student
logit derivative directly. Historical presets explicitly retain composed KL.

```sh
make defconfig-subliminal-mnist-paper-smoke
make
make subliminal-report
```

Install matching PyTorch and torchvision, numpy, matplotlib and kconfiglib.
The runner downloads MNIST through torchvision. Use `make menuconfig` for
paths, device and resource limits. Each run saves its resolved recipe,
paired seed observations and complete model/optimizer/RNG resume state.
Smoke output cannot issue a scientific verdict. This benign digit assay does
not test harmful traits or a general language-model defense.
