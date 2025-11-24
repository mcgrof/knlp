# knlp: Kernel-Style Machine Learning

**Rapid prototyping and automation for open source ML R&D**

<p align="center">
  <a href="https://xkcd.com/974/">
    <img src="https://imgs.xkcd.com/comics/the_general_problem.png" alt="XKCD: The General Problem" width="400">
  </a>
  <br>
  <em>Our approach to ML automation and the general problem</em>
</p>

Applying Linux kernel development methodologies to machine learning research for rapid iteration and reproducible experimentation. Kconfig-driven configuration, defconfig presets, Makefile automation, and rigorous test matrices enable fast prototyping of transformer architectures, pruning algorithms, and optimization techniques while maintaining reproducibility and collaboration at scale.

> **🚀 Reciprocal Attention + KVSplice**: Current research on bidirectional attention mechanisms and learned cache compression. Reciprocal Attention (RA) alternates Q@K.T and K@Q.T across layers, achieving 27% inference speedup with smoother optimization. KVSplice provides learned compression of attention cache with 12x memory reduction while improving quality. See [docs/ra.md](docs/ra.md) for RA details and [docs/kvsplice.md](docs/kvsplice.md) for compression techniques.

> **⚡ Adam State-Based Pruning**: bitter7 achieves **15.6% better perplexity** than magnitude baseline (37.28 vs 44.15 PPL), validating the hypothesis that Adam's gradient statistics enable superior pruning decisions. Tested on NVIDIA B200 GPUs with torch.compile.

## Development Philosophy

knlp applies **Linux kernel development practices** to machine learning research:

- **Kconfig-based configuration**: Hierarchical menus for experiment management (like `make menuconfig`)
- **Defconfig presets**: Reproducible configurations for different hardware and research goals
- **Makefile-driven builds**: Consistent build and test workflows across models
- **Documented decisions**: Every architectural choice explained in `docs/`
- **Collaborative contributions**: Community-driven ideas and ablation studies
- **Rigorous validation**: Automated test matrices before merging experiments

This methodology enables rapid iteration on transformer architectures through reciprocal mechanisms, geometric compression, and state-based optimization while maintaining reproducibility and rigor.

## Key Results

### Attention Architecture: RA + kvsplice R&D Quality and Speed Improvements

![KVSplice Quality Speed Tradeoff](docs/images/kvsplice_quality_speed_tradeoff.png)

Our attention R&D demonstrates that learned compression and reciprocal
mechanisms achieve both quality improvements and inference speedup
simultaneously. **MLA+KVSplice** delivers best quality (3.2 perplexity, 11%
better than MLA alone) with 12x cache compression. **RA+MLA+KVSplice**
achieves baseline GPT-2 quality (3.3 perplexity) with 12x compression and 22%
faster inference - the sweet spot for production deployment.

See [docs/ra.md](docs/ra.md) for Reciprocal Attention details and
[docs/kvsplice.md](docs/kvsplice.md) for learned compression analysis.

### Adam State based Pruning R&D results

| Model | Parameters | Dataset | Sparsity | Accuracy/Perplexity | Notes |
|-------|------------|---------|----------|---------------------|-------|
| GPT-2 | 124M | FineWebEdu | 50% | **37.28 PPL** | **bitter7 (15.6% better)** |
| ResNet-50 | 25.6M | CIFAR-100 | 50% | 74.56% | bitter0 (original hybrid) |
| ResNet-18 | 11.2M | CIFAR-10 | 70% | 90.66% | bitter0 (original hybrid) |
| LeNet-5 | 61.7K | MNIST | 70% | 98.9% | bitter0 (original hybrid) |

bitter0 (original hybrid momentum-stability) achieved excellent results
on CNNs. bitter7 (variance-based) emerged from transformer R&D and is
expected to improve CNN results further. See evolution story below.

## GPT-2 Transformer Results (124M Parameters)

### Current Research: Reciprocal Attention + KVSplice

**Reciprocal Attention (RA)**: Bidirectional attention mechanism that alternates
Q@K.T (standard) and K@Q.T (reciprocal) across transformer layers. Achieves 27%
inference speedup with improved optimization geometry.

**KVSplice**: Learned compression technique for attention cache. Reduces cache
size by 12x while improving model quality through regularization effect.

**Combined results** (RA+MLA+KVSplice):
- 12x cache compression (36 MB → 3 MB)
- 22% inference speedup
- Matches baseline GPT-2 quality
- 2-hour training runs on AMD W7900 GPU

See [docs/ra.md](docs/ra.md) for Reciprocal Attention details and
[docs/kvsplice.md](docs/kvsplice.md) for compression architecture.

---

### Adam State-Based Pruning: Hypothesis Validated

Our Adam state-based pruning research conclusively validates the
hypothesis that **leveraging Adam's accumulated gradient statistics
enables superior pruning decisions** compared to magnitude-based
approaches. State-based variants (bitter7, bitter8) significantly
outperform magnitude pruning baseline when tested with identical
hyperparameters on NVIDIA B200 GPUs.

![AdamWPrune Comparison](images/adamwprune_fair_comparison.png)
*State-based pruning outperforms magnitude baseline with identical
hyperparameters. All runs WITH torch.compile on B200: bitter8
achieves 40.94 PPL (7.3% better), bitter7 achieves 37.28 PPL
(15.6% better) than movement pruning baseline (44.15 PPL).*

**Test Configuration:**
- Model: GPT-2 (124M parameters)
- Dataset: FineWebEdu
- Target Sparsity: 50%
- **Learning Rate:** 0.0006, Weight Decay: 0.1
- **Hyperparameters:** AUTO mode (adapts to available hardware)

See [docs/pruning.md](docs/pruning.md) for complete pruning research details,
hyperparameter auto-detection, ResNet results, and transformer findings.

---

## Research Areas

knlp serves as a collaborative platform for ML architecture research:

- **[Reciprocal Attention (RA)](docs/ra.md)**: Bidirectional attention alternating Q@K.T and K@Q.T across layers, achieving 27% inference speedup with improved optimization geometry
- **[KVSplice](docs/kvsplice.md)**: Learned compression for attention cache achieving 12x compression while improving model quality through regularization effect
- **[AdamWPrune](docs/pruning.md)**: State-based pruning leveraging Adam optimizer state variables for zero-overhead pruning decisions during training
- **[Weight Tying](docs/weight-tying.md)**: Parameter reduction through strategic sharing
- **[KV Tying](docs/kv-tying.md)**: Attention projection parameter reduction
- **[Mechanistic Interpretability](docs/mechint.md)**: Post-training circuit analysis

See [docs/architecture.md](docs/architecture.md) for details on the kernel-inspired infrastructure.

## Getting Started

### System Requirements

For systems using `torch.compile()`, Python development headers are required:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# RHEL/CentOS/Fedora
sudo yum install python3-devel
```

### Installation

```bash
pip install -r requirements.txt
wandb login # optional
```

### Run Your First Experiment

```bash
make defconfig-gpt2-vanilla-baseline
make
```

That's it! The experiment will run and save results automatically.

See [docs/quickstart.md](docs/quickstart.md) for detailed workflow and
advanced options. See [docs/tracker.md](docs/tracker.md) for experiment
tracking with WandB or Trackio.

## Contributing

knlp welcomes contributions following Linux kernel development practices:

### Proposing Ideas
1. **Open an issue** with your research idea or architectural proposal
2. **Provide motivation**: Why this approach might work better
3. **Reference prior work**: Links to papers or existing implementations
4. **Suggest ablation**: How to test your hypothesis

### Submitting Code
1. **Create defconfig**: Add `defconfigs/<model>-<your-feature>`
2. **Document thoroughly**: Add or update `docs/<feature>.md`
3. **Run dry-run**: `make check` to validate architecture
4. **Test ablation**: Run comparison vs baseline
5. **Submit PR**: With results, graphs, and documentation

### Code Style
- **Python**: Follow existing style (black formatter)
- **Kconfig**: Use kernel-style help text
- **Commit messages**: Terse, technical, with "Generated-by" tags
- **Documentation**: See `docs/*.md` for style examples

### Current Contributors
Ideas, architectures, and validation from:
- Luis Chamberlain (maintainer)
- Community contributors (ablation suggestions, architectural ideas)

## Citation

If you use this work, please cite:

```bibtex
@misc{knlp2025,
  title        = {knlp: Kernel-Style Machine Learning - Transformer Architecture Research},
  author       = {Luis Chamberlain and contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/mcgrof/knlp}},
  note         = {Collaborative ML research using Linux kernel development workflows}
}
```

## License

This project is licensed under the **MIT License**.

- **Code**: MIT license
- **Models**: AI models generated by this project can be licensed as you choose
- **Documentation**: CC-BY-SA 4.0 (collaborative, share-alike)

The project previously included GPLv2-licensed C-based Kconfig tools from the Linux kernel. These have been replaced with the Python-based `kconfiglib` library (ISC/Apache 2.0), enabling full MIT licensing for the entire project.

See LICENSE for details.

## Acknowledgments

This project draws inspiration from:
- **Linux kernel development**: Kconfig, defconfigs, Makefile patterns, rigorous testing
- **Andrej Karpathy's nanoGPT**: Clean implementation style, educational focus
- **Community contributors**: Ablation study ideas, architectural suggestions, validation testing

The kernel-style workflow enables collaborative ML research with the rigor and reproducibility of systems programming.

