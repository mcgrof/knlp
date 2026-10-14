<div align="center">

# LayerRoPE

### Dynamic Depth-wise Magnitude & Angular Superposition

[Shikhar Srivastava](https://www.cs.rochester.edu/u/ssrivas9/) &nbsp;·&nbsp; [Christopher Kanan](http://chriskanan.com)<br>
University of Rochester

*Under review* &nbsp;·&nbsp; *Actionable Interpretability Workshop @ CoLM 2026*

<a href="https://arxiv.org/abs/2610.09179"><img src="https://img.shields.io/badge/arXiv-2610.09179-b31b1b?style=flat-square&logo=arxiv&logoColor=white" height="26" alt="arXiv 2610.09179"></a>&nbsp;
<a href="https://www.cs.rochester.edu/u/ssrivas9/layer-rope/"><img src="https://img.shields.io/badge/Project-Page-0969da?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyBmaWxsPSJ3aGl0ZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiI%2BPHBhdGggZD0iTTggMGE4IDggMCAxIDEgMCAxNkE4IDggMCAwIDEgOCAwWk01Ljc4IDguNzVhOS42NCA5LjY0IDAgMCAwIDEuMzYzIDQuMTc3Yy4yNTUuNDI2LjU0Mi44MzIuODU3IDEuMjE1LjI0NS0uMjk2LjU1MS0uNzA1Ljg1Ny0xLjIxNUE5LjY0IDkuNjQgMCAwIDAgMTAuMjIgOC43NVptNC40NC0xLjVhOS42NCA5LjY0IDAgMCAwLTEuMzYzLTQuMTc3Yy0uMzA3LS41MS0uNjEyLS45MTktLjg1Ny0xLjIxNWE5LjkyNyA5LjkyNyAwIDAgMC0uODU3IDEuMjE1QTkuNjQgOS42NCAwIDAgMCA1Ljc4IDcuMjVabS01Ljk0NCAxLjVIMS41NDNhNi41MDcgNi41MDcgMCAwIDAgNC42NjYgNS41Yy0uMTIzLS4xODEtLjI0LS4zNjUtLjM1Mi0uNTUyLS43MTUtMS4xOTItMS40MzctMi44NzQtMS41ODEtNC45NDhabS0yLjczMy0xLjVoMi43MzNjLjE0NC0yLjA3NC44NjYtMy43NTYgMS41OC00Ljk0OC4xMi0uMTk3LjIzNy0uMzgxLjM1My0uNTUyYTYuNTA3IDYuNTA3IDAgMCAwLTQuNjY2IDUuNVptMTAuMTgxIDEuNWMtLjE0NCAyLjA3NC0uODY2IDMuNzU2LTEuNTggNC45NDgtLjEyLjE5Ny0uMjM3LjM4MS0uMzUzLjU1MmE2LjUwNyA2LjUwNyAwIDAgMCA0LjY2Ni01LjVabTIuNzMzLTEuNWE2LjUwNyA2LjUwNyAwIDAgMC00LjY2Ni01LjVjLjEyMy4xODEuMjQuMzY1LjM1My41NTIuNzE0IDEuMTkyIDEuNDM2IDIuODc0IDEuNTggNC45NDhaIi8%2BPC9zdmc%2B" height="26" alt="Project page"></a>

<a href="https://github.com/shikhar-srivastava/parcae-layerrope"><img src="https://img.shields.io/badge/Looped_LMs-Parcae_%2B_LayerRoPE-8250df?style=flat-square&logo=github&logoColor=white" height="26" alt="LayerRoPE in Parcae"></a>&nbsp;
<a href="https://github.com/shikhar-srivastava/deit-layerrope"><img src="https://img.shields.io/badge/Vision_Transformers-DeiT_%2B_LayerRoPE-1a7f37?style=flat-square&logo=github&logoColor=white" height="26" alt="LayerRoPE in DeiT"></a>

<a href="https://github.com/KellerJordan/modded-nanogpt/pull/385"><img src="https://img.shields.io/badge/Speedrun-modded--nanogpt-bc4c00?style=flat-square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyBmaWxsPSJ3aGl0ZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiI%2BPHBhdGggZD0iTTMuMjE3IDYuOTYyQTMuNzUgMy43NSAwIDAgMSAwIDMuMjV2LS41QzAgMS43ODQuNzg0IDEgMS43NSAxaDEuMzU2Yy4yMjgtLjU4NS43OTYtMSAxLjQ2Mi0xaDYuODY0Yy42NDcgMCAxLjIyNy4zOTcgMS40NjIgMWgxLjM1NmMuOTY2IDAgMS43NS43ODQgMS43NSAxLjc1di41YTMuNzUgMy43NSAwIDAgMS0zLjIxNyAzLjcxMiA1LjAxNCA1LjAxNCAwIDAgMS0yLjc3MSAzLjExN2wuMTQ0IDEuNDQ2Yy4wMDUuMDUuMDMuMTIuMTE0LjIwNC4wODYuMDg3LjIxNy4xNy4zNzMuMjI3LjI4My4xMDMuNjE4LjI3NC44OS41NjguMjg1LjMxLjQ2Ny43MjMuNDY3IDEuMjI2di43NWgxLjI1YS43NS43NSAwIDAgMSAwIDEuNUgyLjc1YS43NS43NSAwIDAgMSAwLTEuNUg0di0uNzVjMC0uNTAzLjE4Mi0uOTE2LjQ2OC0xLjIyNi4yNy0uMjk0LjYwNi0uNDY1Ljg4OS0uNTY4LjEzOS0uMDQ4LjI2Ni0uMTI2LjM3My0uMjI3LjA4NC0uMDg1LjEwOS0uMTUzLjExNC0uMjA0bC4xNDQtMS40NDZhNS4wMTUgNS4wMTUgMCAwIDEtMi43Ny0zLjExN1pNNC41IDEuNTY4VjUuNWEzLjUgMy41IDAgMSAwIDcgMFYxLjU2OGEuMDY4LjA2OCAwIDAgMC0uMDY4LS4wNjhINC41NjhhLjA2OC4wNjggMCAwIDAtLjA2OC4wNjhabTIuOTU3IDguOTAyLS4xMiAxLjIwNGMtLjA5My45MjUtLjg1OCAxLjQ3LTEuNDY3IDEuNjkxYS43NjYuNzY2IDAgMCAwLS4zLjE3NmMtLjAzNy4wNC0uMDcuMDkzLS4wNy4yMXYuNzVoNXYtLjc1YzAtLjExNy0uMDMzLS4xNy0uMDctLjIxYS43NjYuNzY2IDAgMCAwLS4zLS4xNzZjLS42MDktLjIyMS0xLjM3NC0uNzY2LTEuNDY2LTEuNjlsLS4xMi0xLjIwNGE1LjA2NCA1LjA2NCAwIDAgMS0xLjA4NyAwWk0xMyAyLjV2Mi44NzJhMi4yNSAyLjI1IDAgMCAwIDEuNS0yLjEyMnYtLjVhLjI1LjI1IDAgMCAwLS4yNS0uMjVIMTNabS0xMCAwSDEuNzVhLjI1LjI1IDAgMCAwLS4yNS4yNXYuNWMwIC45OC42MjYgMS44MTMgMS41IDIuMTIyWiIvPjwvc3ZnPg%3D%3D" height="26" alt="LayerRoPE in the modded-nanogpt speedrun"></a>

<br>

<img src="assets/layerrope_highlight.png" width="100%" alt="Depth-conditioned read and write gains in LayerRoPE (left) and compute scaling on the model ladder (right)">

<sub><b>Left:</b> LayerRoPE conditions each block's read and write gains on depth while maximizing residual variance. <b>Right:</b> compute scaling on a 4× Chinchilla model ladder; LayerRoPE matches Pre-Norm's 1.3B loss with 3.4× less compute.</sub>

</div>

<br>

Code for the model-ladder and depth-scaling experiments of the paper.

LayerRoPE replaces the per-layer normalization gains with one shared gain $\gamma$ per site (the input and output of the attention and MLP blocks), scaled and rotated by a depth-conditioned complex factor:

```math
\begin{gathered}
\gamma_\ell = \gamma \odot_c \exp\big(r(\ell) + i\,\theta(\ell, j)\big), \\
r(\ell) = \alpha + \beta \log(\ell + 1), \qquad \theta(\ell, j) = \exp\big(\alpha_{\mathrm{rot}} + \beta_{\mathrm{rot}} \log(\ell + 1)\big)\, b^{-2j/d}
\end{gathered}
```

where $\odot_c$ multiplies the pairs $(\gamma_{2j}, \gamma_{2j+1})$ as complex numbers. The method is the `LayerRoPE` class in [`peft_pretraining/modeling_llama.py`](peft_pretraining/modeling_llama.py). This repository is built on [LayerNorm-Scaling](https://github.com/lmsdss/LayerNorm-Scaling). The looped-model and ViT experiments are in our forks of [Parcae](https://github.com/shikhar-srivastava/parcae-layerrope) and [DeiT](https://github.com/shikhar-srivastava/deit-layerrope).

## Setup

```bash
conda create -n layerrope python=3.10 -y && conda activate layerrope
pip install -r requirements.txt
huggingface-cli login   # for the meta-llama/Llama-2-7b-hf tokenizer
wandb login
```

## Model ladder

```bash
bash run_130m.sh pre              # Pre-Norm
bash run_130m.sh post             # Post-Norm
bash run_130m.sh peri             # Peri-Norm
bash run_130m.sh lns              # LayerNorm Scaling
bash run_130m.sh layerrope        # LayerRoPE (on Pre-Norm)
bash run_130m.sh layerrope_peri   # LayerRoPE on Peri-Norm
```

`run_60m.sh`, `run_250m.sh`, `run_500m.sh` and `run_1b.sh` take the same methods. Each method runs at its tuned peak learning rate<sup>†</sup>:

| Model | Pre    | Post     | Peri   | LNS    | LayerRoPE | LayerRoPE + Peri |
|:------|:------:|:--------:|:------:|:------:|:---------:|:----------------:|
| 60M   | 1.6e-2 | 2e-3     | 3.2e-2 | 1.6e-2 | 6.4e-2    | 6.4e-2           |
| 130M  | 8e-3   | 2e-3     | 8e-3   | 1.6e-2 | 3.2e-2    | 3.2e-2           |
| 250M  | 8e-3   | 1e-3     | 1.6e-2 | 1.6e-2 | 6.4e-2    | 3.2e-2           |
| 500M  | 4e-3   | 6.25e-5  | 1.6e-2 | 8e-3   | 3.2e-2    | 8e-3             |
| 1B    | 2e-3   | 3.125e-5 | 8e-3   | 8e-3   | 3.2e-2    | 4e-3             |

<sub><sup>†</sup> See the model-ladder learning-rate sweeps in Appendix D.1.2 of the [paper](https://arxiv.org/abs/2610.09179) for details.</sub>

Set the learning rate with `--lr` (e.g. for the learning-rate sweeps) and the seed with `--seed`; any other argument is passed on to `torchrun_main.py`:

```bash
bash run_130m.sh layerrope --lr 1.6e-2 --seed 2 --layerrope_beta_init -1.0
```

The scripts train on 4 GPUs (8 for 1B; set `NGPU`) with a total batch of 512 sequences of 256 tokens; `--batch_size` sets the per-GPU micro-batch.

## Depth scaling

```bash
for L in 48 96 192 256 384 512; do bash run_depth.sh layerrope $L; done
bash run_depth.sh deepnet 256 --deepnet_depth_alpha   # depth-corrected DeepNet
```

The methods are `pre`, `post`, `peri`, `lns`, `deepnet` and `layerrope`, at a shared learning rate of 1e-3 (`--lr` for the per-depth sweep). DeepNet is LayerNorm-Scaling's implementation, which scales the residual by a fixed $\alpha = (2 \cdot 32)^{1/4}$, DeepNet's value for 32 layers; we correct its initialization gain to DeepNet's $(8L)^{-1/4}$ (LayerNorm-Scaling uses $(8L)^{1/4}$). We also include a depth-corrected DeepNet, with $\alpha = (2L)^{1/4}$ for $L$ layers, set with `--deepnet_depth_alpha`; our per-depth sweep used it.

## LayerRoPE options

| Flag | Default | Parameter |
|:-----|:-------:|:----------|
| `--layerrope_alpha_init`     | 0    | $\alpha$ |
| `--layerrope_beta_init`      | −0.5 | $\beta$, the magnitude depth slope |
| `--layerrope_alpha_rot_init` | 0    | $\alpha_{\mathrm{rot}}$ |
| `--layerrope_beta_rot_init`  | −0.5 | $\beta_{\mathrm{rot}}$, the rotation depth slope |
| `--layerrope_base_freq`      | 100  | $b$ |

The defaults are the model-ladder configuration; `run_depth.sh` uses $\beta = \beta_{\mathrm{rot}} = 0$ and $b = 10^4$. For a new setting, it suffices to try $\beta, \beta_{\mathrm{rot}} \in \lbrace -0.5, 0 \rbrace$ and $b \in \lbrace 10^2, 10^4 \rbrace$; no further tuning is needed.

## Notes

- **Precision.** `--rmsnorm_fp32` keeps every normalization gain (RMSNorm weights and LayerRoPE's parameters) in fp32. We used it in all experiments here to avoid bf16 precision issues, given how much the norm gains matter across methods. Disable it with `--no-rmsnorm_fp32`; our ViT and Parcae experiments, for instance, ran without it.
- **1B.** The 1B runs take 1.15 passes over C4 (`--multi_epoch`) and use fp32 output logits from the start. Our original 1B runs switched to fp32 logits midway through training, from a checkpoint (see Appendix D.1).
- **Losses.** `eval_loss` (the model's own output head) and `eval_loss_fp32` (fp32 output logits) are means over a fixed 10M-token sample of C4 validation. The paper's model-ladder figures report `eval_loss_fp32`, its depth-scaling figures `eval_loss`.
- **Data.** C4 is streamed. Our runs read a local copy instead: to reproduce their exact data order, pass `--c4_cache_dir DIR` (C4 is downloaded there once) and keep the scripts' GPU count and per-GPU batch.
- **Depth runs** use the upstream LayerNorm-Scaling data loader (`--legacy_dataloader`), as ours did; see [`peft_pretraining/dataloader.py`](peft_pretraining/dataloader.py).
- **Resume** by rerunning the same command with `--continue_from checkpoints/<run>/model_<step>`.
- **Embedding init** has standard deviation $\sqrt{2/5}$, as in the original LayerNorm-Scaling code (upstream now uses 0.02).

## Citation

```bibtex
@misc{srivastava2026layerrope,
  title         = {LayerRoPE: Dynamic Depth-wise Magnitude \& Angular Superposition},
  author        = {Srivastava, Shikhar and Kanan, Christopher},
  year          = {2026},
  eprint        = {2610.09179},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.09179}
}
```

## Acknowledgement

This repository builds on [LayerNorm-Scaling](https://github.com/lmsdss/LayerNorm-Scaling), [GaLore](https://github.com/jiaweizzhao/GaLore) and [Mix-LN](https://github.com/pixeli99/MixLN).
