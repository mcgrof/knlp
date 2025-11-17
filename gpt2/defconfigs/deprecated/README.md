# Deprecated Defconfigs

These defconfigs are **deprecated** and should not be used for new work.

## Why Deprecated?

These defconfigs were GPU-specific (b200x4, w7900, a100, etc.) and required
maintaining separate configs for each hardware platform.

## Replacement

Use the **hardware-agnostic** defconfigs in the parent directory:

- `gpt2-finewebedu-bitter7` - Single bitter7 test (replaces all bitter7-* variants)
- `gpt2-finewebedu-bitter8` - Single bitter8 test (replaces all bitter8-* variants)
- `gpt2-finewebedu-spam-vs-bitter7` - Test matrix comparing methods

These new defconfigs use:
- **CONFIG_HYPER_PARAM_AUTO=y** - Auto-detects batch size and gradient accumulation
- **CONFIG_COMPILE_AUTO=y** - Auto-detects torch.compile() based on GPU

The same defconfig now works on B200, W7900, A10G, and other GPUs without modification.

## Migration

If you were using:
- `gpt2-finewebedu-b200x4-bitter7` → Use `gpt2-finewebedu-bitter7`
- `gpt2-finewebedu-w7900-bitter7-only` → Use `gpt2-finewebedu-bitter7`
- `gpt2-finewebedu-b200x4-spam-vs-bitter7` → Use `gpt2-finewebedu-spam-vs-bitter7`

## Historical Reference

These files are kept for historical reference only. They document the hyperparameters
used in previous experiments before auto-detection was implemented.
