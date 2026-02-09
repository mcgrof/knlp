# BPA Migration Note: RGSA → Boundary-Pressure Attention

## Summary

RGSA (Route-Gated Sparse Attention) is deprecated as the primary name.
The correct abstraction is **Boundary-Pressure Attention (BPA)**.

## Background

The RGSA line of research (v13-v19) established:

1. **Static allocation fails**: Head-level budget reallocation based on
   variance-weighting or inverted Fisher proxies is dominated by seed
   variance and does not reliably improve quality.

2. **Adam exp_avg_sq measures wrong thing**: Diagonal Fisher proxy
   measures parameter update geometry, NOT runtime state importance.

3. **Importance is query-conditional**: Head importance varies per
   (position, token, context), not statically per head.

4. **boundary_pressure is the key signal**: v19 discovered that
   "attention mass at local boundary" (how much attention tries to
   escape the local window) predicts when far-context access matters:
   - Spearman r = 0.58
   - ROC-AUC = 0.71
   - Stable across random seeds

5. **Threshold gating works**: Enabling far-context only when
   boundary_pressure > threshold achieves 24x KL alignment with 56%
   compute savings.

## Terminology Change

| Old Term | New Term | Meaning |
|----------|----------|---------|
| RGSA | BPA | Boundary-Pressure Attention |
| RGSAConfig | BPAConfig | Configuration for BPA |
| GPT2_RGSA | GPT2_BPA | BPA-enabled GPT-2 model |
| rgsa.py | bpa.py | BPA module (imports from rgsa.py) |

## What BPA Is

BPA is attention where:
- Local window is always available (no compute savings there)
- Far-context access is conditionally enabled per (layer, head, token)
- Gating is based on boundary_pressure signal
- Default policy is threshold-based (not top-k)

## What BPA Is NOT

- NOT a routing mechanism (route-gated was a misnomer)
- NOT static budget allocation (that approach failed)
- NOT based on Fisher/gradient information
- NOT kernel-optimized (no wall-clock claims yet)

## File Locations

After migration:
- `gpt2/bpa.py`: Primary BPA module (wrapper around rgsa.py internals)
- `gpt2/rgsa.py`: Legacy implementation, still functional
- `docs/bp_v1_migration_note.md`: This document

## W&B Naming

- Old: `rgsa-*` groups
- New: `bpa-*` groups
- Reports: `bp-v1_*` prefix

## Usage

```python
# New style (preferred)
from gpt2.bpa import BPAConfig, GPT2_BPA

# Legacy style (still works)
from gpt2.rgsa import RGSAConfig, GPT2_RGSA
```

## References

- rgsa-v19.txt: Query-Conditional Importance discovery
- rgsa_v19_results/rgsa_v19_final_report.md: Full v19 report
- bp-v1.txt: BPA risk reduction plan
