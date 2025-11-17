# TorchVision ROCm Patches

This directory contains patches needed to fix issues with torchvision on ROCm.

## torchvision-rocm-nms-fix.patch

**Problem**: TorchVision 0.20.x on ROCm 6.2 fails to import with error:
```
RuntimeError: operator torchvision::nms does not exist
```

This occurs because `_meta_registrations.py` tries to register fake implementations
for operators before the C++ extensions are loaded.

**Fix**: Comment out the problematic NMS operator registration in:
```
$VIRTUAL_ENV/lib/python*/site-packages/torchvision/_meta_registrations.py
```

**To apply manually**:
```bash
sed -i '163,174s/^/# /' $(python3 -c "import torchvision; import os; print(os.path.dirname(torchvision.__file__))")/_meta_registrations.py
```

**Impact**: This disables the fake NMS operator registration, which is only used
for shape inference in certain compilation scenarios. ResNet models don't use NMS,
so this has no impact on training.

**When to reapply**: After reinstalling torchvision or upgrading PyTorch/ROCm.

**Upstream issue**: Known bug in torchvision 0.20.x on ROCm that should be fixed
in future releases.
