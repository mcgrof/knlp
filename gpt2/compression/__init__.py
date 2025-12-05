"""
KV Cache Compression Plugin System

Pluggable compression for open-weight causal LMs with minimal surgery.
"""

from gpt2.compression.base import KVCompressorBase
from gpt2.compression.kvsplice import KVSpliceCompressor
from gpt2.compression.pca import PCACompressor
from gpt2.compression.wrapper import CompressedKVModelWrapper

__all__ = [
    "KVCompressorBase",
    "KVSpliceCompressor",
    "PCACompressor",
    "CompressedKVModelWrapper",
]
