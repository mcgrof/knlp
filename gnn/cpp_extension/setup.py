"""
Build script for C++ sampler extensions.

Usage:
    cd cpp_extension
    pip install -e .

Or for development:
    python setup.py build_ext --inplace

Extensions:
    - page_aware_sampler_cpp: K-hop neighbor sampling with page priority
    - cpp_page_batch_sampler: Page-batch sampling (entire pages as batches)
    - amex_sampler_cpp: AMEX credit-default streaming benchmark samplers
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def get_extensions():
    extensions = []

    # Detect if OpenMP is available
    extra_compile_args = ['-O3', '-std=c++17']
    extra_link_args = []

    # Try to enable OpenMP
    if os.name == 'posix':  # Linux/Mac
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp')

    # Page-aware neighbor sampler (K-hop with page priority)
    extensions.append(
        CppExtension(
            name='page_aware_sampler_cpp',
            sources=['page_aware_sampler.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )

    # Page-batch sampler (entire pages as batches, intra-page edges only)
    extensions.append(
        CppExtension(
            name='cpp_page_batch_sampler',
            sources=['page_batch_sampler.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )

    # AMEX credit-default streaming benchmark samplers
    extensions.append(
        CppExtension(
            name='amex_sampler_cpp',
            sources=['amex_sampler.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )

    return extensions


setup(
    name='page_samplers_cpp',
    version='1.1.0',
    description='C++ extensions for page-aware and page-batch sampling',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    python_requires='>=3.8',
)
