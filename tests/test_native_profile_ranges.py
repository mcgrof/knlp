# SPDX-License-Identifier: MIT
from tools.kv.native.profile_ranges import join_cuda_events


def test_correlate_async_gpu_time_without_using_host_duration():
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "name": "knlp|prefill|7|layer0",
                "pid": 1,
                "tid": 2,
                "ts": 0,
                "dur": 1000,
            },
            {
                "cat": "cuda_driver",
                "name": "cuLaunchKernel",
                "pid": 1,
                "tid": 2,
                "ts": 100,
                "dur": 10,
                "args": {"correlation": 42},
            },
            {
                "cat": "kernel",
                "name": "attention",
                "ts": 2000,
                "dur": 3,
                "args": {"correlation": 42},
            },
            {
                "cat": "kernel",
                "name": "unmatched",
                "ts": 3000,
                "dur": 99,
                "args": {"correlation": 99},
            },
        ]
    }
    result = join_cuda_events(trace)
    group = result["groups"][0]
    assert result["matched_cuda_kernels"] == 1 and result["unmatched_cuda_kernels"] == 1
    assert group["device_kernel_sum_us"] == 3 and group["device_kernel_union_us"] == 3
    assert group["host_range_us"] == 1000 and group["host_cuda_api_us"] == 10
