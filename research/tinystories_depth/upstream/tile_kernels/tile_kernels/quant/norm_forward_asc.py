import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig, get_sf_shape
from tile_kernels.quant.rstd import make_rstd_impl
from tile_kernels.utils import align, ceil_div


@T.macro
def load_f32(buffer, col):
    """Load 64 contiguous elements as FP32."""
    if buffer.dtype == T.bfloat16:
        return S.vcvt(S.vld(buffer[col], dist='UNPK_B16'), T.float32, part=0)
    return S.vld(buffer[col])


@T.macro
def load_f32_pair(buffer, col):
    """Load BF16 even/odd lanes or FP32 contiguous halves as two FP32 vectors."""
    if buffer.dtype == T.bfloat16:
        packed = S.vld(buffer[col])
        return S.vcvt(packed, T.float32, part=0), S.vcvt(packed, T.float32, part=1)
    return S.vld(buffer[col]), S.vld(buffer[col + 64])


@T.macro
def store_pair(buffer, col, x0, x1):
    """Store a pair in the same lane order as load_f32_pair."""
    if buffer.dtype == T.bfloat16:
        low = S.vcvt(x0, T.bfloat16, part=0)
        high = S.vcvt(x1, T.bfloat16, part=1)
        packed = S.vor(T.reinterpret(low, 'uint16x128'), T.reinterpret(high, 'uint16x128'))
        S.vsts(buffer[col], T.reinterpret(packed, 'bfloat16x128'), dist='NORM_B16')
    else:
        S.vsts(buffer[col], x0)
        S.vsts(buffer[col + 64], x1)


@T.macro
def get_sf_and_inv_simd(amax, out_config: CastOutputConfig):
    """SIMD counterpart of get_sf_and_inv; packed SF is returned in uint32 lanes."""
    clamped = S.vmaxs(amax, out_config.clamp_min_value)
    max_value = float(T.max_value(out_config.dtype))
    if not out_config.round_sf:
        maximum = S.vdup(max_value, T.float32)
        return S.vdiv(clamped, maximum), S.vdiv(maximum, clamped)
    bits = T.reinterpret(S.vmuls(clamped, 1.0 / max_value), 'uint32x64')
    exponent = S.vadds(S.vshrs(S.vsub(bits, S.vdup(1, T.uint32)), 23), 1)
    inverse = T.reinterpret(S.vshls(S.vsub(S.vdup(254, T.uint32), exponent), 23), 'float32x64')
    if out_config.use_packed_ue8m0:
        return exponent, inverse
    return T.reinterpret(S.vshls(exponent, 23), 'float32x64'), inverse


@tilelang.jit
def get_norm_forward_kernel_asc(
    num_tokens: int,
    hidden: int,
    with_weight: bool,
    with_residual: bool,
    eps: float,
    out_scale: float,
    x_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
    with_out_bf16: bool = False,
    mega_moe_block_m: int = 0,
):
    assert hidden > 0
    assert x_dtype in (T.bfloat16, T.float32)
    assert not out_config.with_sf or out_config.sf_block == (1, 32)
    padded_hidden = align(hidden, 128)
    row_bytes = padded_hidden * x_dtype.bytes
    vec_size = 64
    load_size = 256 // x_dtype.bytes
    num_parts = load_size // vec_size
    num_accumulators = min(4, padded_hidden // vec_size)
    reduction_cols = num_accumulators * vec_size

    # SF layout only affects stores. All formats share normalization and quantization.
    with_mega_sf = mega_moe_block_m > 0
    mega_sf_tile_rows = 128
    mega_sf_lane_rows = 32
    mega_sf_cols = mega_sf_tile_rows // mega_sf_lane_rows
    transpose_sf = out_config.with_sf and out_config.use_tma_aligned_col_major_sf
    num_groups = hidden // 32 if out_config.with_sf else 0
    # Mega SF packs four exponent bytes per word; scatter processes 64 words at a time.
    sf_size = align(max(1, num_groups), vec_size * T.uint32.bytes if with_mega_sf else vec_size)
    # Transpose SF as raw bits: pairs of exponent bytes or individual FP32 scales.
    sf_word_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.uint32
    sf_pack = sf_word_dtype.bytes // out_config.sf_dtype.bytes
    sf_lanes = 256 // sf_word_dtype.bytes
    sf_store_dist = f'NORM_B{sf_word_dtype.bits}'
    num_sf_words = ceil_div(num_groups, sf_pack)
    mega_dma_stride = 32 // T.uint32.bytes

    # Group short RSTD stores and bound each core's work.
    rows_per_core = max(min(num_tokens, 4), num_tokens // num_vec_cores)
    rows_per_block = min(128, 1 << (rows_per_core.bit_length() - 1))
    if transpose_sf:
        rows_per_block = min(sf_lanes // 4, rows_per_block)
    # Target 32 KiB of input per tile, shared by x and residual.
    tile_capacity = max(1, 32768 // (1 + with_residual) // row_bytes)
    rows_per_tile = min(vec_size, rows_per_block, 1 << (tile_capacity.bit_length() - 1))
    # Shrink the tile until all buffers fit in UB, including optional SF outputs.
    while True:
        rows_per_block = min(rows_per_block, (64 if out_config.with_sf else 32) * rows_per_tile)
        sf_rows = max(rows_per_block, 32 // sf_word_dtype.bytes) if transpose_sf else rows_per_tile
        rstd_out_size = max(vec_size, rows_per_block)

        fixed_bytes = row_bytes * with_weight + (vec_size + rstd_out_size) * T.float32.bytes
        if out_config.with_sf:
            fixed_bytes += rows_per_tile * sf_size * T.float32.bytes
        if transpose_sf:
            fixed_bytes += 2 * sf_rows * sf_size * out_config.sf_dtype.bytes + vec_size * T.float32.bytes
        tile_bytes = rows_per_tile * row_bytes
        stage_bytes = (2 + with_residual) * tile_bytes
        if out_config.with_sf:
            stage_bytes += rows_per_tile * padded_hidden * out_config.dtype.bytes
            if not transpose_sf:
                stage_bytes += rows_per_tile * sf_size * out_config.sf_dtype.bytes
        if with_out_bf16 and x_dtype != T.bfloat16:
            stage_bytes += rows_per_tile * padded_hidden * T.bfloat16.bytes
        if with_mega_sf:
            fixed_bytes += rows_per_tile * sf_size * T.uint8.bytes
            stage_bytes += rows_per_tile * (sf_size // T.uint32.bytes) * 32
        available_bytes = get_max_ub_per_vector_core(use_simt=False) - fixed_bytes
        num_stages = min(4, available_bytes // stage_bytes)
        reused_stages = min(4, available_bytes // (stage_bytes - tile_bytes))
        reuse_input = num_stages < 3 and reused_stages > num_stages
        if reuse_input:
            num_stages = reused_stages
        if num_stages > 0:
            break
        assert rows_per_tile > 1, 'norm tile exceeds the available UB'
        rows_per_tile //= 2

    # Batches with the same tiling share compiled code, independent of row strides.
    num_tokens = T.dynamic('num_tokens')
    batch = T.dynamic('batch')
    x_stride = T.dynamic('x_stride')
    x_batch_stride = T.dynamic('x_batch_stride')
    out_stride = T.dynamic('out_stride')
    out_batch_stride = T.dynamic('out_batch_stride')
    residual_stride = T.dynamic('residual_stride')
    residual_batch_stride = T.dynamic('residual_batch_stride')
    residual_out_stride = T.dynamic('residual_out_stride')
    residual_out_batch_stride = T.dynamic('residual_out_batch_stride')

    out_bf16_stride = T.dynamic('out_bf16_stride')
    out_bf16_batch_stride = T.dynamic('out_bf16_batch_stride')
    out_sf_stride = T.dynamic('out_sf_stride')
    out_sf_batch_stride = T.dynamic('out_sf_batch_stride')
    mega_moe_sf_rows = T.dynamic('mega_moe_sf_rows')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config) if out_config.with_sf else (1, 1)

    @T.macro
    def clear_padding(buffer):
        zero = S.vdup(0.0, buffer.dtype)
        for vector in T.serial(T.uint16(hidden // load_size), T.uint16(padded_hidden // load_size)):
            indices = S.vadds(S.vci(0, T.int16 if x_dtype == T.bfloat16 else T.int32), vector * load_size)
            mask = S.vcmps(indices, hidden, op='ge')
            for row in T.serial(T.uint16(rows_per_tile)):
                S.vsts(buffer[row * padded_hidden + vector * load_size], zero, mask)

    compute_rstd = make_rstd_impl(hidden, eps)

    @T.macro
    def reduce_row(x_ub, residual_ub, row_start):
        sums = S.alloc_local((num_accumulators,), T.float32)
        for i in T.unroll(num_accumulators, explicit=True):
            sums[i] = S.vdup(0.0, T.float32)
        for group in T.serial(T.uint16(T.ceildiv(padded_hidden, reduction_cols))):
            for chunk in T.unroll(reduction_cols // load_size, explicit=True):
                col = group * reduction_cols + chunk * load_size
                if col < padded_hidden:
                    x = S.alloc_var(x_dtype)
                    x = S.vld(x_ub[row_start + col])
                    if with_residual:
                        x = S.vadd(x, S.vld(residual_ub[row_start + col]))
                        S.vsts(residual_ub[row_start + col], x)
                    # vcvt's half selector is an instruction immediate, not a loop index.
                    low = S.vcvt(x, T.float32, part=0) if x_dtype == T.bfloat16 else x
                    S.vmula(sums[chunk * num_parts], low, low)
                    if x_ub.dtype == T.bfloat16:
                        high = S.vcvt(x, T.float32, part=1)
                        S.vmula(sums[chunk * num_parts + 1], high, high)
        for i in T.unroll(1, num_accumulators, explicit=True):
            sums[0] = S.vadd(sums[0], sums[i])
        return S.vcadd(sums[0])

    @T.macro
    def apply_tile(x_ub, weight_ub, out_ub, rstd_source):
        for pair in T.serial(T.uint16(padded_hidden // 128)):
            col = pair * 128
            if with_weight:
                w0, w1 = load_f32_pair(weight_ub, col)
                if out_scale != 1.0:
                    w0, w1 = S.vmuls(w0, out_scale), S.vmuls(w1, out_scale)
            for row in T.serial(T.uint16(rows_per_tile)):
                offset = row * padded_hidden + col
                inv_rms = S.vld(rstd_source[row], dist='BRC_B32')
                x0, x1 = load_f32_pair(x_ub, offset)
                y0, y1 = S.vmul(x0, inv_rms), S.vmul(x1, inv_rms)
                if with_weight:
                    y0, y1 = S.vmul(y0, w0), S.vmul(y1, w1)
                elif out_scale != 1.0:
                    y0, y1 = S.vmuls(y0, out_scale), S.vmuls(y1, out_scale)
                store_pair(out_ub, offset, y0, y1)

    @T.macro
    def quantize_tile(source, quantized, auxiliary, scratch, scales, scale_row_base, mega_packed, mega_dma):
        with T.SimdVF():
            low_mask = S.pset(T.float32.bits, 'PAT_VL32')
            all_lanes = S.pset(T.float32.bits, 'PAT_ALL')
            high_mask = S.pnot(low_mask, all_lanes)
            for row in T.serial(T.uint16(rows_per_tile)):
                for vector in T.serial(T.uint16(T.ceildiv(num_groups, 2))):
                    col = row * padded_hidden + vector * vec_size
                    value = S.vabs(load_f32(source, col))
                    # One FP32 vector contains two per-32 scaling groups.
                    S.vsts(scratch[row, vector * 2], S.vcmax(value, low_mask), dist='ONEPT_B32')
                    S.vsts(scratch[row, vector * 2 + 1], S.vcmax(value, high_mask), dist='ONEPT_B32')
            S.mem_bar('VST_VLD')
            for row in T.serial(T.uint16(rows_per_tile)):
                for group_tile in T.serial(T.uint16(sf_size // vec_size)):
                    group = group_tile * vec_size
                    # Amax is dead after this load; reuse its storage for inverse scales.
                    sf, inv = get_sf_and_inv_simd(S.vld(scratch[row, group]), out_config)
                    S.vsts(scratch[row, group], inv)
                    if out_config.use_packed_ue8m0:
                        S.vsts(scales[scale_row_base + row, group], T.reinterpret(sf, 'uint8x256'), dist='PK4_B32')
                    else:
                        S.vsts(scales[scale_row_base + row, group], sf)
                    if with_mega_sf:
                        exponent = sf if out_config.use_packed_ue8m0 else S.vshrs(T.reinterpret(sf, 'uint32x64'), 23)
                        S.vsts(mega_packed[row, group], T.reinterpret(exponent, 'uint8x256'), dist='PK4_B32')
            S.mem_bar('VST_VLD')
            if with_mega_sf:
                packed_words = T.Tensor((rows_per_tile, sf_size // T.uint32.bytes), T.uint32, mega_packed.data)
                offsets = S.vmuls(T.reinterpret(S.vci(0, T.int32), 'uint32x64'), mega_dma_stride)
                for row in T.serial(T.uint16(rows_per_tile)):
                    for chunk in T.serial(T.uint16(sf_size // (vec_size * T.uint32.bytes))):
                        S.vscatter(S.vld(packed_words[row, chunk * vec_size]), mega_dma[row, chunk * vec_size, 0], offsets)
            for row in T.serial(T.uint16(rows_per_tile)):
                for vector in T.serial(T.uint16(padded_hidden // vec_size)):
                    col = vector * vec_size
                    group = vector * 2
                    inv0 = S.vld(scratch[row, group], dist='BRC_B32')
                    inv1 = S.vld(scratch[row, group + 1], dist='BRC_B32')
                    inv = S.vsel(inv0, inv1, low_mask)
                    value = load_f32(source, row * padded_hidden + col)
                    if with_out_bf16 and x_dtype != T.bfloat16:
                        S.vsts(auxiliary[row * padded_hidden + col], S.vcvt(value, T.bfloat16), dist='PK_B32')
                    result = S.vcvt(S.vmul(value, inv), out_config.dtype)
                    S.vsts(quantized[row, col], result, dist='PK4_B32')

    @T.macro
    def store_tile_scales(scales, out_sf, mega_sf, mega_dma, batch_id, token_start):
        if not transpose_sf:
            T.copy(scales[:, :num_groups], out_sf[batch_id, token_start : token_start + rows_per_tile, :num_groups])
        if with_mega_sf:
            for row in T.serial(rows_per_tile):
                token = token_start + row
                if token < num_tokens:
                    i = token % mega_moe_block_m
                    mega_row = (
                        token // mega_moe_block_m * align(mega_moe_block_m, mega_sf_tile_rows)
                        + i // mega_sf_tile_rows * mega_sf_tile_rows
                        + i % mega_sf_lane_rows * mega_sf_cols
                        + i % mega_sf_tile_rows // mega_sf_lane_rows
                    )
                    T.copy(mega_dma[row, : num_groups // T.uint32.bytes, :1], mega_sf[:, mega_row : mega_row + 1])

    @T.prim_func
    def norm_forward_kernel(
        x: T.StridedTensor[(batch, num_tokens, hidden), (x_batch_stride, x_stride, 1), x_dtype],
        out: T.StridedTensor[(batch, num_tokens, hidden), (out_batch_stride, out_stride, 1), out_config.dtype],
        out_sf: T.StridedTensor[(batch, *sf_shape), (out_sf_batch_stride, out_sf_stride, 1), out_config.sf_dtype],
        out_bf16: T.StridedTensor[(batch, num_tokens, hidden), (out_bf16_batch_stride, out_bf16_stride, 1), T.bfloat16],
        mega_moe_sf: T.StridedTensor[(mega_moe_sf_rows, hidden // 128, 4), (4, mega_moe_sf_rows * 4, 1), T.uint8],
        weight: T.Tensor[(hidden,), x_dtype],
        rstd: T.Tensor[(batch, num_tokens), T.float32],
        residual: T.StridedTensor[(batch, num_tokens, hidden), (residual_batch_stride, residual_stride, 1), x_dtype],
        residual_out: T.StridedTensor[(batch, num_tokens, hidden), (residual_out_batch_stride, residual_out_stride, 1), x_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((rows_per_tile * padded_hidden,), x_dtype)
            out_ub = x_ub if reuse_input else T.alloc_shared((rows_per_tile * padded_hidden,), x_dtype)
            weight_ub = T.alloc_shared((padded_hidden,), x_dtype)
            residual_ub = T.alloc_shared((rows_per_tile * padded_hidden,), x_dtype)
            out_bf16_ub = out_ub if x_dtype == T.bfloat16 else T.alloc_shared((rows_per_tile * padded_hidden,), T.bfloat16)
            rstd_ub = T.alloc_shared((vec_size,), T.float32)
            rstd_out_ub = T.alloc_shared((rstd_out_size,), T.float32)
            quantized_ub = T.alloc_shared((rows_per_tile, padded_hidden), out_config.dtype)
            sf_scratch_ub = T.alloc_shared((rows_per_tile, sf_size), T.float32)
            scales_ub = T.alloc_shared((sf_rows, sf_size), out_config.sf_dtype)
            mega_packed_ub = T.alloc_shared((rows_per_tile, sf_size), T.uint8)
            mega_dma_ub = T.alloc_shared((rows_per_tile, sf_size // T.uint32.bytes, mega_dma_stride), T.uint32)
            mega_sf_words = T.Tensor((hidden // 128, mega_moe_sf_rows), T.uint32, mega_moe_sf.data)
            if transpose_sf:
                transposed_sf_ub = T.alloc_shared((sf_size // sf_pack, sf_rows), sf_word_dtype)
                transpose_indices_ub = T.alloc_shared((sf_lanes,), sf_word_dtype)
                scale_words = T.Tensor((sf_rows, sf_size // sf_pack), sf_word_dtype, scales_ub.data)
                out_sf_words = T.StridedTensor(
                    (batch, num_sf_words, num_tokens),
                    (out_sf_batch_stride // sf_pack, out_sf_stride // sf_pack, 1),
                    sf_word_dtype,
                    data=out_sf.data,
                )
            versions = {
                x_ub: num_stages,
                out_ub: num_stages,
                residual_ub: num_stages if with_residual else 1,
                quantized_ub: num_stages if out_config.with_sf else 1,
                mega_dma_ub: num_stages if with_mega_sf else 1,
                scales_ub: num_stages if out_config.with_sf and not transpose_sf else 1,
                rstd_ub: 1,
                sf_scratch_ub: 1,
                mega_packed_ub: 1,
            }
            if with_out_bf16:
                versions[out_bf16_ub] = num_stages
            T.annotate_buffer_versions(versions)
            if transpose_sf:
                with T.SimdVF():
                    lane = T.reinterpret(S.vci(0, f'int{sf_word_dtype.bits}'), f'{sf_word_dtype}x{sf_lanes}')
                    row = S.vand(lane, S.vdup(sf_rows - 1, sf_word_dtype))
                    pack = S.vshrs(lane, sf_rows.bit_length() - 1)
                    S.vsts(transpose_indices_ub[0], S.vadd(S.vmuls(row, sf_size // sf_pack), pack), dist=sf_store_dist)
            if with_weight:
                T.copy(weight, weight_ub[:hidden])
            num_token_blocks = T.ceildiv(num_tokens, rows_per_block)
            # Full last dimension keeps the traversal row-major (batch outer, token inner).
            for batch_id, pid_token in T.Persistent(
                [batch, num_token_blocks], num_vec_cores, core_id, group_size=num_token_blocks, num_stages=num_stages
            ):
                token_base = pid_token * rows_per_block
                for tile_index in T.serial(rows_per_block // rows_per_tile):
                    row_base = tile_index * rows_per_tile
                    token_start = token_base + row_base
                    if token_start < num_tokens:
                        T.copy(
                            x[batch_id, token_start : token_start + rows_per_tile, :hidden],
                            T.reshape(x_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                            l2_cache_ctrl='NOTALLOC_KEEP',
                        )
                        if with_residual:
                            T.copy(
                                residual[batch_id, token_start : token_start + rows_per_tile, :hidden],
                                T.reshape(residual_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                                l2_cache_ctrl='NOTALLOC_KEEP',
                            )
                        with T.SimdVF():
                            if hidden != padded_hidden:
                                clear_padding(x_ub)
                                if with_residual:
                                    clear_padding(residual_ub)
                                S.mem_bar('VST_VLD')
                            for row in T.serial(T.uint16(rows_per_tile)):
                                square_sum = reduce_row(x_ub, residual_ub, row * padded_hidden)
                                S.vsts(rstd_ub[row], square_sum, dist='ONEPT_B32')
                            S.mem_bar('VST_VLD')
                            rstd_values = compute_rstd(S.vld(rstd_ub[0]))
                            S.vsts(rstd_ub[0], rstd_values)
                            if rows_per_tile * T.float32.bytes >= 32:
                                mask = S.pset(T.float32.bits, f'PAT_VL{rows_per_tile}')
                                S.vsts(rstd_out_ub[row_base], rstd_values, mask, extent=rows_per_tile)
                            else:
                                # Vector stores require 32-byte alignment; short groups use scalar stores.
                                for row in T.unroll(rows_per_tile, explicit=True):
                                    value = S.vselr(rstd_values, S.vdup(row, T.uint32))
                                    S.vsts(rstd_out_ub[row_base + row], value, dist='ONEPT_B32')
                            S.mem_bar('VST_VLD')
                            apply_tile(residual_ub if with_residual else x_ub, weight_ub, out_ub, rstd_ub)
                        if out_config.with_sf:
                            quantize_tile(
                                out_ub,
                                quantized_ub,
                                out_bf16_ub,
                                sf_scratch_ub,
                                scales_ub,
                                row_base if transpose_sf else 0,
                                mega_packed_ub,
                                mega_dma_ub,
                            )
                        T.copy(
                            T.reshape(quantized_ub if out_config.with_sf else out_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                            out[batch_id, token_start : token_start + rows_per_tile, :hidden],
                        )
                        if out_config.with_sf:
                            store_tile_scales(scales_ub, out_sf, mega_sf_words, mega_dma_ub, batch_id, token_start)
                        if with_out_bf16:
                            T.copy(
                                T.reshape(out_bf16_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                                out_bf16[batch_id, token_start : token_start + rows_per_tile, :hidden],
                            )
                        if with_residual:
                            T.copy(
                                T.reshape(residual_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                                residual_out[batch_id, token_start : token_start + rows_per_tile, :hidden],
                            )
                valid_rows = T.min(rows_per_block, num_tokens - token_base)
                if transpose_sf:
                    # Gather scale words so each column becomes a contiguous DMA.
                    with T.SimdVF():
                        indices = S.vld(transpose_indices_ub[0])
                        for vector in T.unroll(T.ceildiv(num_sf_words * sf_rows, sf_lanes), explicit=True):
                            pack = vector * (sf_lanes // sf_rows)
                            values = S.vgather2(scale_words[0, pack], indices)
                            S.vsts(transposed_sf_ub[pack, 0], values, dist=sf_store_dist)
                    T.copy(
                        transposed_sf_ub[:num_sf_words, :valid_rows],
                        out_sf_words[batch_id, :, token_base : token_base + valid_rows],
                    )
                T.copy(rstd_out_ub[:valid_rows], rstd[batch_id, token_base])

    return norm_forward_kernel
