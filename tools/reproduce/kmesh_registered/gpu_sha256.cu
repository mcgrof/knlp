// SPDX-License-Identifier: MIT
// Standard SHA-256: NIST FIPS 180-4, sections 4.2.2, 5.1.1 and 6.2.2.
// Each message keeps its standard sequential compression chain. Independent
// messages run concurrently; no tree hash or payload transfer is substituted.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {
__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void compress(uint32_t state[8], uint32_t w[16]) {
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
#pragma unroll
  for (int t = 0; t < 64; ++t) {
    if (t >= 16) {
      uint32_t x = w[(t - 15) & 15], y = w[(t - 2) & 15];
      uint32_t s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
      uint32_t s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10);
      w[t & 15] += s0 + w[(t - 7) & 15] + s1;
    }
    uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = h + s1 + ch + k[t] + w[t & 15];
    uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + s0 + maj;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

__global__ void sha256(const uint8_t* input, size_t bytes, size_t stride,
                       uint8_t* output, unsigned count) {
  const unsigned message = blockIdx.x;
  if (message >= count || threadIdx.x != 0) return;
  const uint8_t* source = bytes == 0 ? input : input + message * stride;
  uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint32_t w[16];
  const size_t full = bytes / 64;
  for (size_t block = 0; block < full; ++block) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint8_t* p = source + block * 64 + i * 4;
      w[i] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
             (uint32_t(p[2]) << 8) | uint32_t(p[3]);
    }
    compress(state, w);
  }
  const size_t tail = bytes % 64;
  const size_t pad_bytes = tail < 56 ? 64 : 128;
  const uint64_t bits = uint64_t(bytes) * 8;
  for (size_t block = 0; block < pad_bytes; block += 64) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      uint32_t word = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const size_t pos = block + i * 4 + j;
        uint8_t value = 0;
        if (pos < tail)
          value = source[full * 64 + pos];
        else if (pos == tail)
          value = 0x80;
        else if (pos >= pad_bytes - 8)
          value = uint8_t(bits >> ((pad_bytes - 1 - pos) * 8));
        word = (word << 8) | value;
      }
      w[i] = word;
    }
    compress(state, w);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
      output[message * 32 + i * 4 + j] = uint8_t(state[i] >> (24 - j * 8));
  }
}
}  // namespace

// The caller owns buffers, stream ordering and completion through actual drain.
extern "C" int kmesh_sha256(const void* input, size_t bytes, size_t stride,
                            void* output, unsigned count, void* stream) {
  if ((input == nullptr && bytes != 0) || output == nullptr || count == 0 ||
      stride < bytes || bytes > (UINT64_MAX / 8))
    return int(cudaErrorInvalidValue);
  sha256<<<count, 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(input), bytes, stride,
      static_cast<uint8_t*>(output), count);
  return int(cudaGetLastError());
}
