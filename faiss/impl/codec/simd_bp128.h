#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include <faiss/impl/codec/vbyte_codec.h>

namespace faiss {
namespace codec {

/**
 * Graph-Aware Base-Separated AVX2 SIMD Bit-Packing Codec (SIMD-BP128 / Mini-Block).
 *
 * Format for a sorted ascending list of `count` non-negative int32_t IDs:
 *   - If count == 0: 0 bytes.
 *   - Base ID `v_0`: VByte-encoded uint32_t (1-5 bytes).
 *   - Remaining `rem = count - 1` consecutive deltas (d_i = v_i - v_{i-1}):
 *     - Processed in 16-delta mini-blocks (2 x 8-wide AVX2 lanes):
 *         [1-byte header b] + [2 * b bytes of exact bit-packed deltas]
 *       where b in {0..24, 32}. (Bit-widths 25..31 are rounded up to 32.)
 *     - If remaining tail >= 8:
 *         [1-byte header b] + [b bytes of 8 bit-packed deltas]
 *     - If remaining tail (1..7):
 *         [1-byte header b] + [ceil(tail * b / 8) bytes of bit-packed deltas]
 *
 * Safety requirement for AVX2 decoding:
 *   The backing compressed byte buffer should have at least 16 bytes of trailing
 *   zero-padding so 128-bit unaligned SIMD loads near the end never read out of bounds.
 */
struct SimdBP128Codec {
    static constexpr size_t PADDING_BYTES = 16;

    static inline uint8_t bits_required(uint32_t max_val) {
        if (max_val == 0) {
            return 0;
        }
#if defined(__GNUC__) || defined(__clang__)
        uint8_t b = static_cast<uint8_t>(32 - __builtin_clz(max_val));
#else
        uint8_t b = 0;
        while (max_val > 0) {
            max_val >>= 1;
            ++b;
        }
#endif
        return (b > 24) ? 32 : b;
    }

    /// Pack `n` (1..8) deltas of exact width `b` (0..24 or 32) into `out` (little-endian bitstream).
    static inline void pack_up_to_8(
            const uint32_t* deltas,
            size_t n,
            uint8_t b,
            std::vector<uint8_t>& out) {
        if (b == 0 || n == 0) {
            return;
        }
        if (b == 32) {
            size_t old_sz = out.size();
            out.resize(old_sz + n * 4);
            std::memcpy(out.data() + old_sz, deltas, n * 4);
            return;
        }
        const size_t nbytes = (n * b + 7) / 8;
        const size_t old_sz = out.size();
        out.resize(old_sz + nbytes, 0);
        uint8_t* dst = out.data() + old_sz;

        for (size_t i = 0; i < n; ++i) {
            uint64_t val = static_cast<uint64_t>(deltas[i]);
            size_t bit_pos = i * b;
            size_t byte_pos = bit_pos >> 3;
            size_t bit_off = bit_pos & 7;
            uint64_t shifted = val << bit_off;
            size_t span = (bit_off + b + 7) >> 3;
            for (size_t k = 0; k < span; ++k) {
                dst[byte_pos + k] |= static_cast<uint8_t>((shifted >> (k * 8)) & 0xFFu);
            }
        }
    }

    /// Scalar unpack of `n` (1..8) deltas of exact width `b` + prefix-sum into `out`.
    static inline const uint8_t* unpack_up_to_8_scalar(
            const uint8_t* ptr,
            size_t n,
            uint8_t b,
            uint32_t& prev,
            int32_t* out) {
        if (b == 0) {
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<int32_t>(prev);
            }
            return ptr;
        }
        if (b == 32) {
            for (size_t i = 0; i < n; ++i) {
                uint32_t d = 0;
                std::memcpy(&d, ptr + i * 4, 4);
                prev += d;
                out[i] = static_cast<int32_t>(prev);
            }
            return ptr + n * 4;
        }
        const uint32_t mask = (1u << b) - 1u;
        const size_t nbytes = (n * b + 7) / 8;
        for (size_t i = 0; i < n; ++i) {
            size_t bit_pos = i * b;
            size_t byte_pos = bit_pos >> 3;
            size_t bit_off = bit_pos & 7;
            uint64_t word = 0;
            size_t avail = std::min<size_t>(4, nbytes - byte_pos);
            for (size_t k = 0; k < avail; ++k) {
                word |= static_cast<uint64_t>(ptr[byte_pos + k]) << (k * 8);
            }
            uint32_t d = static_cast<uint32_t>((word >> bit_off) & mask);
            prev += d;
            out[i] = static_cast<int32_t>(prev);
        }
        return ptr + nbytes;
    }

#ifdef __AVX2__
    struct Avx2Lut {
        alignas(32) int8_t shuf[25][32];
        alignas(32) uint32_t shift[25][8];
        alignas(32) uint32_t mask[25][8];
        uint8_t hi_byte_off[25];

        Avx2Lut() {
            std::memset(this, 0, sizeof(*this));
            for (int b = 1; b <= 24; ++b) {
                uint32_t m = (1u << b) - 1u;
                int hi_base = (4 * b) / 8;
                hi_byte_off[b] = static_cast<uint8_t>(hi_base);

                for (int lane = 0; lane < 8; ++lane) {
                    int bit_pos = lane * b;
                    int rel_bit = (lane < 4) ? bit_pos : (bit_pos - hi_base * 8);
                    int byte_idx = rel_bit / 8;
                    int bit_sh = rel_bit % 8;

                    shift[b][lane] = static_cast<uint32_t>(bit_sh);
                    mask[b][lane] = m;

                    int base_shuf = lane * 4;
                    for (int k = 0; k < 4; ++k) {
                        int idx = byte_idx + k;
                        shuf[b][base_shuf + k] =
                                (idx < 16) ? static_cast<int8_t>(idx) : static_cast<int8_t>(-1);
                    }
                }
            }
        }
    };

    static inline const Avx2Lut& lut() {
        static const Avx2Lut instance;
        return instance;
    }

    /// AVX2 in-register prefix-sum of 8 deltas + running `prev_vec` (all lanes = prev).
    static inline __m256i avx2_prefix_sum(__m256i deltas, __m256i& prev_vec) {
        // Step 1: Intra-128-bit-lane prefix sum (4 lanes in lo128 and 4 lanes in hi128)
        __m256i s1 = _mm256_add_epi32(deltas, _mm256_slli_si256(deltas, 4));
        __m256i s2 = _mm256_add_epi32(s1, _mm256_slli_si256(s1, 8));

        // Step 2: Broadcast lane 3 (top of lo128) into all lanes of hi128
        __m128i lo = _mm256_castsi256_si128(s2);
        __m128i lo_sum = _mm_shuffle_epi32(lo, _MM_SHUFFLE(3, 3, 3, 3));
        __m256i cross = _mm256_inserti128_si256(_mm256_setzero_si256(), lo_sum, 1);
        __m256i full = _mm256_add_epi32(s2, cross);

        // Step 3: Add running base `prev_vec`
        __m256i res = _mm256_add_epi32(full, prev_vec);

        // Step 4: Broadcast lane 7 of `res` into all lanes of `prev_vec` for next chunk
        __m128i hi_res = _mm256_extracti128_si256(res, 1);
        __m128i last = _mm_shuffle_epi32(hi_res, _MM_SHUFFLE(3, 3, 3, 3));
        prev_vec = _mm256_broadcastsi128_si256(last);
        return res;
    }

    /// Decode 8 bit-packed deltas using AVX2 (b in 1..24) with preloaded LUT vectors.
    static inline const uint8_t* unpack_8_avx2(
            const uint8_t* ptr,
            uint8_t b,
            __m256i shuf_v,
            __m256i shift_v,
            __m256i mask_v,
            uint8_t hi_off,
            __m256i& prev_vec,
            int32_t* out) {
        __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
        __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + hi_off));
        __m256i raw = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        __m256i aligned = _mm256_shuffle_epi8(raw, shuf_v);
        __m256i shifted = _mm256_srlv_epi32(aligned, shift_v);
        __m256i deltas = _mm256_and_si256(shifted, mask_v);
        __m256i ids = avx2_prefix_sum(deltas, prev_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), ids);
        return ptr + b;
    }
#endif

    /**
     * Compress a sorted ascending list of non-negative neighbor IDs using
     * Base-Separated AVX2 Mini-Block Bit-Packing.
     */
    static inline void encode_delta_list(
            const int32_t* sorted_ids,
            size_t count,
            std::vector<uint8_t>& out) {
        if (count == 0) {
            return;
        }

        // 1. Encode base ID v_0 with VByte (1-5 bytes)
        uint8_t v0_buf[5];
        uint32_t prev = static_cast<uint32_t>(sorted_ids[0]);
        size_t v0_len = VByteCodec::encode_uint32(prev, v0_buf);
        out.insert(out.end(), v0_buf, v0_buf + v0_len);

        if (count == 1) {
            return;
        }

        // Compute consecutive deltas into stack buffer
        const size_t ndeltas = count - 1;
        uint32_t delta_stack[1024];
        std::vector<uint32_t> delta_heap;
        uint32_t* deltas = delta_stack;
        if (ndeltas > 1024) {
            delta_heap.resize(ndeltas);
            deltas = delta_heap.data();
        }

        for (size_t i = 0; i < ndeltas; ++i) {
            uint32_t curr = static_cast<uint32_t>(sorted_ids[i + 1]);
            deltas[i] = curr - prev;
            prev = curr;
        }

        size_t pos = 0;

        // 2. Encode 16-delta mini-blocks (2 x 8-wide AVX2 chunks sharing 1-byte header)
        while (pos + 16 <= ndeltas) {
            uint32_t max_d = 0;
            for (size_t i = 0; i < 16; ++i) {
                max_d |= deltas[pos + i];
            }
            uint8_t b = bits_required(max_d);
            out.push_back(b);
            pack_up_to_8(deltas + pos, 8, b, out);
            pack_up_to_8(deltas + pos + 8, 8, b, out);
            pos += 16;
        }

        // 3. Optional 8-delta block
        if (pos + 8 <= ndeltas) {
            uint32_t max_d = 0;
            for (size_t i = 0; i < 8; ++i) {
                max_d |= deltas[pos + i];
            }
            uint8_t b = bits_required(max_d);
            out.push_back(b);
            pack_up_to_8(deltas + pos, 8, b, out);
            pos += 8;
        }

        // 4. Compact scalar tail (1..7 deltas)
        if (pos < ndeltas) {
            size_t tail = ndeltas - pos;
            uint32_t max_d = 0;
            for (size_t i = 0; i < tail; ++i) {
                max_d |= deltas[pos + i];
            }
            uint8_t b = bits_required(max_d);
            out.push_back(b);
            pack_up_to_8(deltas + pos, tail, b, out);
        }
    }

    /**
     * Decompress a SIMD-BP128 encoded list directly into caller's stack buffer `out`.
     * Returns pointer to the byte immediately following the decoded segment.
     */
    static inline const uint8_t* decode_delta_list(
            const uint8_t* in,
            size_t count,
            int32_t* out) {
        if (count == 0) {
            return in;
        }

        const uint8_t* ptr = in;
        uint32_t prev = VByteCodec::decode_uint32(ptr);
        out[0] = static_cast<int32_t>(prev);
        if (count == 1) {
            return ptr;
        }

        const size_t ndeltas = count - 1;
        size_t pos = 0;

#ifdef __AVX2__
        const Avx2Lut& L = lut();
        __m256i prev_vec = _mm256_set1_epi32(static_cast<int32_t>(prev));

        // 16-delta AVX2 mini-blocks
        while (pos + 16 <= ndeltas) {
            uint8_t b = *ptr++;
            if (b == 0) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 1 + pos), prev_vec);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 1 + pos + 8), prev_vec);
            } else if (b == 32) {
                __m256i d0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
                __m256i d1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr + 32));
                ptr += 64;
                _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(out + 1 + pos), avx2_prefix_sum(d0, prev_vec));
                _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(out + 1 + pos + 8),
                        avx2_prefix_sum(d1, prev_vec));
            } else {
                __m256i shuf_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(L.shuf[b]));
                __m256i shift_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(L.shift[b]));
                __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(L.mask[b]));
                uint8_t hi_off = L.hi_byte_off[b];
                ptr = unpack_8_avx2(
                        ptr, b, shuf_v, shift_v, mask_v, hi_off, prev_vec, out + 1 + pos);
                ptr = unpack_8_avx2(
                        ptr, b, shuf_v, shift_v, mask_v, hi_off, prev_vec, out + 1 + pos + 8);
            }
            pos += 16;
        }

        // Optional 8-delta AVX2 chunk
        if (pos + 8 <= ndeltas) {
            uint8_t b = *ptr++;
            if (b == 0) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 1 + pos), prev_vec);
            } else if (b == 32) {
                __m256i d0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
                ptr += 32;
                _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(out + 1 + pos), avx2_prefix_sum(d0, prev_vec));
            } else {
                __m256i shuf_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(L.shuf[b]));
                __m256i shift_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(L.shift[b]));
                __m256i mask_v = _mm256_load_si256(reinterpret_cast<const __m256i*>(L.mask[b]));
                uint8_t hi_off = L.hi_byte_off[b];
                ptr = unpack_8_avx2(
                        ptr, b, shuf_v, shift_v, mask_v, hi_off, prev_vec, out + 1 + pos);
            }
            pos += 8;
        }

        // Sync scalar `prev` from `out[pos]` before optional tail
        prev = static_cast<uint32_t>(out[pos]);
#else
        while (pos + 16 <= ndeltas) {
            uint8_t b = *ptr++;
            ptr = unpack_up_to_8_scalar(ptr, 8, b, prev, out + 1 + pos);
            ptr = unpack_up_to_8_scalar(ptr, 8, b, prev, out + 1 + pos + 8);
            pos += 16;
        }
        if (pos + 8 <= ndeltas) {
            uint8_t b = *ptr++;
            ptr = unpack_up_to_8_scalar(ptr, 8, b, prev, out + 1 + pos);
            pos += 8;
        }
#endif

        // Compact scalar tail (1..7 deltas)
        if (pos < ndeltas) {
            uint8_t b = *ptr++;
            ptr = unpack_up_to_8_scalar(ptr, ndeltas - pos, b, prev, out + 1 + pos);
        }

        return ptr;
    }
};

} // namespace codec
} // namespace faiss
