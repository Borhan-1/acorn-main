#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <faiss/impl/codec/vbyte_codec.h>

namespace faiss {
namespace codec {

/**
 * Base-Separated Elias-Fano Codec for Sorted Integer Adjacency Lists.
 *
 * Implements standard Elias-Fano coding (Elias 1974; Ottaviano & Venturini,
 * SIGIR 2014) over shifted non-decreasing neighbor IDs:
 *   - Base z_0 = sorted_ids[0] is stored via Base-128 VByte so the Elias-Fano
 *     universe shrinks from [0, z_{m-1}] to [0, U] where U = z_{m-1} - z_0.
 *   - For the remaining n = count - 1 shifted values x_i = z_{i+1} - z_0:
 *       * Optimal lower bit-width: l = (U >= n) ? floor(log2(U / n)) : 0
 *       * 1-byte header storing l (uint8_t)
 *       * Lower bits: n * l bits packed tightly in little-endian bit order
 *       * Upper bits: unary-encoded high parts h_i = x_i >> l with bit (h_i + i)
 *         set in a bitvector of length (h_{n-1} + n) bits.
 *
 * Decoder uses 64-bit word-at-a-time ctzll (tzcnt) scanning and unaligned
 * 64-bit lower-bit extraction directly into a caller-provided stack buffer.
 * Callers must append PADDING_BYTES zero bytes once at the end of the global
 * compressed buffer to ensure safe unaligned 64-bit reads.
 */
struct EliasFanoCodec {
    static constexpr size_t PADDING_BYTES = 16;

    static inline uint8_t floor_log2_u32(uint32_t v) {
        if (v == 0) {
            return 0;
        }
#if defined(__GNUC__) || defined(__clang__)
        return static_cast<uint8_t>(31 - __builtin_clz(v));
#else
        uint8_t r = 0;
        while (v >>= 1) {
            ++r;
        }
        return r;
#endif
    }

    static inline uint32_t ctz_u64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
        return static_cast<uint32_t>(__builtin_ctzll(v));
#else
        uint32_t c = 0;
        while ((v & 1ULL) == 0ULL) {
            v >>= 1;
            ++c;
        }
        return c;
#endif
    }

    /**
     * Encode a sorted non-decreasing list of non-negative int32_t IDs.
     */
    static inline void encode_delta_list(
            const int32_t* sorted_ids,
            size_t count,
            std::vector<uint8_t>& out) {
        if (count == 0) {
            return;
        }

        // 1. Base-separated VByte header for z_0
        const uint32_t base = static_cast<uint32_t>(sorted_ids[0]);
        uint8_t base_buf[5];
        const size_t base_len = VByteCodec::encode_uint32(base, base_buf);
        out.insert(out.end(), base_buf, base_buf + base_len);

        if (count == 1) {
            return;
        }

        const size_t n = count - 1;
        const uint32_t universe = static_cast<uint32_t>(sorted_ids[count - 1]) - base;

        // 2. Optimal lower bit-width l = max(0, floor(log2(U / n)))
        uint8_t l = 0;
        if (universe >= n) {
            l = floor_log2_u32(static_cast<uint32_t>(universe / n));
            if (l > 30) {
                l = 30;
            }
        }
        out.push_back(l);

        // 3. Lower bits: tightly packed n * l bits
        const size_t lower_bytes = (n * static_cast<size_t>(l) + 7) >> 3;
        const uint32_t h_max = (l == 0) ? universe : (universe >> l);
        const size_t max_upper_pos = static_cast<size_t>(h_max) + (n - 1);
        const size_t upper_bytes = (max_upper_pos >> 3) + 1;

        const size_t start_off = out.size();
        out.resize(start_off + lower_bytes + upper_bytes, 0);

        uint8_t* lower_ptr = out.data() + start_off;
        uint8_t* upper_ptr = lower_ptr + lower_bytes;

        const uint64_t low_mask = (l == 0) ? 0ULL : ((1ULL << l) - 1ULL);

        // Write lower bits and set upper unary bits
        if (l > 0) {
            size_t bit_off = 0;
            for (size_t i = 0; i < n; ++i) {
                const uint32_t x = static_cast<uint32_t>(sorted_ids[i + 1]) - base;
                uint64_t low = static_cast<uint64_t>(x) & low_mask;

                // Write l bits starting at bit_off in little-endian byte order
                size_t byte_idx = bit_off >> 3;
                uint32_t bit_in_byte = static_cast<uint32_t>(bit_off & 7);
                uint64_t shifted = low << bit_in_byte;
                uint32_t total_bits = bit_in_byte + l;
                while (total_bits > 0) {
                    lower_ptr[byte_idx++] |= static_cast<uint8_t>(shifted & 0xFFU);
                    shifted >>= 8;
                    total_bits = (total_bits > 8) ? (total_bits - 8) : 0;
                }
                bit_off += l;

                const size_t up_pos = static_cast<size_t>(x >> l) + i;
                upper_ptr[up_pos >> 3] |= static_cast<uint8_t>(1U << (up_pos & 7));
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                const uint32_t x = static_cast<uint32_t>(sorted_ids[i + 1]) - base;
                const size_t up_pos = static_cast<size_t>(x) + i;
                upper_ptr[up_pos >> 3] |= static_cast<uint8_t>(1U << (up_pos & 7));
            }
        }
    }

    /**
     * Decode an Elias-Fano encoded list into out[0..count-1].
     * Returns pointer to the exact byte immediately after the encoded segment.
     */
    static inline const uint8_t* decode_delta_list(
            const uint8_t* in,
            size_t count,
            int32_t* out) {
        if (count == 0) {
            return in;
        }

        const uint8_t* ptr = in;
        const uint32_t base = VByteCodec::decode_uint32(ptr);
        out[0] = static_cast<int32_t>(base);

        if (count == 1) {
            return ptr;
        }

        const size_t n = count - 1;
        const uint8_t l = *ptr++;
        const size_t lower_bytes = (n * static_cast<size_t>(l) + 7) >> 3;
        const uint8_t* lower_ptr = ptr;
        const uint8_t* upper_ptr = lower_ptr + lower_bytes;

        const uint64_t low_mask = (l == 0) ? 0ULL : ((1ULL << l) - 1ULL);

        size_t i = 0;
        size_t word_idx = 0;
        size_t last_bit_pos = 0;

        if (l > 0) {
            size_t bit_off = 0;
            while (i < n) {
                uint64_t word = 0;
                std::memcpy(&word, upper_ptr + (word_idx << 3), sizeof(uint64_t));
                const size_t word_base_bit = word_idx << 6;
                while (word != 0ULL && i < n) {
                    const uint32_t tz = ctz_u64(word);
                    word &= (word - 1ULL);
                    const size_t bit_pos = word_base_bit + tz;
                    last_bit_pos = bit_pos;

                    const uint32_t high = static_cast<uint32_t>(bit_pos - i);

                    uint64_t lo_word = 0;
                    std::memcpy(&lo_word, lower_ptr + (bit_off >> 3), sizeof(uint64_t));
                    const uint32_t low = static_cast<uint32_t>(
                            (lo_word >> (bit_off & 7)) & low_mask);
                    bit_off += l;

                    out[i + 1] = static_cast<int32_t>(base + ((high << l) | low));
                    ++i;
                }
                ++word_idx;
            }
        } else {
            while (i < n) {
                uint64_t word = 0;
                std::memcpy(&word, upper_ptr + (word_idx << 3), sizeof(uint64_t));
                const size_t word_base_bit = word_idx << 6;
                while (word != 0ULL && i < n) {
                    const uint32_t tz = ctz_u64(word);
                    word &= (word - 1ULL);
                    const size_t bit_pos = word_base_bit + tz;
                    last_bit_pos = bit_pos;

                    const uint32_t high = static_cast<uint32_t>(bit_pos - i);
                    out[i + 1] = static_cast<int32_t>(base + high);
                    ++i;
                }
                ++word_idx;
            }
        }

        return upper_ptr + (last_bit_pos >> 3) + 1;
    }
};

} // namespace codec
} // namespace faiss
