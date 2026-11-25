#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>

namespace faiss {
namespace codec {

/**
 * Standard Variable-Byte (VByte / Varint) Encoder & Decoder.
 * Uses 7 bits for data and the 8th bit (MSB) as a continuation flag.
 * - Value < 128: 1 byte
 * - Value < 16,384: 2 bytes
 * - Value < 2,097,152: 3 bytes
 * - Value < 268,435,456: 4 bytes
 */
struct VByteCodec {
    /// Encode a single uint32_t into a byte buffer. Returns number of bytes written (1-5).
    static inline size_t encode_uint32(uint32_t val, uint8_t* out) {
        size_t bytes = 0;
        while (val >= 128) {
            out[bytes++] = static_cast<uint8_t>((val & 0x7F) | 0x80);
            val >>= 7;
        }
        out[bytes++] = static_cast<uint8_t>(val & 0x7F);
        return bytes;
    }

    /// Decode a single uint32_t from a byte buffer. Advances the buffer pointer.
    static inline uint32_t decode_uint32(const uint8_t*& in) {
        uint32_t val = 0;
        uint32_t shift = 0;
        while (true) {
            uint8_t byte = *in++;
            val |= static_cast<uint32_t>(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) {
                break;
            }
            shift += 7;
        }
        return val;
    }

    /**
     * Compress a sorted list of integer neighbor IDs using Delta Encoding + VByte.
     * IDs must be in ascending order: ids[0] <= ids[1] <= ... <= ids[count-1]
     */
    static inline void encode_delta_list(
            const int32_t* sorted_ids,
            size_t count,
            std::vector<uint8_t>& out) {
        if (count == 0) return;

        // Reserve approximate capacity (average ~2 bytes per delta)
        out.reserve(out.size() + count * 2);

        uint8_t temp[5];
        uint32_t prev = 0;

        for (size_t i = 0; i < count; ++i) {
            uint32_t curr = static_cast<uint32_t>(sorted_ids[i]);
            uint32_t delta = curr - prev;
            prev = curr;

            size_t len = encode_uint32(delta, temp);
            out.insert(out.end(), temp, temp + len);
        }
    }

    /**
     * Decompress a VByte delta-encoded list into a target buffer.
     * Fast inline decoding directly into stack-allocated buffer.
     */
    static inline void decode_delta_list(
            const uint8_t* in,
            size_t count,
            int32_t* out) {
        uint32_t prev = 0;
        const uint8_t* ptr = in;

        for (size_t i = 0; i < count; ++i) {
            uint32_t delta = decode_uint32(ptr);
            prev += delta;
            out[i] = static_cast<int32_t>(prev);
        }
    }
};

} // namespace codec
} // namespace faiss

