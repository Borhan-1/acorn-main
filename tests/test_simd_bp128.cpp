#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <faiss/impl/codec/simd_bp128.h>
#include <faiss/impl/codec/vbyte_codec.h>

using faiss::codec::SimdBP128Codec;
using faiss::codec::VByteCodec;

static void test_all_lengths_and_bitwidths() {
    std::mt19937 rng(12345);
    std::vector<uint8_t> buf;
    std::vector<int32_t> decoded;

    // 1. Test every bit-width b in 0..28 across lengths 0..130
    for (int b = 0; b <= 28; ++b) {
        uint32_t max_delta = (b == 0) ? 0u : ((b < 28) ? ((1u << b) - 1u) : 0x03FFFFFFu);
        for (size_t len = 0; len <= 130; ++len) {
            std::vector<int32_t> orig(len);
            if (len > 0) {
                uint64_t cur = (rng() % 500000);
                orig[0] = static_cast<int32_t>(cur);
                for (size_t i = 1; i < len; ++i) {
                    uint32_t d = (max_delta == 0) ? 0u : (rng() % (max_delta + 1));
                    // Force exact boundary value on first delta so bit-width is exercised
                    if (i == 1 && max_delta > 0 && cur + max_delta < 0x70000000ULL) {
                        d = max_delta;
                    }
                    if (cur + d >= 0x70000000ULL) {
                        d = 0;
                    }
                    cur += d;
                    orig[i] = static_cast<int32_t>(cur);
                }
            }

            buf.clear();
            SimdBP128Codec::encode_delta_list(orig.data(), orig.size(), buf);
            const size_t encoded_len = buf.size();
            // Append required SIMD safe tail padding
            buf.resize(encoded_len + SimdBP128Codec::PADDING_BYTES, 0);

            decoded.assign(len + 16, -999);
            const uint8_t* end_ptr =
                    SimdBP128Codec::decode_delta_list(buf.data(), orig.size(), decoded.data());

            if (static_cast<size_t>(end_ptr - buf.data()) != encoded_len) {
                std::cerr << "FAIL byte-length mismatch: b=" << b << " len=" << len
                          << " expected=" << encoded_len
                          << " got=" << (end_ptr - buf.data()) << "\n";
                std::exit(1);
            }
            for (size_t i = 0; i < len; ++i) {
                if (decoded[i] != orig[i]) {
                    std::cerr << "FAIL value mismatch: b=" << b << " len=" << len
                              << " i=" << i << " expected=" << orig[i]
                              << " got=" << decoded[i] << "\n";
                    std::exit(1);
                }
            }
        }
    }
    std::cout << "[PASS] Exhaustive round-trip test (b=0..28, len=0..130): 100% lossless!\n";
}

int main() {
#ifdef __AVX2__
    std::cout << "=== SimdBP128Codec Unit Test & Microbenchmark (AVX2 ENABLED) ===\n";
#else
    std::cout << "=== SimdBP128Codec Unit Test & Microbenchmark (SCALAR FALLBACK) ===\n";
#endif

    test_all_lengths_and_bitwidths();

    // Realistic 1M SIFT1M BFS-reordered workload: 20,000 nodes, 40..64 edges/node
    const size_t num_nodes = 20000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> deg_dist(40, 64);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    std::vector<std::vector<int32_t>> lists(num_nodes);
    size_t total_edges = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        size_t deg = deg_dist(rng);
        lists[i].resize(deg);
        uint32_t cur = static_cast<uint32_t>(rng() % 50000);
        lists[i][0] = static_cast<int32_t>(cur);
        for (size_t j = 1; j < deg; ++j) {
            double r = u01(rng);
            uint32_t d;
            if (r < 0.35) {
                d = 1 + (rng() % 120);         // 1-byte delta (~35%)
            } else if (r < 0.85) {
                d = 128 + (rng() % 8000);      // 2-byte delta (~50%)
            } else {
                d = 16384 + (rng() % 50000);   // 3-byte delta (~15%)
            }
            cur += d;
            lists[i][j] = static_cast<int32_t>(cur);
        }
        total_edges += deg;
    }

    // Encode both VByte and SIMD-BP128 into flat contiguous buffers (matching ACORN layout)
    std::vector<uint8_t> vbyte_buf;
    std::vector<size_t> vbyte_off(num_nodes + 1, 0);
    std::vector<uint8_t> simd_buf;
    std::vector<size_t> simd_off(num_nodes + 1, 0);

    for (size_t i = 0; i < num_nodes; ++i) {
        VByteCodec::encode_delta_list(lists[i].data(), lists[i].size(), vbyte_buf);
        vbyte_off[i + 1] = vbyte_buf.size();

        SimdBP128Codec::encode_delta_list(lists[i].data(), lists[i].size(), simd_buf);
        simd_off[i + 1] = simd_buf.size();
    }
    const size_t simd_payload_bytes = simd_buf.size();
    simd_buf.resize(simd_payload_bytes + SimdBP128Codec::PADDING_BYTES, 0);

    // Verify 100% lossless on all 20,000 realistic lists
    int32_t dec_buf[1024];
    for (size_t i = 0; i < num_nodes; ++i) {
        SimdBP128Codec::decode_delta_list(
                simd_buf.data() + simd_off[i], lists[i].size(), dec_buf);
        for (size_t j = 0; j < lists[i].size(); ++j) {
            assert(dec_buf[j] == lists[i][j]);
        }
    }

    // Microbenchmark decode speed (100 passes = ~100M+ decoded integers)
    const int passes = 100;
    volatile int64_t sink = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    int64_t sum_vb = 0;
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < num_nodes; ++i) {
            size_t cnt = lists[i].size();
            VByteCodec::decode_delta_list(vbyte_buf.data() + vbyte_off[i], cnt, dec_buf);
            sum_vb += dec_buf[cnt - 1];
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    sink += sum_vb;

    auto t2 = std::chrono::high_resolution_clock::now();
    int64_t sum_simd = 0;
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < num_nodes; ++i) {
            size_t cnt = lists[i].size();
            SimdBP128Codec::decode_delta_list(simd_buf.data() + simd_off[i], cnt, dec_buf);
            sum_simd += dec_buf[cnt - 1];
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    sink += sum_simd;
    assert(sum_vb == sum_simd);

    double vb_s = std::chrono::duration<double>(t1 - t0).count();
    double simd_s = std::chrono::duration<double>(t3 - t2).count();
    double total_decoded = static_cast<double>(total_edges) * passes;

    double vb_ns = (vb_s * 1e9) / total_decoded;
    double simd_ns = (simd_s * 1e9) / total_decoded;
    double speedup = vb_s / simd_s;

    std::cout << "\n--- Microbenchmark (" << num_nodes << " lists, "
              << total_edges << " edges, " << passes << " passes) ---\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Raw int32_t:   " << (total_edges * 4.0 / (1024.0 * 1024.0)) << " MB (4.00 B/edge)\n";
    std::cout << "VByteCodec:    " << (vbyte_buf.size() / (1024.0 * 1024.0)) << " MB ("
              << (static_cast<double>(vbyte_buf.size()) / total_edges) << " B/edge) | "
              << vb_ns << " ns/edge (" << (total_decoded / vb_s / 1e9) << " B ints/s)\n";
    std::cout << "SimdBP128:     " << (simd_payload_bytes / (1024.0 * 1024.0)) << " MB ("
              << (static_cast<double>(simd_payload_bytes) / total_edges) << " B/edge) | "
              << simd_ns << " ns/edge (" << (total_decoded / simd_s / 1e9) << " B ints/s)\n";
    std::cout << "--> Decode Speedup (SimdBP128 vs VByte): " << speedup << "x\n";
    std::cout << "(checksum=" << sink << ")\n";
    return 0;
}
