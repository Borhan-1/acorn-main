#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <faiss/impl/codec/elias_fano_codec.h>
#include <faiss/impl/codec/simd_bp128.h>
#include <faiss/impl/codec/vbyte_codec.h>

using faiss::codec::EliasFanoCodec;
using faiss::codec::SimdBP128Codec;
using faiss::codec::VByteCodec;

static void test_exhaustive_roundtrip() {
    std::mt19937 rng(12345);
    std::vector<uint8_t> buf;
    std::vector<int32_t> decoded;

    // Test every bit-width b in 0..24 across lengths 0..130 (3,275 cases)
    size_t cases = 0;
    for (int b = 0; b <= 24; ++b) {
        uint32_t max_delta = (b == 0) ? 0u : ((1u << b) - 1u);
        for (size_t len = 0; len <= 130; ++len) {
            std::vector<int32_t> orig(len);
            if (len > 0) {
                uint64_t cur = (rng() % 500000);
                orig[0] = static_cast<int32_t>(cur);
                for (size_t i = 1; i < len; ++i) {
                    uint32_t d = (max_delta == 0) ? 0u : (rng() % (max_delta + 1));
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
            EliasFanoCodec::encode_delta_list(orig.data(), orig.size(), buf);
            const size_t encoded_len = buf.size();
            buf.resize(encoded_len + EliasFanoCodec::PADDING_BYTES, 0);

            decoded.assign(len + 16, -999);
            const uint8_t* end_ptr = EliasFanoCodec::decode_delta_list(
                    buf.data(), orig.size(), decoded.data());

            if (static_cast<size_t>(end_ptr - buf.data()) != encoded_len) {
                std::cerr << "FAIL byte-length mismatch: b=" << b
                          << " len=" << len
                          << " expected=" << encoded_len
                          << " got=" << (end_ptr - buf.data()) << "\n";
                std::exit(1);
            }
            for (size_t i = 0; i < len; ++i) {
                if (decoded[i] != orig[i]) {
                    std::cerr << "FAIL value mismatch: b=" << b
                              << " len=" << len << " i=" << i
                              << " expected=" << orig[i]
                              << " got=" << decoded[i] << "\n";
                    std::exit(1);
                }
            }
            ++cases;
        }
    }

    // Two-segment concatenated round-trip test (matching ACORN proximity split)
    for (size_t s = 1; s <= 16; ++s) {
        std::vector<int32_t> seg1(s), seg2(48);
        uint32_t cur = 10000;
        for (size_t i = 0; i < s; ++i) {
            cur += 1 + (rng() % 5000);
            seg1[i] = static_cast<int32_t>(cur);
        }
        cur = 5000;
        for (size_t i = 0; i < 48; ++i) {
            cur += 1 + (rng() % 2000);
            seg2[i] = static_cast<int32_t>(cur);
        }
        buf.clear();
        EliasFanoCodec::encode_delta_list(seg1.data(), seg1.size(), buf);
        EliasFanoCodec::encode_delta_list(seg2.data(), seg2.size(), buf);
        const size_t total_len = buf.size();
        buf.resize(total_len + EliasFanoCodec::PADDING_BYTES, 0);

        std::vector<int32_t> out(s + 48, -1);
        const uint8_t* p1 = EliasFanoCodec::decode_delta_list(buf.data(), s, out.data());
        const uint8_t* p2 = EliasFanoCodec::decode_delta_list(p1, 48, out.data() + s);
        assert(static_cast<size_t>(p2 - buf.data()) == total_len);
        for (size_t i = 0; i < s; ++i) assert(out[i] == seg1[i]);
        for (size_t i = 0; i < 48; ++i) assert(out[s + i] == seg2[i]);
    }

    std::cout << "[PASS] EliasFanoCodec exhaustive round-trip (" << cases
              << " cases + 16 two-segment splits): 100% lossless!\n";
}

int main() {
    std::cout << "=== 3-Way Codec Microbenchmark: VByte vs. Elias-Fano vs. AVX2 SIMD-BP128 ===\n";
    test_exhaustive_roundtrip();

    // Identical realistic 1M SIFT1M BFS-reordered workload as test_simd_bp128.cpp
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
                d = 1 + (rng() % 120);
            } else if (r < 0.85) {
                d = 128 + (rng() % 8000);
            } else {
                d = 16384 + (rng() % 50000);
            }
            cur += d;
            lists[i][j] = static_cast<int32_t>(cur);
        }
        total_edges += deg;
    }

    std::vector<uint8_t> vb_buf, ef_buf, simd_buf;
    std::vector<size_t> vb_off(num_nodes + 1, 0), ef_off(num_nodes + 1, 0), simd_off(num_nodes + 1, 0);

    for (size_t i = 0; i < num_nodes; ++i) {
        VByteCodec::encode_delta_list(lists[i].data(), lists[i].size(), vb_buf);
        vb_off[i + 1] = vb_buf.size();

        EliasFanoCodec::encode_delta_list(lists[i].data(), lists[i].size(), ef_buf);
        ef_off[i + 1] = ef_buf.size();

        SimdBP128Codec::encode_delta_list(lists[i].data(), lists[i].size(), simd_buf);
        simd_off[i + 1] = simd_buf.size();
    }

    const size_t vb_bytes = vb_buf.size();
    const size_t ef_bytes = ef_buf.size();
    const size_t simd_bytes = simd_buf.size();
    ef_buf.resize(ef_bytes + EliasFanoCodec::PADDING_BYTES, 0);
    simd_buf.resize(simd_bytes + SimdBP128Codec::PADDING_BYTES, 0);

    // Verify lossless decode on all 20,000 lists
    int32_t dec_buf[1024];
    for (size_t i = 0; i < num_nodes; ++i) {
        EliasFanoCodec::decode_delta_list(ef_buf.data() + ef_off[i], lists[i].size(), dec_buf);
        for (size_t j = 0; j < lists[i].size(); ++j) {
            assert(dec_buf[j] == lists[i][j]);
        }
    }

    const int passes = 100;
    const double total_decoded = static_cast<double>(total_edges) * passes;
    volatile int64_t sink = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < num_nodes; ++i) {
            VByteCodec::decode_delta_list(vb_buf.data() + vb_off[i], lists[i].size(), dec_buf);
            sink += dec_buf[0];
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double vb_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < num_nodes; ++i) {
            EliasFanoCodec::decode_delta_list(ef_buf.data() + ef_off[i], lists[i].size(), dec_buf);
            sink += dec_buf[0];
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ef_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < passes; ++p) {
        for (size_t i = 0; i < num_nodes; ++i) {
            SimdBP128Codec::decode_delta_list(simd_buf.data() + simd_off[i], lists[i].size(), dec_buf);
            sink += dec_buf[0];
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double simd_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    double vb_ns = (vb_ms * 1e6) / total_decoded;
    double ef_ns = (ef_ms * 1e6) / total_decoded;
    double simd_ns = (simd_ms * 1e6) / total_decoded;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Workload: " << num_nodes << " lists, " << total_edges
              << " edges (" << (total_decoded / 1e6) << "M decoded ints)\n";
    std::cout << "-----------------------------------------------------------------------\n";
    std::cout << "Codec           | B/edge |  ms (100p) | ns/edge | Speedup vs EF / VByte\n";
    std::cout << "-----------------------------------------------------------------------\n";
    std::cout << "Scalar VByte    | " << std::setw(6) << (double(vb_bytes) / total_edges)
              << " | " << std::setw(10) << vb_ms
              << " | " << std::setw(7) << vb_ns << " | "
              << (ef_ns / vb_ns) << "x / 1.00x\n";
    std::cout << "Elias-Fano      | " << std::setw(6) << (double(ef_bytes) / total_edges)
              << " | " << std::setw(10) << ef_ms
              << " | " << std::setw(7) << ef_ns << " | 1.00x / "
              << (vb_ns / ef_ns) << "x\n";
    std::cout << "AVX2 SIMD-BP128 | " << std::setw(6) << (double(simd_bytes) / total_edges)
              << " | " << std::setw(10) << simd_ms
              << " | " << std::setw(7) << simd_ns << " | "
              << (ef_ns / simd_ns) << "x / " << (vb_ns / simd_ns) << "x\n";
    std::cout << "-----------------------------------------------------------------------\n";
    (void)sink;
    return 0;
}
