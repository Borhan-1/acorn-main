#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <chrono>
#include "../faiss/impl/codec/vbyte_codec.h"

int main() {
    std::cout << "=== Testing VByteCodec with Simulated Graph Neighbor Lists ===\n";

    // Simulate 1000 nodes, each with 112 neighbors (ACORN Level 0 size)
    const size_t num_nodes = 1000;
    const size_t neighbors_per_node = 112;
    const uint32_t total_nodes = 1000000; // 1M range

    std::mt19937 rng(42);
    // Cluster simulation: neighbors are picked within local clusters (simulating BFS reordering)
    // Local window of +/- 3000 IDs around a base ID
    std::vector<std::vector<int32_t>> original_neighbor_lists(num_nodes);

    size_t raw_bytes = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        uint32_t base_id = (i * 1000) % (total_nodes - 5000);
        std::uniform_int_distribution<uint32_t> dist(base_id, base_id + 4000);
        
        std::vector<int32_t> nlist;
        for (size_t j = 0; j < neighbors_per_node; ++j) {
            nlist.push_back(static_cast<int32_t>(dist(rng)));
        }
        std::sort(nlist.begin(), nlist.end());
        nlist.erase(std::unique(nlist.begin(), nlist.end()), nlist.end());
        while (nlist.size() < neighbors_per_node) {
            nlist.push_back(nlist.back() + 1 + (rng() % 50));
        }

        original_neighbor_lists[i] = nlist;
        raw_bytes += nlist.size() * sizeof(int32_t);
    }

    std::cout << "Generated " << num_nodes << " nodes x " << neighbors_per_node << " neighbors.\n";
    std::cout << "Raw uncompressed memory: " << raw_bytes << " bytes (" << (raw_bytes / 1024.0) << " KB)\n";

    // Compress using VByteCodec
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<uint8_t>> compressed_lists(num_nodes);
    size_t compressed_bytes = 0;

    for (size_t i = 0; i < num_nodes; ++i) {
        faiss::codec::VByteCodec::encode_delta_list(
            original_neighbor_lists[i].data(),
            original_neighbor_lists[i].size(),
            compressed_lists[i]
        );
        compressed_bytes += compressed_lists[i].size();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double enc_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Compressed memory: " << compressed_bytes << " bytes (" << (compressed_bytes / 1024.0) << " KB)\n";
    double compression_ratio = 100.0 * (1.0 - (double)compressed_bytes / raw_bytes);
    double avg_bytes_per_neighbor = (double)compressed_bytes / (num_nodes * neighbors_per_node);
    std::cout << "--> Memory Savings: " << compression_ratio << " %\n";
    std::cout << "--> Average bytes per neighbor: " << avg_bytes_per_neighbor << " bytes (Down from 4.0 bytes!)\n";
    std::cout << "--> Total Encoding Time: " << enc_time_ms << " ms\n";

    // Decompress and verify 100% lossless correctness
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> decode_buf(neighbors_per_node);
    for (size_t i = 0; i < num_nodes; ++i) {
        faiss::codec::VByteCodec::decode_delta_list(
            compressed_lists[i].data(),
            original_neighbor_lists[i].size(),
            decode_buf.data()
        );

        for (size_t j = 0; j < neighbors_per_node; ++j) {
            assert(decode_buf[j] == original_neighbor_lists[i][j]);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double dec_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double dec_ns_per_neighbor = (dec_time_ms * 1e6) / (num_nodes * neighbors_per_node);

    std::cout << "Decompression verification: ALL 100% MATCH! (Lossless)\n";
    std::cout << "--> Decompression speed: " << dec_ns_per_neighbor << " nanoseconds per neighbor\n";
    std::cout << "=== SUCCESS: VByteCodec unit test passed! ===\n";

    return 0;
}

