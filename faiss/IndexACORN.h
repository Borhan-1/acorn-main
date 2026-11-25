/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ACORN.h>
#include <faiss/utils/utils.h>

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

namespace faiss {

struct IndexACORN;




/** The ACORN index is a normal random-access index with a ACORN
 * link structure built on top */

struct IndexACORN : Index {
    typedef ACORN::storage_idx_t storage_idx_t;

    // the link strcuture
    ACORN acorn; // TODO change to hybrid

    // the sequential storage
    bool own_fields;
    Index* storage;

    // Graph-aware ID reordering state (maps between external original IDs and internal reordered IDs)
    bool is_reordered = false;
    std::vector<storage_idx_t> old_to_new;
    std::vector<storage_idx_t> new_to_old;
    std::vector<int> reordered_metadata;

    // Cluster-Gapped ID state for dynamic insertions into compressed ACORN (Part 3D)
    bool has_dynamic_gaps = false;
    size_t gap_cluster_stride = 0; // occupied + gap slots per cluster
    size_t gap_occupied_per_cluster = 0;
    size_t dynamic_external_count = 0;
    std::vector<storage_idx_t> cluster_free_cursor; // next free gap ID inside each cluster
    std::vector<storage_idx_t> cluster_end_id;      // exclusive end ID of each cluster

//     ReconstructFromNeighbors* reconstruct_from_neighbors;

    explicit IndexACORN(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2); // defaults d = 0, M=32, gamma=1
    explicit IndexACORN(Index* storage, int M, int gamma, std::vector<int>& metadata, int M_beta);
//     explicit IndexACORN(); // TODO check this is right

    ~IndexACORN() override;

    // add n vectors of dimension d to the index, x is the matrix of vectors
    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    // search for metadata
    // this doesn't override normal search since definition has a filter param - search is overloaded
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            char* filter_id_map,
            const SearchParameters* params = nullptr) const;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    /// Reorder graph nodes and vector storage using flat Level-0 BFS to localize
    /// neighbor IDs and minimize delta-encoding sizes (lossless via ID map).
    void reorder_graph_bfs();

    /// Reorder graph nodes using Hierarchical HNSW-Guided Reordering (Phase 3):
    /// orders upper-level centroids first, then runs bounded localized Level-0 BFS
    /// (peak queue memory O(local_bfs_budget) instead of O(N) for 100M-500M scale).
    /// Returns peak BFS queue size (in elements) for memory-scaling telemetry.
    size_t reorder_graph_hierarchical(size_t local_bfs_budget = 128);

    /// Compress Level-0 neighbor lists using Delta + VByte encoding (Option A).
    /// If free_uncompressed_l0 is true, compacts neighbors[] to upper levels only.
    void compress_neighbors_vbyte(bool free_uncompressed_l0 = true);

    /// Compress Level-0 neighbor lists using AVX2 SIMD-BP128 bit-packing (Option B).
    /// If proximity_prefix > 0 (and gamma == 1), splits each list into [0..prefix) and
    /// [prefix..count) so closest neighbors are visited first during filtered search.
    void compress_neighbors_simd_bp128(
            bool free_uncompressed_l0 = true,
            int proximity_prefix = 16);

    /// Part 3D: Expand a reordered index into a Cluster-Gapped ID space before compression,
    /// leaving `gap_ratio` free ID slots at the end of every `cluster_size`-node BFS block.
    void enable_dynamic_gaps(double gap_ratio = 0.05, size_t cluster_size = 1000);

    /// Part 3D: Dynamically insert `n` new vectors into a compressed ACORN index using
    /// nearest-cluster free gap IDs + Delta-Overflow back-edge buffer.
    /// Returns number of insertions that landed in their primary cluster's local gap.
    size_t add_dynamic(
            idx_t n,
            const float* x,
            const int* new_metadata = nullptr,
            int ef_insert = 32);

    /// Part 3D: Merge Delta-Overflow edges back into the AVX2 SIMD-BP128 compressed L0 array.
    void compact_dynamic_l0(int proximity_prefix = 16);

    

    // added for debugging
    void printStats(bool print_edge_list=false, bool print_filtered_edge_lists=false, int filter=-1, Operation op=EQUAL);

    private:
        const int debugFlag = 0;

        void debugTime() {
                if (debugFlag) {
                struct timeval tval;
                gettimeofday(&tval, NULL);
                time_t sec = tval.tv_sec;
                struct tm *tm_info = localtime(&sec);
                char timeBuff[25] = "";
                strftime(timeBuff, 25, "%H:%M:%S", tm_info);
                char timeBuffWithMilli[50] = "";
                sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
                std::string timestamp(timeBuffWithMilli);
                        std::cout << timestamp << std::flush;
        }
        }

        //needs atleast 2 args always
        //  alt debugFlag = 1 // fprintf(stderr, fmt, __VA_ARGS__); 
        #define debug(fmt, ...) \
        do { \
                if (debugFlag == 1) { \
                fprintf(stdout, "--" fmt, __VA_ARGS__);\
                } \
                if (debugFlag == 2) { \
                debugTime(); \
                fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
                } \
        } while (0)



        double elapsed() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
        }
};

/** Flat index topped with with a ACORN structure to access elements
 *  more efficiently.
 */

struct IndexACORNFlat : IndexACORN {
    IndexACORNFlat();
    IndexACORNFlat(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2);

};






} // namespace faiss
