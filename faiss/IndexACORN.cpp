/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexACORN.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <queue>
#include <unordered_set>

#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

// # added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>





extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

using MinimaxHeap = ACORN::MinimaxHeap;
using storage_idx_t = ACORN::storage_idx_t;
using NodeDistFarther = ACORN::NodeDistFarther;

ACORNStats acorn_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() {
        delete basedis;
    }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (storage->metric_type == METRIC_INNER_PRODUCT) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

// TODO
void acorn_add_vertices(
        IndexACORN& index_acorn,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_acorn.d;
    ACORN& acorn = index_acorn.acorn;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("acorn_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = acorn.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = acorn.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = acorn.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_acorn.d * acorn.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                DistanceComputer* dis =
                        storage_distance_computer(index_acorn.storage);
                ScopeDeleter1<DistanceComputer> del(dis);
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    acorn.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}


} // namespace

/**************************************************************
 * IndexACORN implementation
 **************************************************************/

IndexACORN::IndexACORN(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric)
        : Index(d, metric),
          acorn(M, gamma, metadata, M_beta),
          own_fields(false),
          storage(nullptr)
          /* reconstruct_from_neighbors(nullptr)*/ {}

IndexACORN::IndexACORN(Index* storage, int M, int gamma, std::vector<int>& metadata, int M_beta)
        : Index(storage->d, storage->metric_type),
          acorn(M, gamma, metadata, M_beta), // TOOD acorn needs to keep metadata now
          own_fields(false),
          storage(storage)
          /* reconstruct_from_neighbors(nullptr) */ {}

IndexACORN::~IndexACORN() {
    if (own_fields) {
        delete storage;
    }
}

void IndexACORN::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    // acorn structure does not require training
    storage->train(n, x);
    is_trained = true;
}

// overloaded search for hybrid search 
void IndexACORN::search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            char* filter_id_map,
            const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    const SearchParametersACORN* params = nullptr;

    int efSearch = acorn.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersACORN*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;
    size_t decode_calls = 0, decoded_edges = 0;
    double decode_time_us = 0.0;
    double candidates_loop = 0, neighbors_loop = 0, tuple_unwrap = 0, skips = 0, visits = 0; //added for profiling

    idx_t check_period =
            InterruptCallback::get_period_hint(acorn.max_level * d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            std::vector<char> local_filter;
            bool need_filter_remap =
                    is_reordered && !(params && params->assume_reordered_filter);
            if (need_filter_remap) {
                local_filter.resize(ntotal);
            }

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder, candidates_loop, decode_calls, decoded_edges, decode_time_us)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                char* filters = filter_id_map + i * ntotal;
                if (need_filter_remap) {
                    for (idx_t nid = 0; nid < ntotal; nid++) {
                        storage_idx_t ext_id = new_to_old[nid];
                        local_filter[nid] = (ext_id >= 0) ? filters[ext_id] : 0;
                    }
                    filters = local_filter.data();
                }
                dis->set_query(x + i * d);

                maxheap_heapify(k, simi, idxi);
                ACORNStats stats = acorn.hybrid_search(*dis, k, idxi, simi, vt, filters, params); //TODO edit to hybrid search

                // ACORNStats stats = acorn.hybrid_search(*dis, k, idxi, simi, vt, filters[i], op, regex, params); //TODO edit to hybrid search
                n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                nreorder += stats.nreorder;
                decode_calls += stats.decode_calls;
                decoded_edges += stats.decoded_edges;
                decode_time_us += stats.decode_time_us;
                // printf("index -- stats updates: %f\n", stats.candidates_loop);
                // printf("index -- stats updates: %f\n", stats.neighbors_loop);
                //added for profiling
                candidates_loop += stats.candidates_loop; 
                neighbors_loop += stats.neighbors_loop;
                tuple_unwrap += stats.tuple_unwrap;
                skips += stats.skips;
                visits += stats.visits;
                maxheap_reorder(k, simi, idxi);
                if (is_reordered) {
                    for (idx_t j = 0; j < k; j++) {
                        if (idxi[j] >= 0) {
                            idxi[j] = new_to_old[idxi[j]];
                        }
                    }
                }
                
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }

    acorn_stats.combine({n1, n2, n3, ndis, nreorder, candidates_loop, neighbors_loop, tuple_unwrap, skips, visits, decode_calls, decoded_edges, decode_time_us}); //added for profiling
}

// TODO figure out what do with this
void IndexACORN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    const SearchParametersACORN* params = nullptr;

    int efSearch = acorn.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersACORN*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;
    size_t decode_calls = 0, decoded_edges = 0;
    double decode_time_us = 0.0;
    
    idx_t check_period =
            InterruptCallback::get_period_hint(acorn.max_level * d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder, decode_calls, decoded_edges, decode_time_us)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                maxheap_heapify(k, simi, idxi);
                ACORNStats stats = acorn.search(*dis, k, idxi, simi, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                nreorder += stats.nreorder;
                decode_calls += stats.decode_calls;
                decoded_edges += stats.decoded_edges;
                decode_time_us += stats.decode_time_us;
                maxheap_reorder(k, simi, idxi);
                if (is_reordered) {
                    for (idx_t j = 0; j < k; j++) {
                        if (idxi[j] >= 0) {
                            idxi[j] = new_to_old[idxi[j]];
                        }
                    }
                }

                
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }

    acorn_stats.combine({n1, n2, n3, ndis, nreorder, 0.0, 0.0, 0.0, 0.0, 0.0, decode_calls, decoded_edges, decode_time_us});
}

// add n vectors of dimension d to the index, x is the matrix of vectors TODO
void IndexACORN::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) instead of IndexACORN directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    acorn_add_vertices(*this, n0, n, x, verbose, acorn.levels.size() == ntotal);
}

void IndexACORN::reset() {
    acorn.reset();
    storage->reset();
    ntotal = 0;
    is_reordered = false;
    old_to_new.clear();
    new_to_old.clear();
    reordered_metadata.clear();
    has_dynamic_gaps = false;
    gap_cluster_stride = 0;
    gap_occupied_per_cluster = 0;
    dynamic_external_count = 0;
    cluster_free_cursor.clear();
    cluster_end_id.clear();
}

void IndexACORN::reconstruct(idx_t key, float* recons) const {
    if (is_reordered) {
        key = old_to_new[key];
    }
    storage->reconstruct(key, recons);
}

namespace {

/// Shared permutation applier for both flat BFS and hierarchical reordering.
void apply_acorn_permutation(
        IndexACORN& index,
        const std::vector<ACORN::storage_idx_t>& old_to_new,
        const std::vector<ACORN::storage_idx_t>& new_to_old) {
    const size_t n = static_cast<size_t>(index.ntotal);
    ACORN& acorn = index.acorn;

    // 1. Permute underlying vector storage IN-PLACE via cycle-leader traversal
    //    (O(1) extra vector RAM + 1 bit/node visited bitset, eliminating 2x vector RAM peak)
    if (auto* flat_codes = dynamic_cast<IndexFlatCodes*>(index.storage)) {
        const size_t cs = flat_codes->code_size;
        uint8_t* base = flat_codes->codes.data();
        std::vector<uint64_t> done((n + 63) >> 6, 0ULL);
        std::vector<uint8_t> tmp_vec(cs);

        for (size_t start = 0; start < n; ++start) {
            if ((done[start >> 6] >> (start & 63)) & 1ULL) {
                continue;
            }
            size_t first_src = static_cast<size_t>(new_to_old[start]);
            if (first_src == start) {
                done[start >> 6] |= (1ULL << (start & 63));
                continue;
            }
            std::memcpy(tmp_vec.data(), base + start * cs, cs);
            size_t cur = start;
            while (true) {
                size_t src = static_cast<size_t>(new_to_old[cur]);
                done[cur >> 6] |= (1ULL << (cur & 63));
                if (src == start) {
                    std::memcpy(base + cur * cs, tmp_vec.data(), cs);
                    break;
                }
                std::memcpy(base + cur * cs, base + src * cs, cs);
                cur = src;
            }
        }
    } else {
        FAISS_THROW_MSG("Graph reordering currently requires IndexFlatCodes-backed storage");
    }

    // 2. Permute metadata copy if present
    if (acorn.metadata != nullptr) {
        index.reordered_metadata.resize(n);
        for (size_t new_id = 0; new_id < n; new_id++) {
            index.reordered_metadata[new_id] = acorn.metadata[new_to_old[new_id]];
        }
        acorn.metadata = index.reordered_metadata.data();
    }

    // 3. Permute levels and rebuild offsets
    std::vector<int> new_levels(n);
    std::vector<size_t> new_offsets(n + 1, 0);
    for (size_t new_id = 0; new_id < n; new_id++) {
        size_t old_id = static_cast<size_t>(new_to_old[new_id]);
        int lv = acorn.levels[old_id];
        new_levels[new_id] = lv;
        new_offsets[new_id + 1] = new_offsets[new_id] + acorn.cum_nb_neighbors(lv);
    }

    // 4. Permute and rewrite all neighbor IDs across all levels
    std::vector<ACORN::NeighNode> new_neighbors(new_offsets[n], ACORN::NeighNode(-1));
    for (size_t new_id = 0; new_id < n; new_id++) {
        size_t old_id = static_cast<size_t>(new_to_old[new_id]);
        int lv = new_levels[new_id];
        for (int layer = 0; layer < lv; layer++) {
            size_t old_begin = 0, old_end = 0;
            acorn.neighbor_range(static_cast<idx_t>(old_id), layer, &old_begin, &old_end);
            size_t dst = new_offsets[new_id] + acorn.cum_nb_neighbors(layer);
            for (size_t j = old_begin; j < old_end; j++) {
                ACORN::storage_idx_t v = acorn.neighbors[j];
                if (v < 0) {
                    break;
                }
                new_neighbors[dst++] = old_to_new[v];
            }
        }
    }

    acorn.levels.swap(new_levels);
    acorn.offsets.swap(new_offsets);
    acorn.neighbors.swap(new_neighbors);
    if (acorn.entry_point >= 0) {
        acorn.entry_point = old_to_new[acorn.entry_point];
    }
    index.is_reordered = true;
}

} // namespace

void IndexACORN::reorder_graph_bfs() {
    if (ntotal <= 1 || is_reordered) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) before reordering");

    const size_t n = static_cast<size_t>(ntotal);
    old_to_new.assign(n, -1);
    new_to_old.clear();
    new_to_old.reserve(n);

    std::vector<storage_idx_t> bfs_queue;
    bfs_queue.reserve(n);

    auto run_bfs_from = [&](storage_idx_t start_node) {
        if (start_node < 0 || static_cast<size_t>(start_node) >= n) {
            return;
        }
        if (old_to_new[start_node] != -1) {
            return;
        }
        old_to_new[start_node] = static_cast<storage_idx_t>(new_to_old.size());
        new_to_old.push_back(start_node);
        size_t head = bfs_queue.size();
        bfs_queue.push_back(start_node);

        while (head < bfs_queue.size()) {
            storage_idx_t u = bfs_queue[head++];
            size_t begin = 0, end = 0;
            acorn.neighbor_range(u, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                storage_idx_t v = acorn.neighbors[j];
                if (v < 0) {
                    break;
                }
                if (old_to_new[v] == -1) {
                    old_to_new[v] = static_cast<storage_idx_t>(new_to_old.size());
                    new_to_old.push_back(v);
                    bfs_queue.push_back(v);
                }
            }
        }
    };

    // Seed from ACORN entry point (top-level hub) first, then cover any remaining components
    if (acorn.entry_point >= 0) {
        run_bfs_from(acorn.entry_point);
    }
    for (size_t i = 0; i < n; i++) {
        if (old_to_new[i] == -1) {
            run_bfs_from(static_cast<storage_idx_t>(i));
        }
    }
    FAISS_ASSERT(new_to_old.size() == n);

    apply_acorn_permutation(*this, old_to_new, new_to_old);
}

size_t IndexACORN::reorder_graph_hierarchical(size_t local_bfs_budget) {
    if (ntotal <= 1 || is_reordered) {
        return 0;
    }
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexACORNFlat (or variants) before reordering");

    const size_t n = static_cast<size_t>(ntotal);
    if (local_bfs_budget == 0) {
        local_bfs_budget = 128;
    }

    old_to_new.assign(n, -1);
    new_to_old.clear();
    new_to_old.reserve(n);

    // Step 1: Collect and order upper-level centroids (levels[i] >= 2 => present on layer >= 1)
    // using BFS on layer 1 seeded from entry_point (only ~N/M nodes).
    std::vector<storage_idx_t> upper_centroids;
    std::vector<uint8_t> upper_seen(n, 0);
    size_t peak_queue = 0;

    auto bfs_upper_layer = [&](storage_idx_t seed, int layer) {
        if (seed < 0 || static_cast<size_t>(seed) >= n || upper_seen[seed]) {
            return;
        }
        if (acorn.levels[seed] <= layer) {
            return;
        }
        size_t head = upper_centroids.size();
        upper_seen[seed] = 1;
        upper_centroids.push_back(seed);
        while (head < upper_centroids.size()) {
            storage_idx_t u = upper_centroids[head++];
            if (acorn.levels[u] <= layer) {
                continue;
            }
            size_t begin = 0, end = 0;
            acorn.neighbor_range(u, layer, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                storage_idx_t v = acorn.neighbors[j];
                if (v < 0) {
                    break;
                }
                if (!upper_seen[v] && acorn.levels[v] > layer) {
                    upper_seen[v] = 1;
                    upper_centroids.push_back(v);
                }
            }
        }
    };

    if (acorn.max_level >= 1) {
        if (acorn.entry_point >= 0) {
            bfs_upper_layer(acorn.entry_point, 1);
        }
        for (size_t i = 0; i < n; i++) {
            if (acorn.levels[i] >= 2 && !upper_seen[i]) {
                bfs_upper_layer(static_cast<storage_idx_t>(i), 1);
            }
        }
    }
    peak_queue = std::max(peak_queue, upper_centroids.size());
    std::vector<uint8_t>().swap(upper_seen);

    // Step 2: Bounded localized Level-0 BFS around each ordered centroid (O(budget) queue)
    std::vector<storage_idx_t> local_q;
    local_q.reserve(local_bfs_budget + 128);

    auto run_bounded_l0_bfs = [&](storage_idx_t seed) {
        if (seed < 0 || static_cast<size_t>(seed) >= n || old_to_new[seed] != -1) {
            return;
        }
        local_q.clear();
        old_to_new[seed] = static_cast<storage_idx_t>(new_to_old.size());
        new_to_old.push_back(seed);
        local_q.push_back(seed);

        size_t head = 0;
        while (head < local_q.size() && local_q.size() < local_bfs_budget) {
            storage_idx_t u = local_q[head++];
            size_t begin = 0, end = 0;
            acorn.neighbor_range(u, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                storage_idx_t v = acorn.neighbors[j];
                if (v < 0) {
                    break;
                }
                if (old_to_new[v] == -1) {
                    old_to_new[v] = static_cast<storage_idx_t>(new_to_old.size());
                    new_to_old.push_back(v);
                    local_q.push_back(v);
                    if (local_q.size() >= local_bfs_budget) {
                        break;
                    }
                }
            }
        }
        peak_queue = std::max(peak_queue, local_q.size());
    };

    for (storage_idx_t c : upper_centroids) {
        run_bounded_l0_bfs(c);
    }

    // Step 3: Sweep any remaining unvisited Level-0 nodes with bounded local BFS
    for (size_t i = 0; i < n; i++) {
        if (old_to_new[i] == -1) {
            run_bounded_l0_bfs(static_cast<storage_idx_t>(i));
        }
    }
    FAISS_ASSERT(new_to_old.size() == n);

    apply_acorn_permutation(*this, old_to_new, new_to_old);
    return peak_queue;
}

void IndexACORN::compress_neighbors_vbyte(bool free_uncompressed_l0) {
    if (ntotal == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            !acorn.l0_freed,
            "Uncompressed Level-0 neighbors have already been freed");

    const size_t n = static_cast<size_t>(ntotal);
    const bool preserve_m_beta_split = (acorn.gamma > 1 && acorn.M_beta > 0);

    acorn.compressed_l0_neighbors.clear();
    acorn.compressed_l0_neighbors.reserve(n * 96);
    acorn.compressed_l0_offsets.assign(n + 1, 0);
    acorn.compressed_l0_counts.assign(n, 0);
    if (preserve_m_beta_split) {
        acorn.compressed_l0_prefix_counts.assign(n, 0);
    } else {
        acorn.compressed_l0_prefix_counts.clear();
    }

    std::vector<int32_t> tmp;
    tmp.reserve(static_cast<size_t>(acorn.nb_neighbors(0)));

    for (size_t i = 0; i < n; i++) {
        size_t begin = 0, end = 0;
        acorn.neighbor_range(static_cast<idx_t>(i), 0, &begin, &end);
        tmp.clear();
        for (size_t j = begin; j < end; j++) {
            storage_idx_t v = acorn.neighbors[j];
            if (v < 0) {
                break;
            }
            tmp.push_back(v);
        }

        const size_t count = tmp.size();
        acorn.compressed_l0_counts[i] = static_cast<uint16_t>(count);

        if (count > 0) {
            if (!preserve_m_beta_split) {
                std::sort(tmp.begin(), tmp.end());
                codec::VByteCodec::encode_delta_list(
                        tmp.data(), count, acorn.compressed_l0_neighbors);
            } else {
                size_t split = std::min(count, static_cast<size_t>(acorn.M_beta));
                acorn.compressed_l0_prefix_counts[i] = static_cast<uint16_t>(split);
                std::sort(tmp.begin(), tmp.begin() + split);
                codec::VByteCodec::encode_delta_list(
                        tmp.data(), split, acorn.compressed_l0_neighbors);
                if (count > split) {
                    std::sort(tmp.begin() + split, tmp.end());
                    codec::VByteCodec::encode_delta_list(
                            tmp.data() + split,
                            count - split,
                            acorn.compressed_l0_neighbors);
                }
            }
        }
        acorn.compressed_l0_offsets[i + 1] = acorn.compressed_l0_neighbors.size();
    }

    acorn.compressed_l0_neighbors.shrink_to_fit();
    acorn.use_compressed_l0 = true;
    acorn.l0_compression_mode = ACORN::L0CompressionMode::VBYTE;

    if (free_uncompressed_l0) {
        const size_t l0_slots = static_cast<size_t>(acorn.nb_neighbors(0));
        std::vector<size_t> upper_offsets(n + 1, 0);
        for (size_t i = 0; i < n; i++) {
            size_t total_slots = static_cast<size_t>(acorn.cum_nb_neighbors(acorn.levels[i]));
            size_t upper_slots = (total_slots > l0_slots) ? (total_slots - l0_slots) : 0;
            upper_offsets[i + 1] = upper_offsets[i] + upper_slots;
        }

        std::vector<ACORN::NeighNode> upper_neighbors(
                upper_offsets[n], ACORN::NeighNode(-1));
        for (size_t i = 0; i < n; i++) {
            size_t upper_slots = upper_offsets[i + 1] - upper_offsets[i];
            if (upper_slots > 0) {
                size_t old_upper_begin = acorn.offsets[i] + l0_slots;
                std::memcpy(
                        upper_neighbors.data() + upper_offsets[i],
                        acorn.neighbors.data() + old_upper_begin,
                        upper_slots * sizeof(ACORN::NeighNode));
            }
        }

        acorn.offsets.swap(upper_offsets);
        acorn.neighbors.swap(upper_neighbors);
        std::vector<ACORN::NeighNode>().swap(upper_neighbors); // release old L0 buffer immediately
        acorn.l0_freed = true;
    }
}

void IndexACORN::compress_neighbors_simd_bp128(
        bool free_uncompressed_l0,
        int proximity_prefix) {
    if (ntotal == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            !acorn.l0_freed,
            "Uncompressed Level-0 neighbors have already been freed");

    const size_t n = static_cast<size_t>(ntotal);
    const size_t split_limit = (acorn.gamma > 1 && acorn.M_beta > 0)
            ? static_cast<size_t>(acorn.M_beta)
            : (proximity_prefix > 0 ? static_cast<size_t>(proximity_prefix) : 0);
    const bool use_two_segments = (split_limit > 0);

    acorn.compressed_l0_neighbors.clear();
    acorn.compressed_l0_neighbors.reserve(n * 112);
    acorn.compressed_l0_offsets.assign(n + 1, 0);
    acorn.compressed_l0_counts.assign(n, 0);
    if (use_two_segments) {
        acorn.compressed_l0_prefix_counts.assign(n, 0);
    } else {
        acorn.compressed_l0_prefix_counts.clear();
    }

    std::vector<int32_t> tmp;
    tmp.reserve(static_cast<size_t>(acorn.nb_neighbors(0)));

    for (size_t i = 0; i < n; i++) {
        size_t begin = 0, end = 0;
        acorn.neighbor_range(static_cast<idx_t>(i), 0, &begin, &end);
        tmp.clear();
        for (size_t j = begin; j < end; j++) {
            storage_idx_t v = acorn.neighbors[j];
            if (v < 0) {
                break;
            }
            tmp.push_back(v);
        }

        const size_t count = tmp.size();
        acorn.compressed_l0_counts[i] = static_cast<uint16_t>(count);

        if (count > 0) {
            if (!use_two_segments) {
                std::sort(tmp.begin(), tmp.end());
                codec::SimdBP128Codec::encode_delta_list(
                        tmp.data(), count, acorn.compressed_l0_neighbors);
            } else {
                size_t split = std::min(count, split_limit);
                acorn.compressed_l0_prefix_counts[i] = static_cast<uint16_t>(split);
                std::sort(tmp.begin(), tmp.begin() + split);
                codec::SimdBP128Codec::encode_delta_list(
                        tmp.data(), split, acorn.compressed_l0_neighbors);
                if (count > split) {
                    std::sort(tmp.begin() + split, tmp.end());
                    codec::SimdBP128Codec::encode_delta_list(
                            tmp.data() + split,
                            count - split,
                            acorn.compressed_l0_neighbors);
                }
            }
        }
        acorn.compressed_l0_offsets[i + 1] = acorn.compressed_l0_neighbors.size();
    }

    // Append trailing zero padding for safe 128-bit/256-bit unaligned AVX2 loads
    acorn.compressed_l0_neighbors.resize(
            acorn.compressed_l0_neighbors.size() + codec::SimdBP128Codec::PADDING_BYTES,
            0);
    acorn.compressed_l0_neighbors.shrink_to_fit();
    acorn.use_compressed_l0 = true;
    acorn.l0_compression_mode = ACORN::L0CompressionMode::SIMD_BP128;

    if (free_uncompressed_l0) {
        const size_t l0_slots = static_cast<size_t>(acorn.nb_neighbors(0));
        std::vector<size_t> upper_offsets(n + 1, 0);
        for (size_t i = 0; i < n; i++) {
            size_t total_slots = static_cast<size_t>(acorn.cum_nb_neighbors(acorn.levels[i]));
            size_t upper_slots = (total_slots > l0_slots) ? (total_slots - l0_slots) : 0;
            upper_offsets[i + 1] = upper_offsets[i] + upper_slots;
        }

        std::vector<ACORN::NeighNode> upper_neighbors(
                upper_offsets[n], ACORN::NeighNode(-1));
        for (size_t i = 0; i < n; i++) {
            size_t upper_slots = upper_offsets[i + 1] - upper_offsets[i];
            if (upper_slots > 0) {
                size_t old_upper_begin = acorn.offsets[i] + l0_slots;
                std::memcpy(
                        upper_neighbors.data() + upper_offsets[i],
                        acorn.neighbors.data() + old_upper_begin,
                        upper_slots * sizeof(ACORN::NeighNode));
            }
        }

        acorn.offsets.swap(upper_offsets);
        acorn.neighbors.swap(upper_neighbors);
        std::vector<ACORN::NeighNode>().swap(upper_neighbors);
        acorn.l0_freed = true;
    }
}

void IndexACORN::compress_neighbors_elias_fano(
        bool free_uncompressed_l0,
        int proximity_prefix) {
    if (ntotal == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            !acorn.l0_freed,
            "Uncompressed Level-0 neighbors have already been freed");

    const size_t n = static_cast<size_t>(ntotal);
    const size_t split_limit = (acorn.gamma > 1 && acorn.M_beta > 0)
            ? static_cast<size_t>(acorn.M_beta)
            : (proximity_prefix > 0 ? static_cast<size_t>(proximity_prefix) : 0);
    const bool use_two_segments = (split_limit > 0);

    acorn.compressed_l0_neighbors.clear();
    acorn.compressed_l0_neighbors.reserve(n * 112);
    acorn.compressed_l0_offsets.assign(n + 1, 0);
    acorn.compressed_l0_counts.assign(n, 0);
    if (use_two_segments) {
        acorn.compressed_l0_prefix_counts.assign(n, 0);
    } else {
        acorn.compressed_l0_prefix_counts.clear();
    }

    std::vector<int32_t> tmp;
    tmp.reserve(static_cast<size_t>(acorn.nb_neighbors(0)));

    for (size_t i = 0; i < n; i++) {
        size_t begin = 0, end = 0;
        acorn.neighbor_range(static_cast<idx_t>(i), 0, &begin, &end);
        tmp.clear();
        for (size_t j = begin; j < end; j++) {
            storage_idx_t v = acorn.neighbors[j];
            if (v < 0) {
                break;
            }
            tmp.push_back(v);
        }

        const size_t count = tmp.size();
        acorn.compressed_l0_counts[i] = static_cast<uint16_t>(count);

        if (count > 0) {
            if (!use_two_segments) {
                std::sort(tmp.begin(), tmp.end());
                codec::EliasFanoCodec::encode_delta_list(
                        tmp.data(), count, acorn.compressed_l0_neighbors);
            } else {
                size_t split = std::min(count, split_limit);
                acorn.compressed_l0_prefix_counts[i] = static_cast<uint16_t>(split);
                std::sort(tmp.begin(), tmp.begin() + split);
                codec::EliasFanoCodec::encode_delta_list(
                        tmp.data(), split, acorn.compressed_l0_neighbors);
                if (count > split) {
                    std::sort(tmp.begin() + split, tmp.end());
                    codec::EliasFanoCodec::encode_delta_list(
                            tmp.data() + split,
                            count - split,
                            acorn.compressed_l0_neighbors);
                }
            }
        }
        acorn.compressed_l0_offsets[i + 1] = acorn.compressed_l0_neighbors.size();
    }

    acorn.compressed_l0_neighbors.resize(
            acorn.compressed_l0_neighbors.size() + codec::EliasFanoCodec::PADDING_BYTES,
            0);
    acorn.compressed_l0_neighbors.shrink_to_fit();
    acorn.use_compressed_l0 = true;
    acorn.l0_compression_mode = ACORN::L0CompressionMode::ELIAS_FANO;

    if (free_uncompressed_l0) {
        const size_t l0_slots = static_cast<size_t>(acorn.nb_neighbors(0));
        std::vector<size_t> upper_offsets(n + 1, 0);
        for (size_t i = 0; i < n; i++) {
            size_t total_slots = static_cast<size_t>(acorn.cum_nb_neighbors(acorn.levels[i]));
            size_t upper_slots = (total_slots > l0_slots) ? (total_slots - l0_slots) : 0;
            upper_offsets[i + 1] = upper_offsets[i] + upper_slots;
        }

        std::vector<ACORN::NeighNode> upper_neighbors(
                upper_offsets[n], ACORN::NeighNode(-1));
        for (size_t i = 0; i < n; i++) {
            size_t upper_slots = upper_offsets[i + 1] - upper_offsets[i];
            if (upper_slots > 0) {
                size_t old_upper_begin = acorn.offsets[i] + l0_slots;
                std::memcpy(
                        upper_neighbors.data() + upper_offsets[i],
                        acorn.neighbors.data() + old_upper_begin,
                        upper_slots * sizeof(ACORN::NeighNode));
            }
        }

        acorn.offsets.swap(upper_offsets);
        acorn.neighbors.swap(upper_neighbors);
        std::vector<ACORN::NeighNode>().swap(upper_neighbors);
        acorn.l0_freed = true;
    }
}

void IndexACORN::enable_dynamic_gaps(double gap_ratio, size_t cluster_size) {
    if (ntotal == 0 || gap_ratio <= 0.0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            !acorn.use_compressed_l0,
            "Call enable_dynamic_gaps() after reordering and BEFORE compressing L0");
    FAISS_THROW_IF_NOT_MSG(
            is_reordered,
            "Call reorder_graph_hierarchical() or reorder_graph_bfs() before enable_dynamic_gaps()");

    const size_t old_n = static_cast<size_t>(ntotal);
    if (cluster_size == 0) {
        cluster_size = 1000;
    }
    const size_t gap_per_cluster =
            std::max<size_t>(1, static_cast<size_t>(std::llround(cluster_size * gap_ratio)));
    const size_t stride = cluster_size + gap_per_cluster;
    const size_t num_clusters = (old_n + cluster_size - 1) / cluster_size;
    const size_t new_cap = num_clusters * stride;

    // Map dense reordered ID `d` -> gapped internal ID `g`
    std::vector<storage_idx_t> dense_to_gap(old_n);
    std::vector<storage_idx_t> new_n2o(new_cap, -1);
    std::vector<storage_idx_t> new_o2n(old_n, -1);

    cluster_free_cursor.resize(num_clusters);
    cluster_end_id.resize(num_clusters);

    for (size_t c = 0; c < num_clusters; c++) {
        size_t d_start = c * cluster_size;
        size_t d_end = std::min(old_n, d_start + cluster_size);
        size_t g_base = c * stride;
        for (size_t d = d_start; d < d_end; d++) {
            storage_idx_t g = static_cast<storage_idx_t>(g_base + (d - d_start));
            dense_to_gap[d] = g;
            storage_idx_t ext_id = new_to_old[d];
            new_n2o[g] = ext_id;
            new_o2n[ext_id] = g;
        }
        cluster_free_cursor[c] = static_cast<storage_idx_t>(g_base + (d_end - d_start));
        cluster_end_id[c] = static_cast<storage_idx_t>(g_base + stride);
    }

    // 1. Expand IndexFlatCodes vector storage with empty gap slots
    if (auto* flat_codes = dynamic_cast<IndexFlatCodes*>(storage)) {
        const size_t cs = flat_codes->code_size;
        std::vector<uint8_t> gapped_codes(new_cap * cs, 0);
        for (size_t d = 0; d < old_n; d++) {
            size_t g = static_cast<size_t>(dense_to_gap[d]);
            std::memcpy(
                    gapped_codes.data() + g * cs,
                    flat_codes->codes.data() + d * cs,
                    cs);
        }
        flat_codes->codes.swap(gapped_codes);
        flat_codes->ntotal = static_cast<idx_t>(new_cap);
    } else {
        FAISS_THROW_MSG("enable_dynamic_gaps requires IndexFlatCodes-backed storage");
    }

    // 2. Expand metadata
    if (acorn.metadata != nullptr) {
        std::vector<int> gapped_meta(new_cap, -1);
        for (size_t d = 0; d < old_n; d++) {
            gapped_meta[dense_to_gap[d]] = acorn.metadata[d];
        }
        reordered_metadata.swap(gapped_meta);
        acorn.metadata = reordered_metadata.data();
    }

    // 3. Expand levels, offsets, and rewrite neighbor IDs into gapped space
    std::vector<int> gapped_levels(new_cap, 1);
    for (size_t d = 0; d < old_n; d++) {
        gapped_levels[dense_to_gap[d]] = acorn.levels[d];
    }
    std::vector<size_t> gapped_offsets(new_cap + 1, 0);
    for (size_t g = 0; g < new_cap; g++) {
        gapped_offsets[g + 1] = gapped_offsets[g] + acorn.cum_nb_neighbors(gapped_levels[g]);
    }
    std::vector<ACORN::NeighNode> gapped_neighbors(gapped_offsets[new_cap], ACORN::NeighNode(-1));
    for (size_t d = 0; d < old_n; d++) {
        size_t g = static_cast<size_t>(dense_to_gap[d]);
        int lv = gapped_levels[g];
        for (int layer = 0; layer < lv; layer++) {
            size_t old_begin = 0, old_end = 0;
            acorn.neighbor_range(static_cast<idx_t>(d), layer, &old_begin, &old_end);
            size_t dst = gapped_offsets[g] + acorn.cum_nb_neighbors(layer);
            for (size_t j = old_begin; j < old_end; j++) {
                storage_idx_t v = acorn.neighbors[j];
                if (v < 0) {
                    break;
                }
                gapped_neighbors[dst++] = dense_to_gap[v];
            }
        }
    }

    acorn.levels.swap(gapped_levels);
    acorn.offsets.swap(gapped_offsets);
    acorn.neighbors.swap(gapped_neighbors);
    if (acorn.entry_point >= 0) {
        acorn.entry_point = dense_to_gap[acorn.entry_point];
    }

    old_to_new.swap(new_o2n);
    new_to_old.swap(new_n2o);
    ntotal = static_cast<idx_t>(new_cap);
    has_dynamic_gaps = true;
    gap_cluster_stride = stride;
    gap_occupied_per_cluster = cluster_size;
    dynamic_external_count = old_n;
}

size_t IndexACORN::add_dynamic(
        idx_t n_add,
        const float* x,
        const int* new_metadata,
        int ef_insert) {
    if (n_add <= 0) {
        return 0;
    }
    FAISS_THROW_IF_NOT_MSG(
            acorn.use_compressed_l0 && has_dynamic_gaps,
            "Call enable_dynamic_gaps() and compress_neighbors_simd_bp128() before add_dynamic()");

    auto* flat_codes = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(flat_codes != nullptr, "add_dynamic requires IndexFlatCodes storage");

    if (acorn.has_overflow.size() < static_cast<size_t>(ntotal)) {
        acorn.has_overflow.resize(static_cast<size_t>(ntotal), 0);
    }

    const size_t cs = flat_codes->code_size;
    const size_t num_clusters = cluster_free_cursor.size();
    const int k_link = std::min(acorn.M, 32);

    SearchParametersACORN sp;
    sp.efSearch = std::max(ef_insert, k_link);

    VisitedTable vt(static_cast<size_t>(ntotal));
    DistanceComputer* dis = storage_distance_computer(storage);
    ScopeDeleter1<DistanceComputer> del(dis);

    std::vector<float> D_tmp(k_link);
    std::vector<idx_t> I_tmp(k_link);
    size_t local_cluster_hits = 0;

    for (idx_t i = 0; i < n_add; i++) {
        const float* vec = x + static_cast<size_t>(i) * d;
        dis->set_query(vec);
        maxheap_heapify(k_link, D_tmp.data(), I_tmp.data());
        acorn.search(*dis, k_link, I_tmp.data(), D_tmp.data(), vt, &sp);
        maxheap_reorder(k_link, D_tmp.data(), I_tmp.data());

        // Pick nearest existing neighbor to determine target BFS cluster
        storage_idx_t anchor = (I_tmp[0] >= 0) ? static_cast<storage_idx_t>(I_tmp[0]) : 0;
        size_t target_c = std::min(
                num_clusters - 1,
                static_cast<size_t>(anchor) / gap_cluster_stride);

        // Allocate nearest free gap ID (O(1) local cluster check, then radial probe)
        storage_idx_t slot = -1;
        if (cluster_free_cursor[target_c] < cluster_end_id[target_c]) {
            slot = cluster_free_cursor[target_c]++;
            local_cluster_hits++;
        } else {
            for (size_t delta = 1; delta < num_clusters; delta++) {
                if (target_c + delta < num_clusters &&
                    cluster_free_cursor[target_c + delta] < cluster_end_id[target_c + delta]) {
                    slot = cluster_free_cursor[target_c + delta]++;
                    break;
                }
                if (target_c >= delta &&
                    cluster_free_cursor[target_c - delta] < cluster_end_id[target_c - delta]) {
                    slot = cluster_free_cursor[target_c - delta]++;
                    break;
                }
            }
        }

        // Tail fallback if all pre-allocated cluster gaps are full
        if (slot < 0) {
            slot = static_cast<storage_idx_t>(ntotal);
            ntotal++;
            flat_codes->ntotal = ntotal;
            flat_codes->codes.resize(static_cast<size_t>(ntotal) * cs, 0);
            acorn.levels.push_back(1);
            acorn.offsets.push_back(acorn.offsets.back());
            acorn.compressed_l0_counts.push_back(0);
            acorn.compressed_l0_offsets.push_back(acorn.compressed_l0_offsets.back());
            if (!acorn.compressed_l0_prefix_counts.empty()) {
                acorn.compressed_l0_prefix_counts.push_back(0);
            }
            acorn.has_overflow.push_back(0);
            new_to_old.push_back(-1);
            if (!reordered_metadata.empty()) {
                reordered_metadata.push_back(-1);
                acorn.metadata = reordered_metadata.data();
            }
            vt = VisitedTable(static_cast<size_t>(ntotal));
            // Recreate DistanceComputer since flat_codes->codes may have reallocated
            delete dis;
            dis = storage_distance_computer(storage);
            del.ptr = dis;
        }

        // Write new vector & bidirectional ID maps
        std::memcpy(flat_codes->codes.data() + static_cast<size_t>(slot) * cs, vec, cs);
        storage_idx_t ext_id = static_cast<storage_idx_t>(dynamic_external_count++);
        if (static_cast<size_t>(ext_id) >= old_to_new.size()) {
            old_to_new.resize(static_cast<size_t>(ext_id) + 1, -1);
        }
        old_to_new[ext_id] = slot;
        new_to_old[slot] = ext_id;
        if (new_metadata != nullptr && !reordered_metadata.empty()) {
            reordered_metadata[slot] = new_metadata[i];
        }

        // Store forward links & bidirectional back-edges in Delta-Overflow Buffer
        auto& fwd = acorn.overflow_l0[slot];
        fwd.reserve(k_link);
        acorn.has_overflow[slot] = 1;
        for (int j = 0; j < k_link; j++) {
            storage_idx_t nb_id = static_cast<storage_idx_t>(I_tmp[j]);
            if (nb_id >= 0 && nb_id != slot) {
                fwd.push_back(nb_id);
                acorn.overflow_l0[nb_id].push_back(slot);
                acorn.has_overflow[nb_id] = 1;
            }
        }
    }

    return local_cluster_hits;
}

void IndexACORN::compact_dynamic_l0(int proximity_prefix) {
    if (!acorn.use_compressed_l0 || acorn.overflow_l0.empty()) {
        return;
    }

    const size_t n = static_cast<size_t>(ntotal);
    const size_t split_limit = (acorn.gamma > 1 && acorn.M_beta > 0)
            ? static_cast<size_t>(acorn.M_beta)
            : (proximity_prefix > 0 ? static_cast<size_t>(proximity_prefix) : 0);
    const bool use_two_segments = (split_limit > 0);

    std::vector<uint8_t> new_payload;
    new_payload.reserve(acorn.compressed_l0_neighbors.size() + acorn.overflow_l0.size() * 64);
    std::vector<size_t> new_offsets(n + 1, 0);
    std::vector<uint16_t> new_counts(n, 0);
    std::vector<uint16_t> new_prefix;
    if (use_two_segments) {
        new_prefix.assign(n, 0);
    }

    int32_t dec_buf[1024];
    std::vector<int32_t> merged;
    merged.reserve(256);

    for (size_t i = 0; i < n; i++) {
        size_t cnt = acorn.decode_l0_neighbors(static_cast<idx_t>(i), dec_buf);
        merged.assign(dec_buf, dec_buf + cnt);
        if (cnt > 1) {
            if (!use_two_segments) {
                std::sort(merged.begin(), merged.end());
                merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            } else {
                size_t split = std::min(merged.size(), split_limit);
                std::sort(merged.begin(), merged.begin() + split);
                auto it1 = std::unique(merged.begin(), merged.begin() + split);
                size_t unique_prefix = static_cast<size_t>(it1 - merged.begin());
                std::vector<int32_t> tail(merged.begin() + split, merged.end());
                std::sort(tail.begin(), tail.end());
                tail.erase(std::unique(tail.begin(), tail.end()), tail.end());
                merged.resize(unique_prefix);
                for (int32_t v : tail) {
                    if (!std::binary_search(merged.begin(), merged.begin() + unique_prefix, v)) {
                        merged.push_back(v);
                    }
                }
                new_prefix[i] = static_cast<uint16_t>(unique_prefix);
            }
        } else if (use_two_segments) {
            new_prefix[i] = static_cast<uint16_t>(merged.size());
        }

        new_counts[i] = static_cast<uint16_t>(merged.size());
        if (!merged.empty()) {
            if (!use_two_segments) {
                codec::SimdBP128Codec::encode_delta_list(
                        merged.data(), merged.size(), new_payload);
            } else {
                size_t split = new_prefix[i];
                codec::SimdBP128Codec::encode_delta_list(merged.data(), split, new_payload);
                if (merged.size() > split) {
                    codec::SimdBP128Codec::encode_delta_list(
                            merged.data() + split, merged.size() - split, new_payload);
                }
            }
        }
        new_offsets[i + 1] = new_payload.size();
    }

    new_payload.resize(new_payload.size() + codec::SimdBP128Codec::PADDING_BYTES, 0);
    new_payload.shrink_to_fit();

    acorn.compressed_l0_neighbors.swap(new_payload);
    acorn.compressed_l0_offsets.swap(new_offsets);
    acorn.compressed_l0_counts.swap(new_counts);
    acorn.compressed_l0_prefix_counts.swap(new_prefix);
    acorn.l0_compression_mode = ACORN::L0CompressionMode::SIMD_BP128;

    // Clear Delta-Overflow Buffer cleanly
    acorn.overflow_l0.clear();
    std::fill(acorn.has_overflow.begin(), acorn.has_overflow.end(), 0);
}

namespace {

template <typename T>
void acorn_write_pod(std::ofstream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
void acorn_write_vec(std::ofstream& out, const std::vector<T>& vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    acorn_write_pod(out, sz);
    if (sz > 0) {
        out.write(
                reinterpret_cast<const char*>(vec.data()),
                static_cast<std::streamsize>(sz * sizeof(T)));
    }
}

template <typename T>
void acorn_read_pod(std::ifstream& in, T& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
}

template <typename T>
void acorn_read_vec(std::ifstream& in, std::vector<T>& vec) {
    uint64_t sz = 0;
    acorn_read_pod(in, sz);
    vec.resize(static_cast<size_t>(sz));
    if (sz > 0) {
        in.read(
                reinterpret_cast<char*>(vec.data()),
                static_cast<std::streamsize>(sz * sizeof(T)));
    }
}

} // namespace

void IndexACORN::save_to_file(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    FAISS_THROW_IF_NOT_FMT(out.is_open(), "Cannot open %s for writing", path.c_str());

    const char magic[8] = {'A', 'C', 'R', 'N', 'S', 'M', '0', '1'};
    out.write(magic, 8);

    // Index base metadata
    int32_t d_i32 = d;
    int64_t ntotal_i64 = ntotal;
    int32_t metric_i32 = static_cast<int32_t>(metric_type);
    uint8_t trained_u8 = is_trained ? 1 : 0;
    acorn_write_pod(out, d_i32);
    acorn_write_pod(out, ntotal_i64);
    acorn_write_pod(out, metric_i32);
    acorn_write_pod(out, trained_u8);

    // Underlying flat vector storage
    const auto* flat_codes = dynamic_cast<const IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_codes != nullptr,
            "save_to_file currently requires IndexFlatCodes-backed storage");
    uint64_t code_size_u64 = static_cast<uint64_t>(flat_codes->code_size);
    acorn_write_pod(out, code_size_u64);
    acorn_write_vec(out, flat_codes->codes);

    // ACORN scalar parameters and state
    acorn_write_pod(out, acorn.M);
    acorn_write_pod(out, acorn.gamma);
    acorn_write_pod(out, acorn.M_beta);
    acorn_write_pod(out, acorn.entry_point);
    acorn_write_pod(out, acorn.max_level);
    acorn_write_pod(out, acorn.efConstruction);
    acorn_write_pod(out, acorn.efSearch);
    uint8_t use_comp = acorn.use_compressed_l0 ? 1 : 0;
    uint8_t l0_freed_u8 = acorn.l0_freed ? 1 : 0;
    uint8_t mode_u8 = static_cast<uint8_t>(acorn.l0_compression_mode);
    acorn_write_pod(out, use_comp);
    acorn_write_pod(out, l0_freed_u8);
    acorn_write_pod(out, mode_u8);

    // ACORN graph & compressed L0 vectors
    acorn_write_vec(out, acorn.assign_probas);
    acorn_write_vec(out, acorn.cum_nneighbor_per_level);
    acorn_write_vec(out, acorn.levels);
    acorn_write_vec(out, acorn.nb_per_level);
    acorn_write_vec(out, acorn.offsets);
    acorn_write_vec(out, acorn.neighbors);
    acorn_write_vec(out, acorn.compressed_l0_neighbors);
    acorn_write_vec(out, acorn.compressed_l0_offsets);
    acorn_write_vec(out, acorn.compressed_l0_counts);
    acorn_write_vec(out, acorn.compressed_l0_prefix_counts);

    // Reordering permutation maps & metadata
    uint8_t reord_u8 = is_reordered ? 1 : 0;
    acorn_write_pod(out, reord_u8);
    acorn_write_vec(out, old_to_new);
    acorn_write_vec(out, new_to_old);
    acorn_write_vec(out, reordered_metadata);

    FAISS_THROW_IF_NOT_FMT(out.good(), "I/O error while writing %s", path.c_str());
}

void IndexACORN::load_from_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    FAISS_THROW_IF_NOT_FMT(in.is_open(), "Cannot open %s for reading", path.c_str());

    char magic[8] = {0};
    in.read(magic, 8);
    FAISS_THROW_IF_NOT_MSG(
            std::memcmp(magic, "ACRNSM01", 8) == 0,
            "Invalid or unsupported ACORN serialized file magic");

    int32_t d_i32 = 0;
    int64_t ntotal_i64 = 0;
    int32_t metric_i32 = 0;
    uint8_t trained_u8 = 0;
    acorn_read_pod(in, d_i32);
    acorn_read_pod(in, ntotal_i64);
    acorn_read_pod(in, metric_i32);
    acorn_read_pod(in, trained_u8);

    d = d_i32;
    ntotal = static_cast<idx_t>(ntotal_i64);
    metric_type = static_cast<MetricType>(metric_i32);
    is_trained = (trained_u8 != 0);

    auto* flat_codes = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_codes != nullptr,
            "load_from_file currently requires IndexFlatCodes-backed storage");
    uint64_t code_size_u64 = 0;
    acorn_read_pod(in, code_size_u64);
    flat_codes->d = d;
    flat_codes->ntotal = ntotal;
    flat_codes->code_size = static_cast<size_t>(code_size_u64);
    acorn_read_vec(in, flat_codes->codes);

    acorn_read_pod(in, acorn.M);
    acorn_read_pod(in, acorn.gamma);
    acorn_read_pod(in, acorn.M_beta);
    acorn_read_pod(in, acorn.entry_point);
    acorn_read_pod(in, acorn.max_level);
    acorn_read_pod(in, acorn.efConstruction);
    acorn_read_pod(in, acorn.efSearch);
    uint8_t use_comp = 0, l0_freed_u8 = 0, mode_u8 = 0;
    acorn_read_pod(in, use_comp);
    acorn_read_pod(in, l0_freed_u8);
    acorn_read_pod(in, mode_u8);
    acorn.use_compressed_l0 = (use_comp != 0);
    acorn.l0_freed = (l0_freed_u8 != 0);
    acorn.l0_compression_mode = static_cast<ACORN::L0CompressionMode>(mode_u8);

    acorn_read_vec(in, acorn.assign_probas);
    acorn_read_vec(in, acorn.cum_nneighbor_per_level);
    acorn_read_vec(in, acorn.levels);
    acorn_read_vec(in, acorn.nb_per_level);
    acorn_read_vec(in, acorn.offsets);
    acorn_read_vec(in, acorn.neighbors);
    acorn_read_vec(in, acorn.compressed_l0_neighbors);
    acorn_read_vec(in, acorn.compressed_l0_offsets);
    acorn_read_vec(in, acorn.compressed_l0_counts);
    acorn_read_vec(in, acorn.compressed_l0_prefix_counts);

    uint8_t reord_u8 = 0;
    acorn_read_pod(in, reord_u8);
    is_reordered = (reord_u8 != 0);
    acorn_read_vec(in, old_to_new);
    acorn_read_vec(in, new_to_old);
    acorn_read_vec(in, reordered_metadata);
    if (!reordered_metadata.empty()) {
        acorn.metadata = reordered_metadata.data();
    }

    acorn.has_overflow.clear();
    acorn.overflow_l0.clear();

    FAISS_THROW_IF_NOT_FMT(in.good(), "I/O error while reading %s", path.c_str());
}





// added for debugging TODO
void IndexACORN::printStats(bool print_edge_list, bool print_filtered_edge_lists, int filter,  Operation op) {
    acorn.print_neighbor_stats(print_edge_list, print_filtered_edge_lists, filter, op);
    printf("METADATA VEC for number nodes per level\n");
    for (int i = 0; i < acorn.nb_per_level.size(); i++) {
        printf("\tlevel %d: %d nodes\n", i, acorn.nb_per_level[i]);
    }
}



/**************************************************************
 * IndexACORNFlat implementation
 **************************************************************/




IndexACORNFlat::IndexACORNFlat(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric)
        : IndexACORN(new IndexFlat(d, metric), M, gamma, metadata, M_beta) {
    own_fields = true;
    is_trained = true;
}





} // namespace faiss
