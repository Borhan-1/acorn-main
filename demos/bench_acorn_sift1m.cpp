#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <omp.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/codec/vbyte_codec.h>
#include <faiss/utils/distances.h>

namespace {

struct BenchConfig {
    size_t N = 100000;
    size_t nq = 1000;
    int M = 32;
    int gamma = 1;
    int M_beta = 64;
    int k = 10;
    double selectivity = 1.0;
    int threads = 0;
    bool reorder = false;
    bool ab_validate = false;
    bool no_reorder_compress = false;
    bool profile_decode = false;
    bool bench_hnsw = false;
    bool bench_prefilter = false;
    bool bench_prefilter_only = false;
    bool bench_predicate_suite = false;
    bool mt_qps = false;
    bool bench_dynamic = false;
    double gap_ratio = 0.05;
    double insert_pct = 0.05;
    std::string predicate_mode = "uniform"; // uniform | real_range | real_compound | zipf
    int min_len = 16;
    int max_len = 999;
    size_t max_rank = 20000;
    double zipf_alpha = 1.2;
    int target_bucket = 99;
    std::string codec = "vbyte"; // vbyte | simd_bp128 | both
    std::string reorder_mode = "bfs"; // bfs | hierarchical
    size_t bfs_budget = 128;
    int proximity_prefix = 16;
    int runs = 1;
    std::vector<int> sweep_prefix;
    std::string data_dir = "Datasets/sift1M";
    std::string save_index_path;
    std::string load_index_path;
    std::vector<int> ef_list = {16, 32, 64};
};

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stoi(item));
        }
    }
    return out;
}

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_val = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--n") {
            cfg.N = std::stoull(next_val(arg));
        } else if (arg == "--nq") {
            cfg.nq = std::stoull(next_val(arg));
        } else if (arg == "--M") {
            cfg.M = std::stoi(next_val(arg));
        } else if (arg == "--gamma") {
            cfg.gamma = std::stoi(next_val(arg));
        } else if (arg == "--M_beta") {
            cfg.M_beta = std::stoi(next_val(arg));
        } else if (arg == "--k") {
            cfg.k = std::stoi(next_val(arg));
        } else if (arg == "--selectivity") {
            cfg.selectivity = std::stod(next_val(arg));
        } else if (arg == "--efSearch") {
            cfg.ef_list = parse_int_list(next_val(arg));
        } else if (arg == "--threads") {
            cfg.threads = std::stoi(next_val(arg));
        } else if (arg == "--reorder") {
            cfg.reorder = true;
        } else if (arg == "--ab_validate") {
            cfg.ab_validate = true;
        } else if (arg == "--no_reorder_compress") {
            cfg.no_reorder_compress = true;
        } else if (arg == "--profile_decode") {
            cfg.profile_decode = true;
        } else if (arg == "--bench_hnsw") {
            cfg.bench_hnsw = true;
        } else if (arg == "--bench_prefilter") {
            cfg.bench_prefilter = true;
        } else if (arg == "--bench_prefilter_only") {
            cfg.bench_prefilter_only = true;
        } else if (arg == "--bench_predicate_suite") {
            cfg.bench_predicate_suite = true;
        } else if (arg == "--predicate_mode") {
            cfg.predicate_mode = next_val(arg);
        } else if (arg == "--min_len") {
            cfg.min_len = std::stoi(next_val(arg));
        } else if (arg == "--max_len") {
            cfg.max_len = std::stoi(next_val(arg));
        } else if (arg == "--max_rank") {
            cfg.max_rank = std::stoull(next_val(arg));
        } else if (arg == "--zipf_alpha") {
            cfg.zipf_alpha = std::stod(next_val(arg));
        } else if (arg == "--target_bucket") {
            cfg.target_bucket = std::stoi(next_val(arg));
        } else if (arg == "--mt_qps") {
            cfg.mt_qps = true;
        } else if (arg == "--bench_dynamic") {
            cfg.bench_dynamic = true;
        } else if (arg == "--gap_ratio") {
            cfg.gap_ratio = std::stod(next_val(arg));
        } else if (arg == "--insert_pct") {
            cfg.insert_pct = std::stod(next_val(arg));
        } else if (arg == "--codec") {
            cfg.codec = next_val(arg);
        } else if (arg == "--reorder_mode") {
            cfg.reorder_mode = next_val(arg);
        } else if (arg == "--bfs_budget") {
            cfg.bfs_budget = std::stoull(next_val(arg));
        } else if (arg == "--proximity_prefix") {
            cfg.proximity_prefix = std::stoi(next_val(arg));
        } else if (arg == "--runs") {
            cfg.runs = std::max(1, std::stoi(next_val(arg)));
        } else if (arg == "--sweep_prefix") {
            cfg.sweep_prefix = parse_int_list(next_val(arg));
            cfg.ab_validate = true;
        } else if (arg == "--data_dir") {
            cfg.data_dir = next_val(arg);
        } else if (arg == "--save_index") {
            cfg.save_index_path = next_val(arg);
        } else if (arg == "--load_index") {
            cfg.load_index_path = next_val(arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench_acorn_sift1m [options]\n"
                      << "  --n <N>                 Base vectors (default: 100000, max: 1000000)\n"
                      << "  --nq <NQ>               Query vectors (default: 1000, max: 10000)\n"
                      << "  --M <M>                 ACORN M parameter (default: 32)\n"
                      << "  --gamma <G>             ACORN gamma parameter (default: 1)\n"
                      << "  --M_beta <MB>           ACORN M_beta parameter (default: 64)\n"
                      << "  --k <K>                 Top-k recall (default: 10)\n"
                      << "  --selectivity <S>       Filter selectivity in (0, 1] (default: 1.0)\n"
                      << "  --efSearch <a,b,c>      Comma-separated efSearch values (default: 16,32,64)\n"
                      << "  --threads <T>           OpenMP thread count (default: 0 = auto)\n"
                      << "  --reorder               Run BFS graph reordering + delta histogram analysis\n"
                      << "  --ab_validate           Same-index A/B test: Vanilla -> Reordered -> Compressed\n"
                      << "  --codec <MODE>          vbyte | simd_bp128 | both (default: vbyte)\n"
                      << "  --proximity_prefix <P>  Prefix split for SIMD-BP128 (default: 16, 0=off)\n"
                      << "  --no_reorder_compress   Ablation: compress without BFS reordering\n"
                      << "  --data_dir <DIR>        SIFT1M directory (default: Datasets/sift1M)\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }
    return cfg;
}

// Read at most max_n vectors from a .fvecs file.
std::vector<float> read_fvecs(const std::string& path, size_t max_n, size_t& d_out, size_t& n_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open fvecs file: " << path << "\n";
        std::exit(1);
    }

    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (!in || dim <= 0) {
        std::cerr << "Invalid fvecs header in: " << path << "\n";
        std::exit(1);
    }

    in.seekg(0, std::ios::end);
    size_t file_bytes = static_cast<size_t>(in.tellg());
    size_t row_bytes = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(float);
    size_t total_vecs = file_bytes / row_bytes;
    size_t n = std::min(max_n, total_vecs);

    d_out = static_cast<size_t>(dim);
    n_out = n;

    std::vector<float> data(n * d_out);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; ++i) {
        int32_t cur_dim = 0;
        in.read(reinterpret_cast<char*>(&cur_dim), sizeof(int32_t));
        if (cur_dim != dim) {
            std::cerr << "Dimension mismatch at vector " << i << " in " << path << "\n";
            std::exit(1);
        }
        in.read(reinterpret_cast<char*>(data.data() + i * d_out),
                static_cast<std::streamsize>(d_out * sizeof(float)));
    }
    return data;
}

// Read at most max_n groundtruth rows from a .ivecs file.
std::vector<int32_t> read_ivecs(const std::string& path, size_t max_n, size_t& k_out, size_t& n_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open ivecs file: " << path << "\n";
        std::exit(1);
    }

    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (!in || dim <= 0) {
        std::cerr << "Invalid ivecs header in: " << path << "\n";
        std::exit(1);
    }

    in.seekg(0, std::ios::end);
    size_t file_bytes = static_cast<size_t>(in.tellg());
    size_t row_bytes = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(int32_t);
    size_t total_rows = file_bytes / row_bytes;
    size_t n = std::min(max_n, total_rows);

    k_out = static_cast<size_t>(dim);
    n_out = n;

    std::vector<int32_t> data(n * k_out);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; ++i) {
        int32_t cur_dim = 0;
        in.read(reinterpret_cast<char*>(&cur_dim), sizeof(int32_t));
        if (cur_dim != dim) {
            std::cerr << "Dimension mismatch at row " << i << " in " << path << "\n";
            std::exit(1);
        }
        in.read(reinterpret_cast<char*>(data.data() + i * k_out),
                static_cast<std::streamsize>(k_out * sizeof(int32_t)));
    }
    return data;
}

// Compute exact top-k L2 groundtruth over vectors passing filter_map[i] != 0.
std::vector<int32_t> compute_exact_gt_bitmap(
        const float* xb,
        size_t nb,
        const float* xq,
        size_t nq,
        size_t d,
        int k,
        const std::vector<char>& filter_map,
        bool is_filtered) {
    std::vector<size_t> valid_ids;
    valid_ids.reserve(nb);
    for (size_t i = 0; i < nb; ++i) {
        if (!is_filtered || filter_map[i] != 0) {
            valid_ids.push_back(i);
        }
    }

    std::vector<int32_t> gt(nq * static_cast<size_t>(k), -1);

#pragma omp parallel for schedule(static)
    for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
        const float* q = xq + static_cast<size_t>(qi) * d;
        std::vector<std::pair<float, int32_t>> best;
        best.reserve(static_cast<size_t>(k) + 1);

        for (size_t idx : valid_ids) {
            const float* v = xb + idx * d;
            float dist = faiss::fvec_L2sqr(q, v, d);
            if (static_cast<int>(best.size()) < k) {
                best.emplace_back(dist, static_cast<int32_t>(idx));
                if (static_cast<int>(best.size()) == k) {
                    std::make_heap(best.begin(), best.end());
                }
            } else if (dist < best.front().first) {
                std::pop_heap(best.begin(), best.end());
                best.back() = {dist, static_cast<int32_t>(idx)};
                std::push_heap(best.begin(), best.end());
            }
        }

        std::sort(best.begin(), best.end());
        for (size_t j = 0; j < best.size(); ++j) {
            gt[static_cast<size_t>(qi) * static_cast<size_t>(k) + j] = best[j].second;
        }
    }
    return gt;
}

double compute_recall_at_k(
        const faiss::idx_t* retrieved,
        const int32_t* gt,
        size_t nq,
        size_t k,
        size_t gt_stride) {
    size_t hits = 0;
    for (size_t q = 0; q < nq; ++q) {
        std::unordered_set<int32_t> gt_set;
        gt_set.reserve(k * 2);
        for (size_t j = 0; j < k; ++j) {
            int32_t gid = gt[q * gt_stride + j];
            if (gid >= 0) {
                gt_set.insert(gid);
            }
        }
        for (size_t j = 0; j < k; ++j) {
            faiss::idx_t rid = retrieved[q * k + j];
            if (rid >= 0 && gt_set.count(static_cast<int32_t>(rid))) {
                ++hits;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(nq * k);
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = (p / 100.0) * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double get_process_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

struct GraphMemBreakdown {
    double neighbors_mb = 0.0;
    double offsets_levels_mb = 0.0;
    double vbyte_payload_mb = 0.0;
    double vbyte_meta_mb = 0.0;
    double id_map_mb = 0.0;
    double total_graph_mb = 0.0;
};

GraphMemBreakdown measure_graph_memory(const faiss::IndexACORN& index) {
    constexpr double MB = 1024.0 * 1024.0;
    GraphMemBreakdown m;
    m.neighbors_mb = (index.acorn.neighbors.size() * sizeof(faiss::ACORN::NeighNode)) / MB;
    m.offsets_levels_mb =
            (index.acorn.offsets.size() * sizeof(size_t) +
             index.acorn.levels.size() * sizeof(int)) /
            MB;
    m.vbyte_payload_mb = (index.acorn.compressed_l0_neighbors.size() * sizeof(uint8_t)) / MB;
    m.vbyte_meta_mb =
            (index.acorn.compressed_l0_offsets.size() * sizeof(size_t) +
             index.acorn.compressed_l0_counts.size() * sizeof(uint16_t) +
             index.acorn.compressed_l0_prefix_counts.size() * sizeof(uint16_t)) /
            MB;
    m.id_map_mb =
            (index.old_to_new.size() * sizeof(faiss::ACORN::storage_idx_t) +
             index.new_to_old.size() * sizeof(faiss::ACORN::storage_idx_t)) /
            MB;
    m.total_graph_mb = m.neighbors_mb + m.offsets_levels_mb + m.vbyte_payload_mb + m.vbyte_meta_mb;
    return m;
}

static int g_bench_runs = 1;

static double student_t_975(int df) {
    static const double tbl[] = {
            1.96, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228};
    if (df <= 0) return 0.0;
    if (df <= 10) return tbl[df];
    return 1.96;
}

struct SearchRunResult {
    int efs = 0;
    double recall = 0.0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p50_ci95_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double qps_1t = 0.0;
    double avg_ndis = 0.0;
    double avg_decode_calls = 0.0;
    double avg_decoded_edges = 0.0;
    double avg_decode_us = 0.0;
};

std::vector<SearchRunResult> run_search_suite(
        faiss::IndexACORNFlat& index,
        const float* xq,
        size_t nq,
        size_t d_base,
        int k,
        bool is_filtered,
        std::vector<char>& active_filter_map,
        bool assume_reordered_filter,
        bool profile_decode,
        const int32_t* gt,
        size_t gt_stride,
        const std::vector<int>& ef_list,
        const char* title) {
    std::cout << "\n--- " << title << " (Recall@" << k << ") ---\n";
    if (profile_decode) {
        std::cout << "| efSearch | Recall@" << k
                  << " | Mean (ms) | p50 (ms) | 1T QPS | Avg ndis | DecCalls/q | DecEdges/q | DecTime (us) | Dec % |\n";
        std::cout << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    } else {
        std::cout << "| efSearch | Recall@" << k
                  << " | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | 1T QPS | Avg ndis |\n";
        std::cout << "|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    }

    std::vector<float> D(nq * static_cast<size_t>(k));
    std::vector<faiss::idx_t> I(nq * static_cast<size_t>(k));
    std::vector<double> lat_ms(nq);
    std::vector<SearchRunResult> results;
    results.reserve(ef_list.size());

    for (int efs : ef_list) {
        faiss::SearchParametersACORN params;
        params.efSearch = efs;
        params.assume_reordered_filter = assume_reordered_filter;
        params.enable_decode_timing = profile_decode;
        index.acorn.efSearch = efs;

        size_t warmup = std::min<size_t>(10, nq);
        for (size_t q = 0; q < warmup; ++q) {
            if (is_filtered) {
                index.search(
                        1,
                        xq + q * d_base,
                        k,
                        D.data() + q * k,
                        I.data() + q * k,
                        active_filter_map.data(),
                        &params);
            } else {
                index.search(1, xq + q * d_base, k, D.data() + q * k, I.data() + q * k, &params);
            }
        }

        const int num_runs = std::max(1, g_bench_runs);
        std::vector<double> run_p50(num_runs), run_mean(num_runs), run_p95(num_runs),
                run_p99(num_runs), run_qps(num_runs);

        SearchRunResult r;
        r.efs = efs;
        for (int ri = 0; ri < num_runs; ++ri) {
            faiss::acorn_stats.reset();
            auto t_all0 = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < nq; ++q) {
                auto tq0 = std::chrono::high_resolution_clock::now();
                if (is_filtered) {
                    index.search(
                            1,
                            xq + q * d_base,
                            k,
                            D.data() + q * k,
                            I.data() + q * k,
                            active_filter_map.data(),
                            &params);
                } else {
                    index.search(1, xq + q * d_base, k, D.data() + q * k, I.data() + q * k, &params);
                }
                auto tq1 = std::chrono::high_resolution_clock::now();
                lat_ms[q] = std::chrono::duration<double, std::milli>(tq1 - tq0).count();
            }
            auto t_all1 = std::chrono::high_resolution_clock::now();
            double total_s = std::chrono::duration<double>(t_all1 - t_all0).count();
            run_mean[ri] = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) / static_cast<double>(nq);
            run_p50[ri] = percentile(lat_ms, 50.0);
            run_p95[ri] = percentile(lat_ms, 95.0);
            run_p99[ri] = percentile(lat_ms, 99.0);
            run_qps[ri] = static_cast<double>(nq) / total_s;
            if (ri == 0) {
                r.recall = compute_recall_at_k(I.data(), gt, nq, static_cast<size_t>(k), gt_stride);
                r.avg_ndis = static_cast<double>(faiss::acorn_stats.n3) / static_cast<double>(nq);
                r.avg_decode_calls = static_cast<double>(faiss::acorn_stats.decode_calls) / static_cast<double>(nq);
                r.avg_decoded_edges = static_cast<double>(faiss::acorn_stats.decoded_edges) / static_cast<double>(nq);
                r.avg_decode_us = faiss::acorn_stats.decode_time_us / static_cast<double>(nq);
            }
        }

        auto vec_median = [](std::vector<double> v) -> double {
            std::sort(v.begin(), v.end());
            size_t m = v.size();
            return (m % 2 == 1) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
        };
        r.mean_ms = vec_median(run_mean);
        r.p50_ms = vec_median(run_p50);
        r.p95_ms = vec_median(run_p95);
        r.p99_ms = vec_median(run_p99);
        r.qps_1t = vec_median(run_qps);
        if (num_runs > 1) {
            double avg = std::accumulate(run_p50.begin(), run_p50.end(), 0.0) / num_runs;
            double var = 0.0;
            for (double x : run_p50) var += (x - avg) * (x - avg);
            double sd = std::sqrt(var / static_cast<double>(num_runs - 1));
            r.p50_ci95_ms = student_t_975(num_runs - 1) * sd / std::sqrt(static_cast<double>(num_runs));
        }
        results.push_back(r);

        if (profile_decode) {
            double dec_pct = (r.mean_ms > 0.0) ? (100.0 * (r.avg_decode_us / 1000.0) / r.mean_ms) : 0.0;
            std::cout << "| " << r.efs
                      << " | " << std::fixed << std::setprecision(4) << r.recall
                      << " | " << std::setprecision(3) << r.mean_ms
                      << " | " << std::setprecision(3) << r.p50_ms
                      << " | " << std::setprecision(1) << r.qps_1t
                      << " | " << std::setprecision(1) << r.avg_ndis
                      << " | " << std::setprecision(1) << r.avg_decode_calls
                      << " | " << std::setprecision(0) << r.avg_decoded_edges
                      << " | " << std::setprecision(1) << r.avg_decode_us
                      << " | " << std::setprecision(1) << dec_pct << "%"
                      << " |\n";
        } else {
            std::cout << "| " << r.efs
                      << " | " << std::fixed << std::setprecision(4) << r.recall
                      << " | " << std::setprecision(3) << r.mean_ms
                      << " | " << std::setprecision(3) << r.p50_ms;
            if (g_bench_runs > 1) {
                std::cout << " +/- " << std::setprecision(3) << r.p50_ci95_ms;
            }
            std::cout << " | " << std::setprecision(3) << r.p95_ms
                      << " | " << std::setprecision(3) << r.p99_ms
                      << " | " << std::setprecision(1) << r.qps_1t
                      << " | " << std::setprecision(1) << r.avg_ndis
                      << " |\n";
        }
    }
    return results;
}

void run_mt_qps_suite(
        faiss::IndexACORNFlat& index,
        const float* xq,
        size_t nq,
        size_t d_base,
        int k,
        bool is_filtered,
        std::vector<char>& active_filter_map,
        bool assume_reordered_filter,
        int efs,
        const char* label) {
    std::vector<int> thread_counts = {1, 2, 4};
    std::vector<float> D(nq * static_cast<size_t>(k));
    std::vector<faiss::idx_t> I(nq * static_cast<size_t>(k));

    faiss::SearchParametersACORN params;
    params.efSearch = efs;
    params.assume_reordered_filter = assume_reordered_filter;
    index.acorn.efSearch = efs;

    int prev_threads = omp_get_max_threads();
    std::cout << "| " << label << " (ef=" << efs << ")";
    double qps_1t = 0.0;
    for (int t : thread_counts) {
        omp_set_num_threads(t);
        auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 8) num_threads(t)
        for (int64_t q = 0; q < static_cast<int64_t>(nq); ++q) {
            if (is_filtered) {
                index.search(
                        1,
                        xq + static_cast<size_t>(q) * d_base,
                        k,
                        D.data() + static_cast<size_t>(q) * k,
                        I.data() + static_cast<size_t>(q) * k,
                        active_filter_map.data(),
                        &params);
            } else {
                index.search(
                        1,
                        xq + static_cast<size_t>(q) * d_base,
                        k,
                        D.data() + static_cast<size_t>(q) * k,
                        I.data() + static_cast<size_t>(q) * k,
                        &params);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double qps = static_cast<double>(nq) / sec;
        if (t == 1) qps_1t = qps;
        std::cout << " | " << std::fixed << std::setprecision(1) << qps;
    }
    omp_set_num_threads(prev_threads);
    std::cout << " |\n";
}

void run_postfilter_hnsw_baseline(
        const float* xb,
        size_t nb,
        const float* xq,
        size_t nq,
        size_t d_base,
        int M,
        int k,
        bool is_filtered,
        const std::vector<char>& filter_map,
        const int32_t* gt,
        size_t gt_stride,
        const std::vector<int>& ef_list) {
    constexpr double MB = 1024.0 * 1024.0;
    std::cout << "\n=== Post-Filter HNSW Baseline (faiss::IndexHNSWFlat, M=" << M << ") ===\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWFlat hnsw(static_cast<int>(d_base), M, faiss::METRIC_L2);
    hnsw.hnsw.efConstruction = 40;
    hnsw.add(static_cast<faiss::idx_t>(nb), xb);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();

    double graph_mb =
            (hnsw.hnsw.neighbors.size() * sizeof(faiss::HNSW::storage_idx_t) +
             hnsw.hnsw.offsets.size() * sizeof(size_t) +
             hnsw.hnsw.levels.size() * sizeof(int)) /
            MB;
    std::cout << "HNSW Build time: " << std::fixed << std::setprecision(2) << build_s
              << " s | HNSW Graph Memory: " << graph_mb << " MB\n";

    // Post-filter overfetch factor: retrieve k_fetch = max(efSearch, k) and keep top-k passing
    std::cout << "| efSearch | k_fetch | Recall@" << k << " | p50 (ms) | 1T QPS |\n";
    std::cout << "|---:|---:|---:|---:|---:|\n";

    for (int efs : ef_list) {
        int k_fetch = is_filtered ? std::max(efs, k) : k;
        faiss::SearchParametersHNSW params;
        params.efSearch = efs;
        hnsw.hnsw.efSearch = efs;

        std::vector<float> D_raw(static_cast<size_t>(k_fetch));
        std::vector<faiss::idx_t> I_raw(static_cast<size_t>(k_fetch));
        std::vector<faiss::idx_t> I_filt(nq * static_cast<size_t>(k), -1);
        std::vector<double> lat_ms(nq);

        auto ta0 = std::chrono::high_resolution_clock::now();
        for (size_t q = 0; q < nq; ++q) {
            auto tq0 = std::chrono::high_resolution_clock::now();
            hnsw.search(1, xq + q * d_base, k_fetch, D_raw.data(), I_raw.data(), &params);
            size_t found = 0;
            for (int j = 0; j < k_fetch && found < static_cast<size_t>(k); ++j) {
                faiss::idx_t id = I_raw[j];
                if (id >= 0 && (!is_filtered || filter_map[static_cast<size_t>(id)])) {
                    I_filt[q * static_cast<size_t>(k) + found++] = id;
                }
            }
            auto tq1 = std::chrono::high_resolution_clock::now();
            lat_ms[q] = std::chrono::duration<double, std::milli>(tq1 - tq0).count();
        }
        auto ta1 = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(ta1 - ta0).count();
        double recall = compute_recall_at_k(
                I_filt.data(), gt, nq, static_cast<size_t>(k), gt_stride);
        double p50 = percentile(lat_ms, 50.0);
        double qps = static_cast<double>(nq) / total_s;

        std::cout << "| " << efs << " | " << k_fetch
                  << " | " << std::setprecision(4) << recall
                  << " | " << std::setprecision(3) << p50
                  << " | " << std::setprecision(1) << qps << " |\n";
    }
}

struct PreFilterResult {
    double bitmap_p50_ms = 0.0;
    double bitmap_qps_1t = 0.0;
    double invlist_p50_ms = 0.0;
    double invlist_qps_1t = 0.0;
    double recall = 1.0;
};

PreFilterResult run_prefilter_baseline(
        const float* xb,
        size_t nb,
        const float* xq,
        size_t nq,
        size_t d_base,
        int k,
        const std::vector<char>& filter_map,
        const int32_t* gt,
        size_t gt_stride,
        const char* label) {
    std::vector<int32_t> valid_ids;
    valid_ids.reserve(nb / 8 + 16);
    for (size_t i = 0; i < nb; ++i) {
        if (filter_map[i] != 0) {
            valid_ids.push_back(static_cast<int32_t>(i));
        }
    }

    std::vector<faiss::idx_t> I_out(nq * static_cast<size_t>(k), -1);
    std::vector<double> lat_bmp(nq), lat_inv(nq);
    std::vector<std::pair<float, int32_t>> heap;
    heap.reserve(static_cast<size_t>(k) + 1);

    // 1. Bitmap-Scan Pre-Filter (FAISS IDSelectorBitmap + Flat L2 style)
    size_t warmup = std::min<size_t>(5, nq);
    for (size_t q = 0; q < warmup; ++q) {
        const float* qv = xq + q * d_base;
        heap.clear();
        for (size_t i = 0; i < nb; ++i) {
            if (!filter_map[i]) continue;
            float dist = faiss::fvec_L2sqr(qv, xb + i * d_base, d_base);
            if (static_cast<int>(heap.size()) < k) {
                heap.emplace_back(dist, static_cast<int32_t>(i));
                if (static_cast<int>(heap.size()) == k) std::make_heap(heap.begin(), heap.end());
            } else if (dist < heap.front().first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {dist, static_cast<int32_t>(i)};
                std::push_heap(heap.begin(), heap.end());
            }
        }
    }

    auto tb0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < nq; ++q) {
        auto tq0 = std::chrono::high_resolution_clock::now();
        const float* qv = xq + q * d_base;
        heap.clear();
        for (size_t i = 0; i < nb; ++i) {
            if (!filter_map[i]) continue;
            float dist = faiss::fvec_L2sqr(qv, xb + i * d_base, d_base);
            if (static_cast<int>(heap.size()) < k) {
                heap.emplace_back(dist, static_cast<int32_t>(i));
                if (static_cast<int>(heap.size()) == k) std::make_heap(heap.begin(), heap.end());
            } else if (dist < heap.front().first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {dist, static_cast<int32_t>(i)};
                std::push_heap(heap.begin(), heap.end());
            }
        }
        std::sort(heap.begin(), heap.end());
        for (size_t j = 0; j < heap.size(); ++j) {
            I_out[q * static_cast<size_t>(k) + j] = heap[j].second;
        }
        auto tq1 = std::chrono::high_resolution_clock::now();
        lat_bmp[q] = std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    }
    auto tb1 = std::chrono::high_resolution_clock::now();
    double bmp_sec = std::chrono::duration<double>(tb1 - tb0).count();

    // 2. Inverted-List Oracle Pre-Filter (zero predicate scan overhead, random gather + L2)
    auto ti0 = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < nq; ++q) {
        auto tq0 = std::chrono::high_resolution_clock::now();
        const float* qv = xq + q * d_base;
        heap.clear();
        for (int32_t id : valid_ids) {
            float dist = faiss::fvec_L2sqr(qv, xb + static_cast<size_t>(id) * d_base, d_base);
            if (static_cast<int>(heap.size()) < k) {
                heap.emplace_back(dist, id);
                if (static_cast<int>(heap.size()) == k) std::make_heap(heap.begin(), heap.end());
            } else if (dist < heap.front().first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = {dist, id};
                std::push_heap(heap.begin(), heap.end());
            }
        }
        std::sort(heap.begin(), heap.end());
        for (size_t j = 0; j < heap.size(); ++j) {
            I_out[q * static_cast<size_t>(k) + j] = heap[j].second;
        }
        auto tq1 = std::chrono::high_resolution_clock::now();
        lat_inv[q] = std::chrono::duration<double, std::milli>(tq1 - tq0).count();
    }
    auto ti1 = std::chrono::high_resolution_clock::now();
    double inv_sec = std::chrono::duration<double>(ti1 - ti0).count();

    PreFilterResult r;
    r.bitmap_p50_ms = percentile(lat_bmp, 50.0);
    r.bitmap_qps_1t = static_cast<double>(nq) / bmp_sec;
    r.invlist_p50_ms = percentile(lat_inv, 50.0);
    r.invlist_qps_1t = static_cast<double>(nq) / inv_sec;
    r.recall = compute_recall_at_k(I_out.data(), gt, nq, static_cast<size_t>(k), gt_stride);

    std::cout << "\n--- Pre-Filter Exact L2 Scan Baseline (" << label
              << ", passing=" << valid_ids.size() << ") ---\n";
    std::cout << "| Pre-Filter Mode | Recall@" << k << " | p50 (ms) | 1T QPS |\n";
    std::cout << "|---|---:|---:|---:|\n";
    std::cout << "| Bitmap-Scan + Exact L2 | " << std::fixed << std::setprecision(4) << r.recall
              << " | " << std::setprecision(3) << r.bitmap_p50_ms
              << " | " << std::setprecision(1) << r.bitmap_qps_1t << " |\n";
    std::cout << "| Inverted-List + Exact L2 | " << std::setprecision(4) << r.recall
              << " | " << std::setprecision(3) << r.invlist_p50_ms
              << " | " << std::setprecision(1) << r.invlist_qps_1t << " |\n";
    return r;
}

void analyze_level0_deltas(const faiss::IndexACORN& index, const char* label) {
    const size_t n = static_cast<size_t>(index.ntotal);
    size_t total_active = 0;
    size_t total_l0_slots = 0;
    size_t total_vbyte_bytes = 0;
    uint64_t sum_deltas = 0;
    size_t b1 = 0, b2 = 0, b3 = 0, b4p = 0;

    std::vector<int32_t> sorted_ids;
    std::vector<uint8_t> encoded;
    sorted_ids.reserve(256);
    encoded.reserve(512);

    for (size_t i = 0; i < n; ++i) {
        size_t begin = 0, end = 0;
        index.acorn.neighbor_range(static_cast<faiss::idx_t>(i), 0, &begin, &end);
        total_l0_slots += (end - begin);

        sorted_ids.clear();
        for (size_t j = begin; j < end; ++j) {
            int32_t v = index.acorn.neighbors[j];
            if (v < 0) break;
            sorted_ids.push_back(v);
        }
        if (sorted_ids.empty()) continue;

        std::sort(sorted_ids.begin(), sorted_ids.end());
        uint32_t prev = 0;
        for (size_t j = 0; j < sorted_ids.size(); ++j) {
            uint32_t curr = static_cast<uint32_t>(sorted_ids[j]);
            uint32_t delta = curr - prev;
            prev = curr;
            sum_deltas += delta;
            if (delta < 128u) {
                ++b1;
            } else if (delta < 16384u) {
                ++b2;
            } else if (delta < 2097152u) {
                ++b3;
            } else {
                ++b4p;
            }
        }
        total_active += sorted_ids.size();

        encoded.clear();
        faiss::codec::VByteCodec::encode_delta_list(sorted_ids.data(), sorted_ids.size(), encoded);
        total_vbyte_bytes += encoded.size();
    }

    constexpr double MB = 1024.0 * 1024.0;
    double raw_l0_mb = (total_l0_slots * sizeof(int32_t)) / MB;
    double vbyte_mb = total_vbyte_bytes / MB;
    double metadata_mb = (n * (sizeof(size_t) + sizeof(uint16_t))) / MB;
    double savings_vs_raw = 100.0 * (1.0 - (vbyte_mb + metadata_mb) / raw_l0_mb);
    double pct = total_active > 0 ? (100.0 / static_cast<double>(total_active)) : 0.0;

    std::cout << "\n--- Level-0 Delta & VByte Analysis (" << label << ") ---\n";
    std::cout << "Active L0 edges:     " << total_active << " / " << total_l0_slots << " slots\n";
    std::cout << "Mean delta:          " << std::fixed << std::setprecision(1)
              << (total_active > 0 ? static_cast<double>(sum_deltas) / total_active : 0.0) << "\n";
    std::cout << "Delta byte histogram:\n"
              << "  1-byte (<128):       " << std::setprecision(2) << (b1 * pct) << "% (" << b1 << ")\n"
              << "  2-byte (<16,384):    " << (b2 * pct) << "% (" << b2 << ")\n"
              << "  3-byte (<2,097,152): " << (b3 * pct) << "% (" << b3 << ")\n"
              << "  4+ bytes:            " << (b4p * pct) << "% (" << b4p << ")\n";
    std::cout << "Raw L0 neighbors[]:  " << raw_l0_mb << " MB (4.00 B/slot)\n";
    std::cout << "VByte L0 payload:    " << vbyte_mb << " MB ("
              << (total_active > 0 ? static_cast<double>(total_vbyte_bytes) / total_active : 0.0)
              << " B/active edge, "
              << (total_l0_slots > 0 ? static_cast<double>(total_vbyte_bytes) / total_l0_slots : 0.0)
              << " B/slot)\n";
    std::cout << "VByte + offsets/cnt: " << (vbyte_mb + metadata_mb) << " MB  -->  "
              << savings_vs_raw << "% savings vs Raw L0\n";
}

} // namespace

int main(int argc, char** argv) try {
    std::cout << std::unitbuf;
    BenchConfig cfg = parse_args(argc, argv);
    g_bench_runs = cfg.runs;

    if (cfg.threads > 0) {
        omp_set_num_threads(cfg.threads);
    }
    int active_threads = omp_get_max_threads();

    std::cout << "=== Vanilla ACORN SIFT1M Baseline Benchmark ===\n";
    std::cout << "Config: N=" << cfg.N
              << ", nq=" << cfg.nq
              << ", M=" << cfg.M
              << ", gamma=" << cfg.gamma
              << ", M_beta=" << cfg.M_beta
              << ", k=" << cfg.k
              << ", selectivity=" << cfg.selectivity
              << ", runs=" << cfg.runs
              << ", omp_threads=" << active_threads << "\n";

    // 1. Load SIFT1M base and query vectors
    size_t d_base = 0, nb = 0;
    size_t d_query = 0, nq = 0;
    auto t_load0 = std::chrono::high_resolution_clock::now();
    std::vector<float> xb = read_fvecs(cfg.data_dir + "/sift_base.fvecs", cfg.N, d_base, nb);
    std::vector<float> xq = read_fvecs(cfg.data_dir + "/sift_query.fvecs", cfg.nq, d_query, nq);
    auto t_load1 = std::chrono::high_resolution_clock::now();

    if (d_base != d_query) {
        std::cerr << "Base/query dimension mismatch: " << d_base << " vs " << d_query << "\n";
        return 1;
    }
    double load_s = std::chrono::duration<double>(t_load1 - t_load0).count();
    std::cout << "Loaded " << nb << " base vectors (d=" << d_base << ") and "
              << nq << " queries in " << std::fixed << std::setprecision(2) << load_s << " s\n";

    // 2. Generate or load metadata / predicate bitmap
    bool is_filtered = (cfg.selectivity < 0.9999) || (cfg.predicate_mode != "uniform");
    int num_classes = (cfg.selectivity < 0.9999)
            ? std::max(1, static_cast<int>(std::llround(1.0 / cfg.selectivity)))
            : 1;
    std::vector<int> metadata(nb, 0);
    std::vector<char> filter_map(nb, 1);
    size_t passing_count = 0;

    if (cfg.bench_predicate_suite) {
        size_t wl_dim = 0, wl_n = 0;
        std::vector<int32_t> word_lens =
                read_ivecs(cfg.data_dir + "/word_lengths.ivecs", nb, wl_dim, wl_n);
        if (wl_n < nb) {
            std::cerr << "word_lengths.ivecs has fewer rows than nb\n";
            return 1;
        }

        // Build Zipfian categorical distribution (50 classes, alpha = 1.1)
        const int zipf_classes = 50;
        const double alpha = 1.1;
        std::vector<double> cdf(zipf_classes, 0.0);
        double norm = 0.0;
        for (int c = 0; c < zipf_classes; ++c) {
            norm += 1.0 / std::pow(static_cast<double>(c + 1), alpha);
            cdf[c] = norm;
        }
        for (int c = 0; c < zipf_classes; ++c) cdf[c] /= norm;

        std::mt19937 zrng(42);
        std::uniform_real_distribution<double> udist(0.0, 1.0);
        std::vector<int> zipf_bucket(nb, 0);
        for (size_t i = 0; i < nb; ++i) {
            double u = udist(zrng);
            int b = static_cast<int>(
                    std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
            zipf_bucket[i] = std::min(b, zipf_classes - 1);
        }

        struct PredWorkload {
            std::string name;
            std::vector<char> fmap;
            std::vector<char> fmap_reord;
            std::vector<int32_t> gt;
            size_t passing = 0;
            PreFilterResult pre;
            std::vector<SearchRunResult> van;
            std::vector<SearchRunResult> vb;
            std::vector<SearchRunResult> simd;
        };

        std::vector<PredWorkload> preds(5);
        preds[0].name = "P1_RealRange_len>=16";
        preds[1].name = "P2_RealRange_len>=14";
        preds[2].name = "P3_Compound_len>=10_rank<20k";
        preds[3].name = "P4_ZipfTail_b49_a1.1";
        preds[4].name = "P5_ZipfHead_b0_a1.1";

        for (auto& p : preds) p.fmap.assign(nb, 0);
        for (size_t i = 0; i < nb; ++i) {
            int wl = word_lens[i * wl_dim];
            if (wl >= 16) preds[0].fmap[i] = 1;
            if (wl >= 14) preds[1].fmap[i] = 1;
            if (wl >= 10 && i < 20000) preds[2].fmap[i] = 1;
            if (zipf_bucket[i] == 49) preds[3].fmap[i] = 1;
            if (zipf_bucket[i] == 0) preds[4].fmap[i] = 1;
        }

        size_t gt_stride = static_cast<size_t>(cfg.k);
        std::cout << "\n=== Step 3.5A: Real Range, Compound & Zipfian Predicate Suite ===\n";
        for (auto& p : preds) {
            for (char c : p.fmap) p.passing += static_cast<size_t>(c != 0);
            p.gt = compute_exact_gt_bitmap(
                    xb.data(), nb, xq.data(), nq, d_base, cfg.k, p.fmap, true);
            p.pre = run_prefilter_baseline(
                    xb.data(), nb, xq.data(), nq, d_base, cfg.k,
                    p.fmap, p.gt.data(), gt_stride, p.name.c_str());
        }

        // Scoped Post-Filter HNSW Baseline across all 5 predicates (freed before ACORN build)
        {
            std::cout << "\n=== Post-Filter HNSW (M=" << cfg.M << ") Across All 5 Real/Skewed Predicates ===\n";
            faiss::IndexHNSWFlat hnsw(static_cast<int>(d_base), cfg.M, faiss::METRIC_L2);
            hnsw.hnsw.efConstruction = 40;
            hnsw.add(static_cast<faiss::idx_t>(nb), xb.data());
            for (const auto& p : preds) {
                std::cout << "\n[Post-Filter HNSW] " << p.name
                          << " (sel=" << std::setprecision(4)
                          << (static_cast<double>(p.passing) / nb) << ")\n";
                std::cout << "| efSearch | Recall@" << cfg.k << " | p50 (ms) | 1T QPS |\n|---:|---:|---:|---:|\n";
                for (int efs : cfg.ef_list) {
                    int k_fetch = std::max(efs, cfg.k);
                    faiss::SearchParametersHNSW hparams;
                    hparams.efSearch = efs;
                    hnsw.hnsw.efSearch = efs;
                    std::vector<float> D_raw(static_cast<size_t>(k_fetch));
                    std::vector<faiss::idx_t> I_raw(static_cast<size_t>(k_fetch));
                    std::vector<faiss::idx_t> I_filt(nq * static_cast<size_t>(cfg.k), -1);
                    std::vector<double> lat_ms(nq);
                    auto ta0 = std::chrono::high_resolution_clock::now();
                    for (size_t q = 0; q < nq; ++q) {
                        auto tq0 = std::chrono::high_resolution_clock::now();
                        hnsw.search(1, xq.data() + q * d_base, k_fetch, D_raw.data(), I_raw.data(), &hparams);
                        size_t found = 0;
                        for (int j = 0; j < k_fetch && found < static_cast<size_t>(cfg.k); ++j) {
                            faiss::idx_t id = I_raw[j];
                            if (id >= 0 && p.fmap[static_cast<size_t>(id)]) {
                                I_filt[q * static_cast<size_t>(cfg.k) + found++] = id;
                            }
                        }
                        auto tq1 = std::chrono::high_resolution_clock::now();
                        lat_ms[q] = std::chrono::duration<double, std::milli>(tq1 - tq0).count();
                    }
                    auto ta1 = std::chrono::high_resolution_clock::now();
                    double sec = std::chrono::duration<double>(ta1 - ta0).count();
                    double rec = compute_recall_at_k(I_filt.data(), p.gt.data(), nq, static_cast<size_t>(cfg.k), gt_stride);
                    std::cout << "| " << efs << " | " << std::setprecision(4) << rec
                              << " | " << std::setprecision(3) << percentile(lat_ms, 50.0)
                              << " | " << std::setprecision(1) << (nq / sec) << " |\n";
                }
            }
        }

        // Build single predicate-agnostic ACORN-1 index & evaluate Vanilla -> VByte -> AVX2 SIMD-BP128
        faiss::IndexACORNFlat acorn_idx(
                static_cast<int>(d_base), cfg.M, cfg.gamma, metadata, cfg.M_beta, faiss::METRIC_L2);
        acorn_idx.verbose = false;
        acorn_idx.add(static_cast<faiss::idx_t>(nb), xb.data());
        std::vector<float>().swap(xb);

        for (auto& p : preds) {
            p.van = run_search_suite(
                    acorn_idx, xq.data(), nq, d_base, cfg.k, true, p.fmap, false, false,
                    p.gt.data(), gt_stride, cfg.ef_list, ("Vanilla ACORN: " + p.name).c_str());
        }

        acorn_idx.reorder_graph_hierarchical(cfg.bfs_budget);
        for (auto& p : preds) {
            p.fmap_reord.resize(nb);
            for (size_t nid = 0; nid < nb; ++nid) {
                p.fmap_reord[nid] = p.fmap[acorn_idx.new_to_old[nid]];
            }
        }

        acorn_idx.compress_neighbors_vbyte(false);
        for (auto& p : preds) {
            p.vb = run_search_suite(
                    acorn_idx, xq.data(), nq, d_base, cfg.k, true, p.fmap_reord, true, false,
                    p.gt.data(), gt_stride, cfg.ef_list, ("Option A (VByte): " + p.name).c_str());
        }

        acorn_idx.compress_neighbors_simd_bp128(true, cfg.proximity_prefix);
        for (auto& p : preds) {
            p.simd = run_search_suite(
                    acorn_idx, xq.data(), nq, d_base, cfg.k, true, p.fmap_reord, true, false,
                    p.gt.data(), gt_stride, cfg.ef_list, ("Option B (AVX2 SIMD-BP128): " + p.name).c_str());
        }

        size_t last_idx = cfg.ef_list.size() - 1;
        int top_ef = cfg.ef_list[last_idx];
        std::cout << "\n=== Unified Real/Skewed Predicate Summary (efSearch=" << top_ef << ") ===\n";
        std::cout << "| Predicate | Sel % | Pre(Bmp) p50 | Pre(Inv) p50 | Van R@10 | SIMD R@10 | Van p50 | VByte p50 | SIMD p50 | SIMD vs Van |\n";
        std::cout << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        for (const auto& p : preds) {
            double sel_pct = 100.0 * static_cast<double>(p.passing) / nb;
            double ovh = 100.0 * (p.simd[last_idx].p50_ms - p.van[last_idx].p50_ms) / p.van[last_idx].p50_ms;
            std::cout << "| " << p.name
                      << " | " << std::setprecision(2) << sel_pct << "%"
                      << " | " << std::setprecision(3) << p.pre.bitmap_p50_ms << " ms"
                      << " | " << std::setprecision(3) << p.pre.invlist_p50_ms << " ms"
                      << " | " << std::setprecision(4) << p.van[last_idx].recall
                      << " | " << std::setprecision(4) << p.simd[last_idx].recall
                      << " | " << std::setprecision(3) << p.van[last_idx].p50_ms << " ms"
                      << " | " << std::setprecision(3) << p.vb[last_idx].p50_ms << " ms"
                      << " | " << std::setprecision(3) << p.simd[last_idx].p50_ms << " ms"
                      << " | " << std::showpos << std::setprecision(1) << ovh << "%" << std::noshowpos
                      << " |\n";
        }
        return 0;
    }

    if (cfg.predicate_mode == "real_range" || cfg.predicate_mode == "real_compound") {
        size_t wl_dim = 0, wl_n = 0;
        std::vector<int32_t> word_lens =
                read_ivecs(cfg.data_dir + "/word_lengths.ivecs", nb, wl_dim, wl_n);
        for (size_t i = 0; i < nb; ++i) {
            int wl = word_lens[i * wl_dim];
            bool ok = (wl >= cfg.min_len && wl <= cfg.max_len);
            if (cfg.predicate_mode == "real_compound") {
                ok = ok && (i < cfg.max_rank);
            }
            filter_map[i] = ok ? 1 : 0;
            metadata[i] = ok ? 0 : 1;
            passing_count += static_cast<size_t>(filter_map[i]);
        }
    } else if (cfg.predicate_mode == "zipf") {
        const int zclasses = 100;
        std::vector<double> cdf(zclasses, 0.0);
        double norm = 0.0;
        for (int c = 0; c < zclasses; ++c) {
            norm += 1.0 / std::pow(static_cast<double>(c + 1), cfg.zipf_alpha);
            cdf[c] = norm;
        }
        for (int c = 0; c < zclasses; ++c) cdf[c] /= norm;
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> udist(0.0, 1.0);
        for (size_t i = 0; i < nb; ++i) {
            int b = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), udist(rng)) - cdf.begin());
            metadata[i] = std::min(b, zclasses - 1);
            filter_map[i] = (metadata[i] == cfg.target_bucket) ? 1 : 0;
            passing_count += static_cast<size_t>(filter_map[i]);
        }
    } else if (is_filtered) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, num_classes - 1);
        for (size_t i = 0; i < nb; ++i) {
            metadata[i] = dist(rng);
            filter_map[i] = (metadata[i] == 0) ? 1 : 0;
            passing_count += static_cast<size_t>(filter_map[i]);
        }
    } else {
        passing_count = nb;
    }
    double actual_sel = static_cast<double>(passing_count) / static_cast<double>(nb);
    std::cout << "Predicate mode=" << cfg.predicate_mode
              << ", passing_nodes=" << passing_count
              << " (actual selectivity=" << std::setprecision(4) << actual_sel << ")\n";

    // 3. Load or compute exact groundtruth
    std::vector<int32_t> gt;
    size_t gt_stride = static_cast<size_t>(cfg.k);
    auto t_gt0 = std::chrono::high_resolution_clock::now();
    if (!is_filtered && nb == 1000000) {
        size_t gt_n = 0;
        gt = read_ivecs(cfg.data_dir + "/sift_groundtruth.ivecs", nq, gt_stride, gt_n);
        std::cout << "Loaded official SIFT1M groundtruth (" << gt_n << " x " << gt_stride << ")\n";
    } else {
        gt = compute_exact_gt_bitmap(xb.data(), nb, xq.data(), nq, d_base, cfg.k, filter_map, is_filtered);
        auto t_gt1 = std::chrono::high_resolution_clock::now();
        double gt_s = std::chrono::duration<double>(t_gt1 - t_gt0).count();
        std::cout << "Computed exact groundtruth in " << std::setprecision(2) << gt_s << " s\n";
    }

    if (cfg.bench_prefilter || cfg.bench_prefilter_only) {
        run_prefilter_baseline(
                xb.data(), nb, xq.data(), nq, d_base, cfg.k,
                filter_map, gt.data(), gt_stride, cfg.predicate_mode.c_str());
        if (cfg.bench_prefilter_only) {
            return 0;
        }
    }

    // Optional Post-Filter HNSW baseline (before freeing xb)
    if (cfg.bench_hnsw) {
        run_postfilter_hnsw_baseline(
                xb.data(), nb, xq.data(), nq, d_base, cfg.M, cfg.k,
                is_filtered, filter_map, gt.data(), gt_stride, cfg.ef_list);
        return 0;
    }

    // Optional Part 3D: Live Dynamic Insertion Benchmark (Cluster-Gapped IDs + Delta-Overflow)
    if (cfg.bench_dynamic) {
        constexpr double MB = 1024.0 * 1024.0;
        const size_t n_ins = std::max<size_t>(1, static_cast<size_t>(std::llround(nb * cfg.insert_pct)));
        const size_t n_init = nb - n_ins;
        std::cout << "\n=== Part 3D: Dynamic Insertion Benchmark (Cluster-Gapped IDs + Delta-Overflow) ===\n";
        std::cout << "Initial build: " << n_init << " vectors | Live insert: +" << n_ins
                  << " vectors (" << std::setprecision(1) << (cfg.insert_pct * 100.0)
                  << "%) | gap_ratio=" << cfg.gap_ratio << "\n";

        std::vector<int> init_meta(metadata.begin(), metadata.begin() + n_init);
        auto tb0 = std::chrono::high_resolution_clock::now();
        faiss::IndexACORNFlat dyn_index(
                static_cast<int>(d_base), cfg.M, cfg.gamma, init_meta, cfg.M_beta, faiss::METRIC_L2);
        dyn_index.verbose = false;
        dyn_index.add(static_cast<faiss::idx_t>(n_init), xb.data());
        auto tb1 = std::chrono::high_resolution_clock::now();
        double build_s = std::chrono::duration<double>(tb1 - tb0).count();
        double full_vanilla_est_mb = (static_cast<double>(nb) * 113.03 * 4.0 + nb * 12.0) / MB;

        auto tr0 = std::chrono::high_resolution_clock::now();
        size_t peak_q = dyn_index.reorder_graph_hierarchical(cfg.bfs_budget);
        dyn_index.enable_dynamic_gaps(cfg.gap_ratio, 1000);
        dyn_index.compress_neighbors_simd_bp128(true, cfg.proximity_prefix);
        auto tr1 = std::chrono::high_resolution_clock::now();
        double prep_s = std::chrono::duration<double>(tr1 - tr0).count();

        auto mem_before = measure_graph_memory(dyn_index);
        double rss_before = get_process_rss_mb();
        std::cout << "Build (" << n_init << "): " << std::setprecision(2) << build_s
                  << " s | Reorder+Gap+Compress: " << prep_s << " s (peak_q=" << peak_q << ")\n";
        std::cout << "Before Live Inserts -> Graph: " << mem_before.total_graph_mb
                  << " MB | OS RSS: " << rss_before << " MB\n";

        // Perform live dynamic insertions of held-out vectors
        auto ti0 = std::chrono::high_resolution_clock::now();
        size_t local_hits = dyn_index.add_dynamic(
                static_cast<faiss::idx_t>(n_ins),
                xb.data() + n_init * d_base,
                metadata.data() + n_init,
                32);
        auto ti1 = std::chrono::high_resolution_clock::now();
        double ins_s = std::chrono::duration<double>(ti1 - ti0).count();

        // Free temporary xb now that all vectors are inside dyn_index
        std::vector<float>().swap(xb);

        size_t ovf_edges = 0;
        for (const auto& kv : dyn_index.acorn.overflow_l0) {
            ovf_edges += kv.second.size();
        }
        double ovf_mb = (ovf_edges * sizeof(int32_t) + dyn_index.acorn.overflow_l0.size() * 48.0 +
                         dyn_index.acorn.has_overflow.size()) / MB;
        auto mem_after = measure_graph_memory(dyn_index);
        double rss_after = get_process_rss_mb();

        std::cout << "\n--- After +" << n_ins << " Live Dynamic Inserts ---\n";
        std::cout << "Insert time:         " << std::setprecision(3) << ins_s << " s ("
                  << std::setprecision(1) << (n_ins / ins_s) << " inserts/sec)\n";
        std::cout << "Local cluster hits:  " << local_hits << " / " << n_ins << " ("
                  << std::setprecision(2) << (100.0 * local_hits / n_ins) << "% landed in primary BFS cluster!)\n";
        std::cout << "Graph + Overflow MB: " << std::setprecision(2) << (mem_after.total_graph_mb + ovf_mb)
                  << " MB (compressed=" << mem_after.total_graph_mb << " MB, overflow=" << ovf_mb
                  << " MB) --> " << (100.0 * (1.0 - (mem_after.total_graph_mb + ovf_mb) / full_vanilla_est_mb))
                  << "% savings vs full-" << nb << " Vanilla\n";
        std::cout << "OS Process RSS:      " << std::setprecision(2) << rss_after << " MB\n";

        // Build gapped filter map for fast pure-search measurement
        std::vector<char> gapped_filter(static_cast<size_t>(dyn_index.ntotal), 0);
        for (size_t slot = 0; slot < static_cast<size_t>(dyn_index.ntotal); ++slot) {
            faiss::ACORN::storage_idx_t ext_id = dyn_index.new_to_old[slot];
            if (ext_id >= 0 && static_cast<size_t>(ext_id) < nb) {
                gapped_filter[slot] = filter_map[static_cast<size_t>(ext_id)];
            }
        }

        run_search_suite(
                dyn_index, xq.data(), nq, d_base, cfg.k, is_filtered, gapped_filter, true,
                false, gt.data(), gt_stride, cfg.ef_list,
                "Dynamic ACORN: AFTER Live Inserts (Before Compaction)");

        // Run fast re-compaction
        auto tc0 = std::chrono::high_resolution_clock::now();
        dyn_index.compact_dynamic_l0(cfg.proximity_prefix);
        auto tc1 = std::chrono::high_resolution_clock::now();
        double compact_s = std::chrono::duration<double>(tc1 - tc0).count();
        auto mem_comp = measure_graph_memory(dyn_index);
        double rss_comp = get_process_rss_mb();

        std::cout << "\n--- After Fast Re-Compaction (compact_dynamic_l0) ---\n";
        std::cout << "Compaction time:     " << std::setprecision(3) << compact_s << " s\n";
        std::cout << "Compacted Graph MB:  " << std::setprecision(2) << mem_comp.total_graph_mb
                  << " MB --> " << (100.0 * (1.0 - mem_comp.total_graph_mb / full_vanilla_est_mb))
                  << "% savings vs full-" << nb << " Vanilla\n";
        std::cout << "OS Process RSS:      " << std::setprecision(2) << rss_comp << " MB\n";

        run_search_suite(
                dyn_index, xq.data(), nq, d_base, cfg.k, is_filtered, gapped_filter, true,
                false, gt.data(), gt_stride, cfg.ef_list,
                "Dynamic ACORN: AFTER Re-Compaction (Overflow Merged)");
        return 0;
    }

    // 4. Build or Load IndexACORNFlat
    faiss::IndexACORNFlat index(
            static_cast<int>(d_base),
            cfg.M,
            cfg.gamma,
            metadata,
            cfg.M_beta,
            faiss::METRIC_L2);
    index.verbose = false;

    if (!cfg.load_index_path.empty()) {
        std::vector<float>().swap(xb); // free base vectors before loading serialized index
        std::cout << "Loading serialized IndexACORNFlat from " << cfg.load_index_path << "...\n";
        auto tl0 = std::chrono::high_resolution_clock::now();
        index.load_from_file(cfg.load_index_path);
        auto tl1 = std::chrono::high_resolution_clock::now();
        double load_s = std::chrono::duration<double>(tl1 - tl0).count();
        auto mem_loaded = measure_graph_memory(index);
        std::cout << "Loaded index (ntotal=" << index.ntotal << ", compressed="
                  << (index.acorn.use_compressed_l0 ? "yes" : "no")
                  << ", reordered=" << (index.is_reordered ? "yes" : "no")
                  << ") in " << std::fixed << std::setprecision(3) << load_s << " s\n";
        std::cout << "Loaded Graph MB:     " << std::setprecision(2) << mem_loaded.total_graph_mb
                  << " MB | OS RSS: " << get_process_rss_mb() << " MB\n";

        std::vector<char> loaded_fmap = filter_map;
        if (is_filtered && index.is_reordered) {
            loaded_fmap.resize(nb);
            for (size_t nid = 0; nid < nb; ++nid) {
                loaded_fmap[nid] = filter_map[index.new_to_old[nid]];
            }
        }
        run_search_suite(
                index, xq.data(), nq, d_base, cfg.k, is_filtered, loaded_fmap,
                index.is_reordered, cfg.profile_decode, gt.data(), gt_stride, cfg.ef_list,
                "Loaded IndexACORNFlat Search Benchmark");
        return 0;
    }

    std::cout << "Building IndexACORNFlat...\n" << std::flush;
    auto t_build0 = std::chrono::high_resolution_clock::now();
    index.add(static_cast<faiss::idx_t>(nb), xb.data());
    auto t_build1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t_build1 - t_build0).count();

    // 5. Measure exact graph memory footprint
    size_t total_slots = index.acorn.neighbors.size();
    size_t active_edges = 0;
    for (faiss::ACORN::storage_idx_t v : index.acorn.neighbors) {
        if (v >= 0) {
            ++active_edges;
        }
    }
    size_t neighbors_bytes = total_slots * sizeof(faiss::ACORN::NeighNode);
    size_t offsets_bytes = index.acorn.offsets.size() * sizeof(size_t);
    size_t levels_bytes = index.acorn.levels.size() * sizeof(int);
    size_t graph_total_bytes = neighbors_bytes + offsets_bytes + levels_bytes;
    size_t vectors_bytes = nb * d_base * sizeof(float);

    constexpr double MB = 1024.0 * 1024.0;
    std::cout << "\n--- Index Build & Memory Summary ---\n";
    std::cout << "Build time:          " << std::setprecision(2) << build_s << " s\n";
    std::cout << "Max level:           " << index.acorn.max_level << "\n";
    std::cout << "Neighbor slots:      " << total_slots
              << " (" << std::setprecision(2) << (static_cast<double>(total_slots) / nb) << " slots/node)\n";
    std::cout << "Active edges (>=0):  " << active_edges
              << " (" << std::setprecision(2) << (static_cast<double>(active_edges) / nb) << " edges/node)\n";
    std::cout << "neighbors[] memory:  " << std::setprecision(2) << (neighbors_bytes / MB) << " MB ("
              << (static_cast<double>(neighbors_bytes) / total_slots) << " bytes/slot)\n";
    std::cout << "offsets+levels mem:  " << std::setprecision(2) << ((offsets_bytes + levels_bytes) / MB) << " MB\n";
    std::cout << "Total graph memory:  " << std::setprecision(2) << (graph_total_bytes / MB) << " MB\n";
    std::cout << "Raw vectors memory:  " << std::setprecision(2) << (vectors_bytes / MB) << " MB\n";
    // Release temporary xb copy now that index owns its own vectors (for cleaner OS RSS readings)
    std::vector<float>().swap(xb);
    double rss_vanilla = get_process_rss_mb();
    std::cout << "OS Process RSS:      " << std::setprecision(2) << rss_vanilla << " MB\n";

    if (cfg.ab_validate) {
        // Stage 1: Vanilla ACORN (Uncompressed)
        auto res_vanilla = run_search_suite(
                index, xq.data(), nq, d_base, cfg.k, is_filtered, filter_map, false, false,
                gt.data(), gt_stride, cfg.ef_list, "Stage 1: Vanilla ACORN (Uncompressed)");
        auto mem_vanilla = measure_graph_memory(index);
        if (cfg.mt_qps) {
            std::cout << "\n--- Multi-Thread QPS Scaling (1T / 2T / 4T) ---\n";
            std::cout << "| Variant | 1T QPS | 2T QPS | 4T QPS |\n|---|---:|---:|---:|\n";
            run_mt_qps_suite(index, xq.data(), nq, d_base, cfg.k, is_filtered, filter_map, false, 64, "Vanilla ACORN");
            run_mt_qps_suite(index, xq.data(), nq, d_base, cfg.k, is_filtered, filter_map, false, 128, "Vanilla ACORN");
        }

        // Stage 2: BFS Reordered (Uncompressed)
        std::vector<char> active_filter_map = filter_map;
        std::vector<SearchRunResult> res_reordered;
        if (!cfg.no_reorder_compress) {
            auto t_r0 = std::chrono::high_resolution_clock::now();
            size_t peak_q = nb;
            if (cfg.reorder_mode == "hierarchical") {
                peak_q = index.reorder_graph_hierarchical(cfg.bfs_budget);
            } else {
                index.reorder_graph_bfs();
            }
            auto t_r1 = std::chrono::high_resolution_clock::now();
            double reord_s = std::chrono::duration<double>(t_r1 - t_r0).count();
            std::cout << "\n[Stage 2] Reordering (" << cfg.reorder_mode
                      << ") time: " << std::setprecision(3) << reord_s
                      << " s | peak_queue=" << peak_q << " ("
                      << std::setprecision(2) << (peak_q * 4.0 / 1024.0) << " KB)\n";

            if (is_filtered) {
                active_filter_map.resize(nb);
                for (size_t nid = 0; nid < nb; ++nid) {
                    active_filter_map[nid] = filter_map[index.new_to_old[nid]];
                }
            }
            res_reordered = run_search_suite(
                    index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map, true, false,
                    gt.data(), gt_stride, cfg.ef_list, "Stage 2: BFS Reordered (Uncompressed)");
        }

        // Stage 3: VByte, Elias-Fano, and/or AVX2 SIMD-BP128 Compression
        if (cfg.codec == "both" || cfg.codec == "all") {
            const bool run_ef = (cfg.codec == "all");
            // Stage 3A: Option A (Scalar VByte, keep L0 temporarily so subsequent codecs run on exact same index)
            auto t_vb0 = std::chrono::high_resolution_clock::now();
            index.compress_neighbors_vbyte(false);
            auto t_vb1 = std::chrono::high_resolution_clock::now();
            double vb_comp_s = std::chrono::duration<double>(t_vb1 - t_vb0).count();
            auto mem_vb_raw = measure_graph_memory(index);
            // Upper-level neighbors size = total - L0
            double l0_raw_mb = (static_cast<double>(nb) * index.acorn.nb_neighbors(0) * 4.0) / MB;
            double vb_graph_mb = (mem_vb_raw.neighbors_mb - l0_raw_mb) + mem_vb_raw.offsets_levels_mb +
                    mem_vb_raw.vbyte_payload_mb + mem_vb_raw.vbyte_meta_mb;
            double vb_savings_pct = 100.0 * (1.0 - vb_graph_mb / mem_vanilla.total_graph_mb);

            std::cout << "\n--- Stage 3A: Option A (Scalar VByte) Summary ---\n";
            std::cout << "VByte compress time: " << std::setprecision(3) << vb_comp_s << " s\n";
            std::cout << "VByte Graph MB:      " << std::setprecision(2) << vb_graph_mb
                      << " MB (payload=" << mem_vb_raw.vbyte_payload_mb
                      << ", meta=" << mem_vb_raw.vbyte_meta_mb << ")  -->  "
                      << vb_savings_pct << "% savings\n";

            auto res_vbyte = run_search_suite(
                    index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map,
                    !cfg.no_reorder_compress, cfg.profile_decode, gt.data(), gt_stride, cfg.ef_list,
                    "Stage 3A: Option A — BFS Reordered + Scalar VByte");
            if (cfg.mt_qps) {
                run_mt_qps_suite(index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map, !cfg.no_reorder_compress, 64, "Option A (VByte)");
                run_mt_qps_suite(index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map, !cfg.no_reorder_compress, 128, "Option A (VByte)");
            }

            // Optional Stage 3E: Elias-Fano Baseline (keep L0 temporarily so 3B runs on exact same index)
            std::vector<SearchRunResult> res_ef;
            double ef_graph_mb = 0.0;
            double ef_savings_pct = 0.0;
            if (run_ef) {
                auto t_ef0 = std::chrono::high_resolution_clock::now();
                index.compress_neighbors_elias_fano(false, cfg.proximity_prefix);
                auto t_ef1 = std::chrono::high_resolution_clock::now();
                double ef_comp_s = std::chrono::duration<double>(t_ef1 - t_ef0).count();
                auto mem_ef_raw = measure_graph_memory(index);
                ef_graph_mb = (mem_ef_raw.neighbors_mb - l0_raw_mb) + mem_ef_raw.offsets_levels_mb +
                        mem_ef_raw.vbyte_payload_mb + mem_ef_raw.vbyte_meta_mb;
                ef_savings_pct = 100.0 * (1.0 - ef_graph_mb / mem_vanilla.total_graph_mb);

                std::cout << "\n--- Stage 3E: Baseline (Elias-Fano) Summary ---\n";
                std::cout << "Elias-Fano time:     " << std::setprecision(3) << ef_comp_s
                          << " s (proximity_prefix=" << cfg.proximity_prefix << ")\n";
                std::cout << "Elias-Fano Graph MB: " << std::setprecision(2) << ef_graph_mb
                          << " MB (payload=" << mem_ef_raw.vbyte_payload_mb
                          << ", meta=" << mem_ef_raw.vbyte_meta_mb << ")  -->  "
                          << ef_savings_pct << "% savings\n";

                res_ef = run_search_suite(
                        index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map,
                        !cfg.no_reorder_compress, cfg.profile_decode, gt.data(), gt_stride, cfg.ef_list,
                        "Stage 3E: Baseline — Reordered + Elias-Fano");
            }

            // Stage 3B: Option B (AVX2 SIMD-BP128, free uncompressed L0)
            auto t_bp0 = std::chrono::high_resolution_clock::now();
            index.compress_neighbors_simd_bp128(true, cfg.proximity_prefix);
            auto t_bp1 = std::chrono::high_resolution_clock::now();
            double bp_comp_s = std::chrono::duration<double>(t_bp1 - t_bp0).count();
            auto mem_bp = measure_graph_memory(index);
            double rss_bp = get_process_rss_mb();
            double bp_savings_pct = 100.0 * (1.0 - mem_bp.total_graph_mb / mem_vanilla.total_graph_mb);
            double bp_with_id_pct =
                    100.0 * (1.0 - (mem_bp.total_graph_mb + mem_bp.id_map_mb) / mem_vanilla.total_graph_mb);

            std::cout << "\n--- Stage 3B: Option B (AVX2 SIMD-BP128, L0 Freed) Summary ---\n";
            std::cout << "SIMD-BP128 time:     " << std::setprecision(3) << bp_comp_s
                      << " s (proximity_prefix=" << cfg.proximity_prefix << ")\n";
            std::cout << "Vanilla Graph MB:    " << std::setprecision(2) << mem_vanilla.total_graph_mb << " MB\n";
            std::cout << "SIMD-BP128 Graph MB: " << std::setprecision(2) << mem_bp.total_graph_mb
                      << " MB (upper=" << mem_bp.neighbors_mb
                      << ", off+lvl=" << mem_bp.offsets_levels_mb
                      << ", payload=" << mem_bp.vbyte_payload_mb
                      << ", meta=" << mem_bp.vbyte_meta_mb << ")\n";
            std::cout << "Graph Savings:       " << std::setprecision(2) << bp_savings_pct
                      << "% (excl. ID map) | " << bp_with_id_pct << "% (incl. ID map)\n";
            std::cout << "OS Process RSS:      " << std::setprecision(2) << rss_vanilla
                      << " MB -> " << rss_bp << " MB (delta: " << (rss_bp - rss_vanilla) << " MB)\n";

            auto res_bp = run_search_suite(
                    index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map,
                    !cfg.no_reorder_compress, cfg.profile_decode, gt.data(), gt_stride, cfg.ef_list,
                    "Stage 3B: Option B — BFS Reordered + AVX2 SIMD-BP128 (L0 Freed)");
            if (cfg.mt_qps) {
                run_mt_qps_suite(index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map, !cfg.no_reorder_compress, 64, "Option B (AVX2 SIMD)");
                run_mt_qps_suite(index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map, !cfg.no_reorder_compress, 128, "Option B (AVX2 SIMD)");
            }

            if (run_ef) {
                std::cout << "\n=== 4-Way Ablation Summary (Vanilla vs VByte vs Elias-Fano vs AVX2 SIMD-BP128) ===\n";
                std::cout << "Graph RAM: Vanilla=" << std::setprecision(2) << mem_vanilla.total_graph_mb
                          << " MB | VByte=" << vb_graph_mb << " MB (-" << vb_savings_pct
                          << "%) | Elias-Fano=" << ef_graph_mb << " MB (-" << ef_savings_pct
                          << "%) | SIMD-BP128=" << mem_bp.total_graph_mb << " MB (-" << bp_savings_pct << "%)\n";
                std::cout << "| efSearch | Van R@" << cfg.k << " | EF R@" << cfg.k
                          << " | SIMD R@" << cfg.k << " | Van p50 | VByte p50 | EF p50 | SIMD p50 | SIMD vs Van | SIMD vs EF |\n";
                std::cout << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
                for (size_t i = 0; i < res_vanilla.size(); ++i) {
                    double ovh_van = 100.0 * (res_bp[i].p50_ms - res_vanilla[i].p50_ms) / res_vanilla[i].p50_ms;
                    double ovh_ef = 100.0 * (res_bp[i].p50_ms - res_ef[i].p50_ms) / res_ef[i].p50_ms;
                    std::cout << "| " << res_vanilla[i].efs
                              << " | " << std::setprecision(4) << res_vanilla[i].recall
                              << " | " << std::setprecision(4) << res_ef[i].recall
                              << " | " << std::setprecision(4) << res_bp[i].recall
                              << " | " << std::setprecision(3) << res_vanilla[i].p50_ms;
                    if (g_bench_runs > 1) std::cout << " +/- " << res_vanilla[i].p50_ci95_ms;
                    std::cout << " ms | " << std::setprecision(3) << res_vbyte[i].p50_ms;
                    if (g_bench_runs > 1) std::cout << " +/- " << res_vbyte[i].p50_ci95_ms;
                    std::cout << " ms | " << std::setprecision(3) << res_ef[i].p50_ms;
                    if (g_bench_runs > 1) std::cout << " +/- " << res_ef[i].p50_ci95_ms;
                    std::cout << " ms | " << std::setprecision(3) << res_bp[i].p50_ms;
                    if (g_bench_runs > 1) std::cout << " +/- " << res_bp[i].p50_ci95_ms;
                    std::cout << " ms | " << std::showpos << std::setprecision(1) << ovh_van << "%"
                              << " | " << std::setprecision(1) << ovh_ef << "%" << std::noshowpos
                              << " |\n";
                }
            } else {
                std::cout << "\n=== 3-Way Ablation Summary (Vanilla vs VByte vs AVX2 SIMD-BP128) ===\n";
                std::cout << "| efSearch | Van R@" << cfg.k << " | VByte R@" << cfg.k
                          << " | SIMD R@" << cfg.k << " | Van p50 | VByte p50 | SIMD p50 | SIMD vs Van | SIMD vs VByte |\n";
                std::cout << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
                for (size_t i = 0; i < res_vanilla.size(); ++i) {
                    double ovh_van = 100.0 * (res_bp[i].p50_ms - res_vanilla[i].p50_ms) / res_vanilla[i].p50_ms;
                    double ovh_vb = 100.0 * (res_bp[i].p50_ms - res_vbyte[i].p50_ms) / res_vbyte[i].p50_ms;
                    std::cout << "| " << res_vanilla[i].efs
                              << " | " << std::setprecision(4) << res_vanilla[i].recall
                              << " | " << std::setprecision(4) << res_vbyte[i].recall
                              << " | " << std::setprecision(4) << res_bp[i].recall
                              << " | " << std::setprecision(3) << res_vanilla[i].p50_ms << " ms"
                              << " | " << std::setprecision(3) << res_vbyte[i].p50_ms << " ms"
                              << " | " << std::setprecision(3) << res_bp[i].p50_ms << " ms"
                              << " | " << std::showpos << std::setprecision(1) << ovh_van << "%"
                              << " | " << std::setprecision(1) << ovh_vb << "%" << std::noshowpos
                              << " |\n";
                }
            }
            return 0;
        }

        if (!cfg.sweep_prefix.empty()) {
            struct SweepRow {
                int p = 0;
                double graph_mb = 0.0;
                double savings_pct = 0.0;
                double b_per_edge = 0.0;
                std::vector<SearchRunResult> res;
            };
            std::vector<SweepRow> rows;
            rows.reserve(cfg.sweep_prefix.size());

            size_t l0_active_edges = 0;
            for (size_t i = 0; i < nb; ++i) {
                size_t b = 0, e = 0;
                index.acorn.neighbor_range(static_cast<faiss::idx_t>(i), 0, &b, &e);
                for (size_t j = b; j < e; ++j) {
                    if (index.acorn.neighbors[j] < 0) break;
                    ++l0_active_edges;
                }
            }

            // Snapshot full reordered offsets + neighbors so every P runs in true l0_freed=true mode
            std::vector<size_t> saved_offsets = index.acorn.offsets;
            std::vector<faiss::ACORN::NeighNode> saved_neighbors = index.acorn.neighbors;

            for (size_t pi = 0; pi < cfg.sweep_prefix.size(); ++pi) {
                int p = cfg.sweep_prefix[pi];
                index.acorn.offsets = saved_offsets;
                if (pi + 1 == cfg.sweep_prefix.size()) {
                    index.acorn.neighbors.swap(saved_neighbors);
                    std::vector<size_t>().swap(saved_offsets);
                } else {
                    index.acorn.neighbors = saved_neighbors;
                }
                index.acorn.l0_freed = false;
                index.acorn.use_compressed_l0 = false;

                index.compress_neighbors_simd_bp128(true, p);
                auto m = measure_graph_memory(index);

                SweepRow row;
                row.p = p;
                row.graph_mb = m.total_graph_mb;
                row.savings_pct = 100.0 * (1.0 - m.total_graph_mb / mem_vanilla.total_graph_mb);
                row.b_per_edge = (l0_active_edges > 0)
                        ? (static_cast<double>(index.acorn.compressed_l0_neighbors.size()) /
                           static_cast<double>(l0_active_edges))
                        : 0.0;
                std::string t = "SIMD-BP128 (l0_freed=true, proximity_prefix=" + std::to_string(p) + ")";
                row.res = run_search_suite(
                        index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map,
                        !cfg.no_reorder_compress, cfg.profile_decode, gt.data(), gt_stride,
                        cfg.ef_list, t.c_str());
                rows.push_back(std::move(row));
            }

            size_t last_ei = cfg.ef_list.size() - 1;
            std::cout << "\n=== Proximity-Prefix Sensitivity Summary (Vanilla Graph="
                      << std::setprecision(2) << mem_vanilla.total_graph_mb << " MB) ===\n";
            std::cout << "| Prefix P | Graph MB | Savings % | B/L0-Edge | R@10 (ef="
                      << cfg.ef_list[0] << ") | R@10 (ef=" << cfg.ef_list[last_ei]
                      << ") | R@10 Delta vs Van | p50 (ef=" << cfg.ef_list[last_ei]
                      << ") | vs Van p50 |\n";
            std::cout << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
            std::cout << "| Vanilla | " << std::setprecision(2) << mem_vanilla.total_graph_mb
                      << " MB | 0.00% | 4.00 B | " << std::setprecision(4) << res_vanilla[0].recall
                      << " | " << res_vanilla[last_ei].recall << " | +0.0000 | "
                      << std::setprecision(3) << res_vanilla[last_ei].p50_ms << " ms | Baseline |\n";
            for (const auto& r : rows) {
                double r_diff = r.res[last_ei].recall - res_vanilla[last_ei].recall;
                double p_ovh = 100.0 * (r.res[last_ei].p50_ms - res_vanilla[last_ei].p50_ms) /
                        res_vanilla[last_ei].p50_ms;
                std::cout << "| P=" << r.p
                          << " | " << std::setprecision(2) << r.graph_mb << " MB"
                          << " | " << r.savings_pct << "%"
                          << " | " << std::setprecision(2) << r.b_per_edge << " B"
                          << " | " << std::setprecision(4) << r.res[0].recall
                          << " | " << r.res[last_ei].recall
                          << " | " << std::showpos << r_diff << std::noshowpos
                          << " | " << std::setprecision(3) << r.res[last_ei].p50_ms << " ms"
                          << " | " << std::showpos << std::setprecision(1) << p_ovh << "%"
                          << std::noshowpos << " |\n";
            }
            return 0;
        }

        // Single-codec Stage 3 (vbyte, elias_fano, or simd_bp128)
        bool use_simd = (cfg.codec == "simd_bp128");
        bool use_ef = (cfg.codec == "elias_fano");
        auto t_c0 = std::chrono::high_resolution_clock::now();
        if (use_simd) {
            index.compress_neighbors_simd_bp128(true, cfg.proximity_prefix);
        } else if (use_ef) {
            index.compress_neighbors_elias_fano(true, cfg.proximity_prefix);
        } else {
            index.compress_neighbors_vbyte(true);
        }
        auto t_c1 = std::chrono::high_resolution_clock::now();
        double comp_s = std::chrono::duration<double>(t_c1 - t_c0).count();

        auto mem_comp = measure_graph_memory(index);
        double rss_comp = get_process_rss_mb();
        double graph_savings_pct = 100.0 * (1.0 - mem_comp.total_graph_mb / mem_vanilla.total_graph_mb);
        double graph_with_idmap_pct =
                100.0 * (1.0 - (mem_comp.total_graph_mb + mem_comp.id_map_mb) / mem_vanilla.total_graph_mb);

        std::cout << "\n--- Stage 3: " << (use_simd ? "AVX2 SIMD-BP128" : "VByte")
                  << " Compression & Real Memory Summary ---\n";
        std::cout << "Compression time:    " << std::setprecision(3) << comp_s << " s\n";
        std::cout << "Vanilla Graph MB:    " << std::setprecision(2) << mem_vanilla.total_graph_mb << " MB\n";
        std::cout << "Compressed Graph MB: " << std::setprecision(2) << mem_comp.total_graph_mb
                  << " MB (upper_neigh=" << mem_comp.neighbors_mb
                  << ", off+lvl=" << mem_comp.offsets_levels_mb
                  << ", payload=" << mem_comp.vbyte_payload_mb
                  << ", meta=" << mem_comp.vbyte_meta_mb << ")\n";
        std::cout << "ID perm map MB:      " << std::setprecision(2) << mem_comp.id_map_mb << " MB\n";
        std::cout << "Graph Savings:       " << std::setprecision(2) << graph_savings_pct
                  << "% (excl. ID map) | " << graph_with_idmap_pct << "% (incl. ID map)\n";
        std::cout << "OS Process RSS:      " << std::setprecision(2) << rss_vanilla
                  << " MB -> " << rss_comp << " MB (delta: " << (rss_comp - rss_vanilla) << " MB)\n";

        auto res_comp = run_search_suite(
                index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map,
                !cfg.no_reorder_compress, cfg.profile_decode, gt.data(), gt_stride, cfg.ef_list,
                use_simd ? "Stage 3: BFS Reordered + AVX2 SIMD-BP128 (L0 Freed)"
                         : (use_ef ? "Stage 3: BFS Reordered + Elias-Fano (L0 Freed)"
                                   : (cfg.no_reorder_compress ? "Stage 3: VByte Compressed (NO Reordering)"
                                                              : "Stage 3: BFS Reordered + VByte Compressed (L0 Freed)")));

        if (!cfg.save_index_path.empty()) {
            auto ts0 = std::chrono::high_resolution_clock::now();
            index.save_to_file(cfg.save_index_path);
            auto ts1 = std::chrono::high_resolution_clock::now();
            double save_s = std::chrono::duration<double>(ts1 - ts0).count();
            std::cout << "Saved compressed index to " << cfg.save_index_path
                      << " in " << std::setprecision(3) << save_s << " s\n";
        }

        // Print A/B Delta Summary
        std::cout << "\n=== A/B Gate Summary (Stage 3 vs Stage 1 Vanilla) ===\n";
        std::cout << "| efSearch | Vanilla R@" << cfg.k << " | Compressed R@" << cfg.k
                  << " | Recall Delta | Vanilla p50 | Comp p50 | p50 Overhead |\n";
        std::cout << "|---:|---:|---:|---:|---:|---:|---:|\n";
        for (size_t i = 0; i < res_vanilla.size(); ++i) {
            double r_delta = res_comp[i].recall - res_vanilla[i].recall;
            double lat_ovh = 100.0 * (res_comp[i].p50_ms - res_vanilla[i].p50_ms) / res_vanilla[i].p50_ms;
            std::cout << "| " << res_vanilla[i].efs
                      << " | " << std::setprecision(4) << res_vanilla[i].recall
                      << " | " << std::setprecision(4) << res_comp[i].recall
                      << " | " << std::showpos << std::setprecision(4) << r_delta << std::noshowpos
                      << " | " << std::setprecision(3) << res_vanilla[i].p50_ms << " ms"
                      << " | " << std::setprecision(3) << res_comp[i].p50_ms << " ms"
                      << " | " << std::showpos << std::setprecision(2) << lat_ovh << "%" << std::noshowpos
                      << " |\n";
        }
        return 0;
    }

    // Optional Part 4: Graph-aware BFS reordering + delta histogram analysis
    std::vector<char> active_filter_map = filter_map;
    if (cfg.reorder) {
        analyze_level0_deltas(index, "BEFORE Reordering");
        auto t_reord0 = std::chrono::high_resolution_clock::now();
        size_t peak_q = nb;
        if (cfg.reorder_mode == "hierarchical") {
            peak_q = index.reorder_graph_hierarchical(cfg.bfs_budget);
        } else {
            index.reorder_graph_bfs();
        }
        auto t_reord1 = std::chrono::high_resolution_clock::now();
        double reord_s = std::chrono::duration<double>(t_reord1 - t_reord0).count();
        std::cout << "\nGraph Reordering (" << cfg.reorder_mode << ") time: "
                  << std::setprecision(3) << reord_s << " s | peak_queue=" << peak_q
                  << " (" << std::setprecision(2) << (peak_q * 4.0 / 1024.0) << " KB)\n";
        analyze_level0_deltas(index,
                cfg.reorder_mode == "hierarchical" ? "AFTER Hierarchical Reordering"
                                                   : "AFTER BFS Reordering");

        if (is_filtered) {
            active_filter_map.resize(nb);
            for (size_t nid = 0; nid < nb; ++nid) {
                active_filter_map[nid] = filter_map[index.new_to_old[nid]];
            }
        }
    }

    run_search_suite(
            index, xq.data(), nq, d_base, cfg.k, is_filtered, active_filter_map, cfg.reorder,
            cfg.profile_decode, gt.data(), gt_stride, cfg.ef_list,
            "Search Benchmark (Single-Query Latency)");

    return 0;
} catch (const std::exception& e) {
    std::cerr << "\n[EXCEPTION] " << e.what() << "\n";
    return 1;
}
