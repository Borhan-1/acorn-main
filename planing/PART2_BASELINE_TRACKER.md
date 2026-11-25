# Part 2 Tracker — Vanilla ACORN SIFT1M Baseline

Tracks the 4-part execution of **Part 2 (Reproduce the vanilla ACORN baseline on SIFT1M)** from [NEXT_10_PART_PLAN.md](NEXT_10_PART_PLAN.md) and [THESIS_PLAN.md](THESIS_PLAN.md).

## Overall Progress

| Sub-Part | Title | Status |
|---|---|---|
| **Part 2.1** | Verify SIFT1M dataset, compiler toolchain, and ACORN API | ✅ **Done** |
| **Part 2.2** | Write & compile portable C++ SIFT1M baseline benchmark (`bench_acorn_sift1m.cpp`) | ✅ **Done** |
| **Part 2.3** | Run quick validation on 100K SIFT1M subset (verify recall, filters, memory stats) | ✅ **Done** |
| **Part 2.4** | Run full 1M SIFT1M baseline & record official results table | ✅ **Done** |

---

## Part 2.1 — Environment, Dataset & API Verification (Done)

### 1. SIFT1M Dataset (`Datasets/sift1M/`)
All 4 standard SIFT1M files are present with exact expected byte sizes:

| File | Size (Bytes) | Verified Dimensions / Count |
|---|---|---|
| `sift_base.fvecs` | `516,000,000` | 1,000,000 vectors × 128-dim `float32` (`516 B/vec`) |
| `sift_query.fvecs` | `5,160,000` | 10,000 queries × 128-dim `float32` (`516 B/vec`) |
| `sift_learn.fvecs` | `51,600,000` | 100,000 vectors × 128-dim `float32` (`516 B/vec`) |
| `sift_groundtruth.ivecs` | `4,040,000` | 10,000 queries × top-100 `int32` IDs (`404 B/query`) |

### 2. Build & Toolchain (`build-release/`)
- **Compiler:** `g++.exe` (MinGW-W64 `x86_64-ucrt-posix-seh`, GCC 16.1.0)
- **CMake Generator:** `MinGW Makefiles` (`CMAKE_BUILD_TYPE=Release`)
- **Compiled Library:** `build-release/faiss/libfaiss.a`
- **BLAS / LAPACK:** `c:/Users/Muhammad Borhan/Desktop/Acorn/openblas/lib/libopenblas.lib`
- **Parallelism:** OpenMP (`-fopenmp`)

### 3. ACORN API & Portability Findings
- **Index Class:** `faiss::IndexACORNFlat(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2)` in `faiss/IndexACORN.h`.
- **Search API:**
  - Unfiltered: `search(n, x, k, distances, labels, params)`
  - Filtered: `search(n, x, k, distances, labels, char* filter_id_map, params)`
- **Graph Memory Layout (`faiss/impl/ACORN.h`):**
  - `acorn.neighbors` (`std::vector<int32_t>` — main graph footprint, 4 bytes/slot)
  - `acorn.offsets` (`std::vector<size_t>`)
  - `acorn.levels` (`std::vector<int>`)
- **Portability Note:** Existing `demos/test_acorn.cpp` includes Linux-only POSIX headers (`<arpa/inet.h>`, `<sys/resource.h>`). In **Part 2.2**, we will write a clean, portable C++ benchmark (`demos/bench_acorn_sift1m.cpp`) linked directly against `build-release/faiss/libfaiss.a`.

---

## Part 2.2 — Create & Compile SIFT1M Benchmark (Done)
- **Source:** [demos/bench_acorn_sift1m.cpp](../demos/bench_acorn_sift1m.cpp)
- **Binary:** `build-release/bench_acorn_sift1m.exe`
- **Build Command:**
  ```powershell
  g++ -O3 -std=c++17 -fopenmp -I. demos\bench_acorn_sift1m.cpp build-release\faiss\libfaiss.a "..\openblas\lib\libopenblas.lib" -o build-release\bench_acorn_sift1m.exe
  ```
- **Capabilities:**
  - Loads SIFT1M `.fvecs` / `.ivecs` (configurable `--n`, `--nq`, `--M`, `--gamma`, `--M_beta`, `--k`, `--selectivity`, `--efSearch`, `--threads`).
  - Uses official `sift_groundtruth.ivecs` for unfiltered 1M, or computes exact OpenMP brute-force L2 groundtruth for subsets and filtered queries.
  - Reports build time, exact graph memory (`neighbors[]`, `offsets[]`, `levels[]`, active edges/node, bytes/slot), Recall@k, latency (`mean`, `p50`, `p95`, `p99`), 1T QPS, and avg distance computations (`ndis`).

## Part 2.3 — 100K Subset Validation Run (Done)

**Config:** `N=100,000`, `nq=1,000`, `d=128`, `M=32`, `gamma=1`, `M_beta=64`, `k=10`, `omp_threads=4`

### Index Build & Memory (100K)
- **Build time:** `13.12 – 16.77 s`
- **Max level:** `3`
- **Neighbor slots:** `11,305,152` (`113.05 slots/node`)
- **Active edges (`>= 0`):** `5,296,101` (`52.96 edges/node` — **~53% of allocated slots are `-1` padding!**)
- **`neighbors[]` memory:** `43.13 MB` (`4.00 bytes/slot`)
- **`offsets[] + levels[]` memory:** `1.14 MB`
- **Total graph memory:** `44.27 MB` (vs. `48.83 MB` raw vectors)

### Unfiltered (`selectivity = 1.0`)
| efSearch | Recall@10 | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | 1T QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.8638 | 1.021 | 0.804 | 2.103 | 3.768 | 979.0 |
| 32 | 0.9423 | 1.418 | 0.850 | 2.333 | 20.763 | 705.0 |
| 64 | 0.9779 | 1.115 | 1.002 | 1.984 | 3.047 | 897.0 |

### Filtered (`selectivity = 0.10`, actual `0.0994`, ACORN-1)
| efSearch | Recall@10 | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | 1T QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.9579 | 1.267 | 1.097 | 2.309 | 4.271 | 789.2 |
| 32 | 0.9890 | 1.905 | 1.228 | 3.381 | 14.014 | 525.0 |
| 64 | 0.9982 | 1.767 | 1.651 | 2.606 | 3.832 | 565.9 |

### Key Observations
- **Validation passed:** High recall on both unfiltered (`0.9779 @ ef=64`) and 10%-selectivity filtered hybrid search (`0.9982 @ ef=64`).
- **Strong compression opportunity:** Only `52.96` of `113.05` slots/node hold active edges; eliminating `-1` padding + delta/VByte compression should comfortably beat the 50%–60% thesis target.

## Part 2.4 — Full 1M SIFT1M Baseline & Results Table (Done)

**Config:** `N=1,000,000`, `nq=1,000`, `d=128`, `M=32`, `gamma=1`, `M_beta=64`, `k=10`, `omp_threads=4`

### Index Build & Exact Memory Footprint (1M)
| Metric | Value | Notes |
|---|---:|---|
| **Build time** | `133.03 – 145.19 s` | ~2.2 – 2.4 min on 4 threads |
| **Max level** | `4` | — |
| **Total neighbor slots** | `113,032,320` | `113.03 slots/node` |
| **Active edges (`>= 0`)** | `50,517,355` | `50.52 edges/node` (**55.3% `-1` padding**) |
| **`neighbors[]` memory** | **`431.18 MB`** | `4.00 bytes/slot` (primary compression target) |
| **`offsets[] + levels[]` memory** | `11.44 MB` | — |
| **Total graph memory** | **`442.63 MB`** | Nearly equal to raw vectors (`488.28 MB`) |

### Unfiltered (`selectivity = 1.0`, Official SIFT1M Groundtruth)
| efSearch | Recall@10 | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | 1T QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.7037 | 1.346 | 1.260 | 1.982 | 2.534 | 743.0 |
| 32 | 0.8317 | 1.171 | 1.136 | 1.783 | 2.243 | 853.8 |
| 64 | 0.9118 | 1.263 | 1.188 | 1.868 | 2.438 | 792.0 |
| 128 | 0.9554 | 1.837 | 1.791 | 2.652 | 3.214 | 544.4 |

### Filtered (`selectivity = 0.10`, actual `0.1001`, ACORN-1)
| efSearch | Recall@10 | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | 1T QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.7942 | 2.224 | 2.154 | 3.141 | 4.230 | 449.6 |
| 32 | 0.9005 | 4.651 | 3.300 | 7.643 | 38.524 | 215.0 |
| 64 | 0.9581 | 4.945 | 4.414 | 8.249 | 13.630 | 202.2 |
| 128 | 0.9852 | 8.059 | 7.558 | 11.049 | 15.377 | 124.1 |

---

## Status: Part 2 Complete ✅
All 4 sub-parts of **Part 2 (Vanilla ACORN SIFT1M Baseline)** are finished. Ready to proceed to **Part 3 (Compression Integration Checklist)** and **Part 4 (Graph Reordering Prototype)**.
