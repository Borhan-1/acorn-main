# ACORN-SIMD: Hierarchical Graph Reordering and AVX2 Delta Compression for Hybrid Vector Search

This repository extends the C++ FAISS/ACORN hybrid vector search index (Patel et al., SIGMOD 2024) with **Hierarchical Multi-Layer Graph Reordering**, a **Base-Separated 16-Delta Mini-Block AVX2 Bit-Packing Codec (`SimdBP128Codec`)**, a **Proximity-Preserving Two-Segment Split**, and **Cluster-Gapped Dynamic Insertions** with fast batch re-compaction.

Built on top of [`stanford-futuredata/ACORN`](https://github.com/stanford-futuredata/ACORN) and [The FAISS Library](https://github.com/facebookresearch/faiss).

---

## Key Contributions & Headline Results

| Dataset ($N, d$) | Vanilla ACORN | Standard HNSW ($M=32$) | **Ours (AVX2 SIMD-BP128)** | Savings vs. ACORN / HNSW | Fixed Point (`σ`, `ef`) | `p50` vs. Vanilla & Recall@10 |
| :--- | ---: | ---: | ---: | :---: | :---: | :--- |
| **SIFT1M** (`1M × 128-d`, `ACORN-1`) | `442.63 MiB` | `259.52 MiB` | **`130.46 MiB`** (`2.20 B/edge`) | **`-70.5%` / `-49.7%`** | `σ=1%`, `ef=128` (`ef=32`) | **`-11.1%`** (`4.277 → 3.804 ms`; `ef=32`: **`-29.5%`**), R@10 `0.9755` vs. `0.9757` |
| **GloVe-400K** (`399K × 100-d`, NLP) | `176.61 MiB` | `103.55 MiB` | **`46.92 MiB`** (`1.99 B/edge`) | **`-73.4%` / `-54.7%`** | `σ=1%`, `ef=128` (`ef=32`) | **`+2.3%`** parity (`3.469 → 3.549 ms`; `ef=32`: **`-46.7%`**), R@10 `0.8952` vs. `0.8953` |
| **DBpedia-OpenAI** (`100K × 1,536-d`, LLM) | `44.27 MiB` | `25.96 MiB` | **`11.92 MiB`** (`1.72 B/edge`) | **`-73.1%` / `-54.1%`** | `σ=1%`, `ef=128` (`ef=32`) | **`-2.0%`** parity (`4.236 → 4.149 ms`; `ef=32`: **`-35.0%`**), R@10 `0.9783` vs. `0.9789` |
| **SIFT100K Dense** (`100K × 128-d`, `ACORN-γ`) | `48.68 MiB` | `25.96 MiB` | **`20.85 MiB`** (`1.50 B/edge`) | **`-57.2%` / `-19.7%`** | `σ=5%`, `ef=64` (`ef=32`) | **`-1.3%`** parity (`0.851 → 0.839 ms`; `ef=32`: **`-24.7%`**), R@10 `0.9958` vs. `0.9968` |

> *Single sequential runs on identical built graphs (4-thread AVX2 workstation; latency deltas within `±3%` represent practical parity). At `1M SIFT1M` (`σ=10%, ef=128`), 4-thread throughput improves by **`+21.4%`** (`297.2 → 360.8 QPS`).*

- **Hierarchical Reordering:** Bounds transient BFS queue workspace to $O(N/M + B_{\text{loc}})$ — **`32.0×` smaller peak queue** (`3,906 KiB → 122.1 KiB`) and **`1.93×` faster** than flat BFS on `1M SIFT1M`.
- **AVX2 Codec (`SimdBP128Codec`):** Decodes into stack memory at **`1.58 ns/edge`** (`633M edges/s`, **`5.02×` faster** than scalar `VByte`) with zero heap allocations and a bit-compatible scalar fallback.
- **Dynamic Live `add()`:** Sustains **`1,897.8 inserts/s`** (`+50,000` live inserts on `1M`, `75.36%` primary BFS cluster hits) and restores `-69.72%` compression in **`1.95 s`** via `compact_dynamic_l0()`.

---

## Repository Structure

- **Core C++ Engine:**
  - `faiss/impl/codec/simd_bp128.h` — Base-Separated 16-Delta Mini-Block AVX2 codec (`Avx2Lut` + `avx2_prefix_sum` + scalar fallback)
  - `faiss/impl/codec/vbyte_codec.h` — Scalar Variable-Byte ablation codec
  - `faiss/impl/ACORN.h`, `faiss/impl/ACORN.cpp` — Compressed L0 storage, stack-buffer 2-hop decoding, delta-overflow buffer, upstream memory-safety fixes
  - `faiss/IndexACORN.h`, `faiss/IndexACORN.cpp` — `reorder_graph_hierarchical()`, `compress_neighbors_simd_bp128()`, `enable_dynamic_gaps()`, `add_dynamic()`, `compact_dynamic_l0()`
- **Unit Tests & Benchmark Harness:**
  - `tests/test_simd_bp128.cpp` — 3,799 exhaustive bit-exactness cases + 104M-integer decode microbenchmark
  - `demos/bench_acorn_sift1m.cpp` — Unified benchmark harness (Pre-Filter, Post-Filter HNSW, Vanilla ACORN, VByte, AVX2 SIMD-BP128, real/Zipfian predicates, prefix sweep, multi-thread QPS, dynamic `add()`)
- **Dataset Scripts & Paper Assets:**
  - `scripts/download_sift.py`, `scripts/download_glove.py`, `scripts/download_dbpedia_openai.py`
  - `benchs/generate_thesis_plots.py` & `paper_figures/` — 6 camera-ready 300-DPI PNG + vector PDF figures & `tables.tex`
  - `planing/MANUSCRIPT_DRAFT.md`, `planing/THEORETICAL_ANALYSIS.md`, and phase trackers (`PHASE2`–`PHASE4`)

---

## Quick Start: Build & Reproduce

### 1. Clone & Download Datasets
```bash
git clone https://github.com/Borhan-1/acorn-main.git
cd acorn-main

# Download SIFT1M (1M x 128-d), GloVe-400K (399K x 100-d), and DBpedia-OpenAI (100K x 1,536-d)
python scripts/download_sift.py
python scripts/download_glove.py
python scripts/download_dbpedia_openai.py
```

### 2. Build `libfaiss.a` (Release + AVX2)
```bash
cmake -B build-release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target faiss -j 4
```

### 3. Run Codec Unit Test & Microbenchmark
```bash
g++ -O3 -std=c++17 -mavx2 -I. tests/test_simd_bp128.cpp -o test_simd_bp128_avx2
./test_simd_bp128_avx2
```

### 4. Compile & Run Main Benchmarks
**Linux:**
```bash
g++ -O3 -std=c++17 -fopenmp -mavx2 -I. demos/bench_acorn_sift1m.cpp build-release/faiss/libfaiss.a -lopenblas -o build-release/bench_acorn_sift1m
```
**Windows (MinGW-W64):**
```powershell
g++ -O3 -std=c++17 -fopenmp -mavx2 -I. demos\bench_acorn_sift1m.cpp build-release\faiss\libfaiss.a build-release\libopenblas.dll -lpsapi -o build-release\bench_acorn_sift1m.exe
```

**Example Benchmark Commands:**
```bash
# 1M SIFT1M 3-way ablation (Vanilla vs. VByte vs. AVX2 SIMD-BP128) at 1% selectivity
./build-release/bench_acorn_sift1m --n 1000000 --nq 1000 --selectivity 0.01 --efSearch 16,32,64,128 --reorder_mode hierarchical --proximity_prefix 4 --ab_validate --codec both

# GloVe-400K 5-predicate real range, compound AND & Zipfian suite
./build-release/bench_acorn_sift1m --data_dir Datasets/glove400k --n 400000 --nq 1000 --efSearch 32,64,128 --bench_predicate_suite

# DBpedia-OpenAI 1,536-d LLM embeddings (1% selectivity)
./build-release/bench_acorn_sift1m --data_dir Datasets/dbpedia_openai100k --n 100000 --nq 1000 --selectivity 0.01 --efSearch 32,64,128 --reorder_mode hierarchical --proximity_prefix 8 --ab_validate --codec both

# 1M SIFT1M live +50,000 dynamic insertions + fast re-compaction
./build-release/bench_acorn_sift1m --n 1000000 --nq 1000 --selectivity 0.10 --efSearch 32,64,128 --bench_dynamic --gap_ratio 0.05 --insert_pct 0.05

# Regenerate all 6 camera-ready PNG + PDF figures in paper_figures/
python benchs/generate_thesis_plots.py
```

---

## C++ API Usage Example

```cpp
// 1. Build standard ACORN-1 index
faiss::IndexACORNFlat index(d, /*M=*/32, /*gamma=*/1, metadata, /*M_beta=*/64, faiss::METRIC_L2);
index.add(nb, xb);

// 2. Hierarchical Multi-Layer Reordering (O(N/M + B) queue RAM)
index.reorder_graph_hierarchical(/*local_bfs_budget=*/128);

// 3. Compress Level-0 with AVX2 SIMD-BP128 & free uncompressed L0 array
index.compress_neighbors_simd_bp128(/*free_uncompressed_l0=*/true, /*proximity_prefix=*/4);

// 4. Filtered hybrid search (transparent ID & filter remapping)
index.search(nq, xq, k, D.data(), I.data(), filter_map.data(), &params);
```

---

## Acknowledgments & Upstream Attribution

This project builds upon the open-source SIGMOD 2024 **ACORN** implementation by Stanford FutureData:
> L. Patel, P. Kraft, C. Guestrin, and M. Zaharia, *"ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data,"* Proc. ACM Manag. Data (SIGMOD), vol. 2, no. 3, 2024. https://dl.acm.org/doi/10.1145/3654923
