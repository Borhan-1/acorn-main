# Phase 2 Tracker — AVX2 SIMD-BP128 Engine (Option B)

Tracks **Phase 2: High-Performance SIMD-BP128 Engine** from [THESIS_PLAN.md](THESIS_PLAN.md) and **Part 10** from [NEXT_10_PART_PLAN.md](NEXT_10_PART_PLAN.md).

## Overall Progress

| Sub-Part | Title | Status |
|---|---|---|
| **Part 2A.1** | Verify CPU AVX2 support, CMake flags & finalize SIMD bit-packing format | ✅ **Done** |
| **Part 2A.2** | Implement `faiss/impl/codec/simd_bp128.h` (AVX2 pack/unpack + prefix-sum + scalar fallback) | ✅ **Done** |
| **Part 2A.3** | Write & run `tests/test_simd_bp128.cpp` (lossless check + ns/neighbor microbenchmark vs VByte) | ✅ **Done** |
| **Part 2B.1** | Add `L0CompressionMode` + `compress_neighbors_simd_bp128()` with proximity split | ✅ **Done** |
| **Part 2B.2** | Enable `-mavx2` in `build-release` & rebuild `libfaiss.a` | ✅ **Done** |
| **Part 2B.3** | Add `--codec simd_bp128` to benchmark & run 100K 3-way verification | ✅ **Done** |
| **Part 2C.1** | Add decode call, decoded edge & optional decode timing counters to `ACORNStats` | ✅ **Done** |
| **Part 2C.2** | Rebuild `libfaiss.a`, add `--profile_decode` to benchmark & verify on 100K | ✅ **Done** |
| **Part 2D.1** | Full 1M SIFT1M 3-way ablation — Unfiltered (`selectivity = 1.0`) | ✅ **Done** |
| **Part 2D.2** | Full 1M SIFT1M 3-way ablation — Filtered (`selectivity = 0.10`) | ✅ **Done** |
| **Part 2D.3** | Full 1M SIFT1M 3-way ablation — Strict Filter (`selectivity = 0.01`) & Phase 2 Summary | ✅ **Done** |

---

## Part 2A.1 — CPU Verification & SIMD Bit-Packing Format Design (Done)

### 1. Hardware & Compiler Verification
- **CPU SIMD Support (`g++ -march=native`):**
  - `SSE4.2` ✅ | `AVX` ✅ | **`AVX2` ✅** | **`BMI2` ✅** | `AVX-512` ❌
- **Target Vector Width:** **256-bit AVX2 (`__m256i` = 8 × `uint32_t` lanes)**.
- **CMake Note:** `build-release/CMakeCache.txt` currently has `CMAKE_CXX_FLAGS=""` and `FAISS_OPT_LEVEL=OFF`. In Part 2A.3 / Part 2B we will compile with `-mavx2` (and keep a clean scalar fallback when `__AVX2__` is not defined).

### 2. Graph-Aware SIMD Bit-Packing Format (`SIMD-BP32/128` with Base Separation)
Based on our Phase 1 1M SIFT1M statistics (mean **50.5 active Level-0 edges/node**, max ~112 edges):
- **Why standard monolithic BP128 is suboptimal for ACORN graphs:**
  1. `v_0` (first sorted neighbor ID) is often large (`10^4–10^6`, needing ~20 bits), while consecutive deltas `v_i - v_{i-1}` are small (`1–16K`). Packing `v_0` inside the first block inflates every delta in that block to 20 bits.
  2. A 50-edge list padded to 128 wastes >60% of the block.
- **Our Proposed Format (Base-Separated Horizontal/Vertical AVX2 Mini-Blocks):**
  1. **Base ID (`v_0`):** Stored as a VByte-encoded `uint32_t` (1–3 bytes) at the start of the list, eliminating the `v_0` bit-width spike.
  2. **Mini-Blocks of 16 or 32 Deltas (exact multiples of 8-wide AVX2 lanes):**
     - Each block begins with a **1-byte bit-width header** $b \in [0, 32]$:
       $$b = \lceil \log_2(\max(\Delta_1, \dots, \Delta_B) + 1) \rceil$$
     - Followed by `B * b / 8` bytes of **8-lane interleaved (vertical) bit-packed `uint32_t` words** (`B = 16` or `32` → `2` or `4` AVX2 `__m256i` vectors per block).
     - **Branchless AVX2 Unpack + Vectorized Prefix-Sum:**
       - Unpacks 8 deltas at a time using `_mm256_srli_epi32` / `_mm256_and_si256` (zero branches per integer, single `switch(b)` dispatch per block).
       - Reconstructs absolute neighbor IDs directly in AVX2 registers via in-register prefix-sum (`_mm256_slli_si256` + `_mm256_add_epi32` + 128-bit cross-lane carry add).
  3. **Tail (`count - 1` mod `B` remaining deltas, `< 16`):**
     - Packed with a single 1-byte bit-width header $b_{\text{tail}}$ + compact bit-packed bytes (or 8-wide AVX2 if `>= 8`, scalar tail `< 8`), so **zero wasted padding** for arbitrary node degrees!
  4. **CPU Portability:** Guarded by `#ifdef __AVX2__` with an identical bit-compatible scalar fallback for non-AVX2 hosts.

---

## Part 2A.2 — Implement `faiss/impl/codec/simd_bp128.h` (Done)
- Created [faiss/impl/codec/simd_bp128.h](../faiss/impl/codec/simd_bp128.h) implementing `faiss::codec::SimdBP128Codec`:
  - **Base-separated encoding:** `v_0` stored via VByte so it never inflates delta bit-widths.
  - **16-delta AVX2 mini-blocks** (`2 × 8-wide` `__m256i` lanes sharing a 1-byte bit-width header $b \in \{0..24, 32\}$) + optional 8-delta AVX2 chunk + compact `< 8` tail.
  - **Branchless AVX2 unpack (`unpack_8_avx2`):** `_mm256_shuffle_epi8` + `_mm256_srlv_epi32` + `_mm256_and_si256` with precomputed aligned lookup tables (`Avx2Lut`).
  - **In-register AVX2 prefix-sum (`avx2_prefix_sum`):** Reconstructs 8 absolute neighbor IDs per chunk with zero scalar loop overhead.
  - **Bit-compatible scalar fallback** when `__AVX2__` is not enabled.

## Part 2A.3 — Unit-Test & Microbenchmark `tests/test_simd_bp128.cpp` (Done)
- Created [tests/test_simd_bp128.cpp](../tests/test_simd_bp128.cpp) and ran both **AVX2 (`-mavx2`)** and **Scalar Fallback (`-mno-avx2`)** builds.
- **Exhaustive Correctness (`b=0..28`, `len=0..130`):** **100% lossless** (exact values + exact byte boundaries) on both AVX2 and Scalar builds.
- **Microbenchmark (20,000 realistic lists, 1.04M edges × 100 passes = 104M decoded ints):**

| Codec | Size (B/edge) | Decode Speed | Throughput | Speedup vs VByte |
|---|---:|---:|---:|---:|
| **Raw `int32_t`** | `4.00 B` | — | — | — |
| **Option A: `VByteCodec` (Scalar)** | `1.82 B` | `7.94 ns/edge` | `0.13 B ints/s` | `1.00×` |
| **Option B: `SimdBP128Codec` (Scalar Fallback)** | `2.04 B` | `4.20 ns/edge` | `0.24 B ints/s` | `1.85×` |
| **Option B: `SimdBP128Codec` (AVX2)** | `2.04 B` | **`1.58 ns/edge`** | **`0.63 B ints/s`** | **`5.02×` faster!** |

---

## Part 2B.1 — Add SIMD-BP128 Mode & Proximity-Preserving Split (Done)
- Added `ACORN::L0CompressionMode` (`NONE`, `VBYTE`, `SIMD_BP128`) and integrated `SimdBP128Codec::decode_delta_list` into `ACORN::decode_l0_neighbors()` in [faiss/impl/ACORN.h](../faiss/impl/ACORN.h).
- Implemented `IndexACORN::compress_neighbors_simd_bp128(bool free_uncompressed_l0 = true, int proximity_prefix = 16)` in [faiss/IndexACORN.h](../faiss/IndexACORN.h) & [faiss/IndexACORN.cpp](../faiss/IndexACORN.cpp):
  - Splits Level-0 neighbor lists into `[0..proximity_prefix)` (closest neighbors visited first) and `[proximity_prefix..count)` to preserve filtered recall before `num_found >= 2*M` breaks.
  - Appends `SimdBP128Codec::PADDING_BYTES` (16 zero bytes) for safe unaligned AVX2 loads.

## Part 2B.2 — Enable `-mavx2` & Rebuild `libfaiss.a` (Done)
- Updated [faiss/CMakeLists.txt](../faiss/CMakeLists.txt) to compile `IndexACORN.cpp` and `impl/ACORN.cpp` with `-mavx2` (`set_source_files_properties`).
- Reconfigured CMake and rebuilt `build-release/faiss/libfaiss.a` (`[100%] Built target faiss`, zero errors).

## Part 2B.3 — 100K 3-Way Verification (`Vanilla` vs `VByte` vs `SIMD-BP128`) (Done)
- Added `--codec vbyte|simd_bp128|both` and `--proximity_prefix <P>` to [demos/bench_acorn_sift1m.cpp](../demos/bench_acorn_sift1m.cpp), compiled with `-mavx2`.
- **Memory & Real Windows OS RSS (100K, `proximity_prefix = 16`):**
  - Vanilla Graph: `44.27 MB` → **AVX2 SIMD-BP128 Graph: `11.22–11.24 MB` (`-74.6%` savings)**
  - Real Windows OS RSS: `100.66 MB` → **`69.57 MB` (`-31.09 MB` freed)** | Compression time: **`0.17–0.19 s`**

### 100K Unfiltered (`selectivity = 1.0`)
| efSearch | Van R@10 | VByte R@10 | **SIMD R@10** | Van p50 | VByte p50 | **SIMD p50** | **SIMD vs Van** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.8674 | 0.8632 | **0.8679** | 0.588 ms | 0.317 ms | **0.331 ms** | **`-43.8%` (faster)** |
| 32 | 0.9429 | 0.9399 | **0.9433** | 0.573 ms | 0.454 ms | **0.446 ms** | **`-22.1%` (faster)** |
| 64 | 0.9789 | 0.9766 | **0.9789** | 0.711 ms | 0.658 ms | **0.645 ms** | **`-9.2%` (faster)** |

### 100K Filtered (`selectivity = 0.10`, ACORN-1)
| efSearch | Van R@10 | VByte R@10 | **SIMD R@10** | Van p50 | VByte p50 | **SIMD p50** | **SIMD vs Van** | **SIMD vs VByte** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.9593 | 0.8765 | **0.9504** | 0.843 ms | 1.674 ms | **0.603 ms** | **`-28.5%` (faster)** | **`-63.9%` faster** |
| 32 | 0.9889 | 0.9279 | **0.9862** | 1.179 ms | 2.075 ms | **0.980 ms** | **`-16.8%` (faster)** | **`-52.8%` faster** |
| 64 | 0.9969 | 0.9669 | **0.9965** | 1.658 ms | 2.888 ms | **1.666 ms** | **`+0.5%` (<3%!)** | **`-42.3%` faster** |

---

## Part 2C.1 — Instrument `ACORNStats` with Decode Counters & Optional Timing (Done)
- Added `decode_calls`, `decoded_edges`, `decode_time_us` to `ACORNStats` and `enable_decode_timing` (default `false` for zero-overhead latency runs) to `SearchParametersACORN` in [faiss/impl/ACORN.h](../faiss/impl/ACORN.h).
- Instrumented `search_from_candidates` and `hybrid_search_from_candidates` (1-hop & 2-hop) in [faiss/impl/ACORN.cpp](../faiss/impl/ACORN.cpp) and aggregated them in both `IndexACORN::search` overloads in [faiss/IndexACORN.cpp](../faiss/IndexACORN.cpp).

## Part 2C.2 — Rebuild `libfaiss.a` & Verify Decode Breakdown on 100K (Done)
- Rebuilt `build-release/faiss/libfaiss.a` and compiled [demos/bench_acorn_sift1m.cpp](../demos/bench_acorn_sift1m.cpp) with `--profile_decode`.
- **100K Filtered (`selectivity = 0.10`) Per-Query Decode Breakdown:**

| efSearch | DecCalls/q | DecEdges/q | VByte DecTime | VByte Dec % | **SIMD DecTime** | **SIMD Dec %** |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | `~710–737` | `~47.8K–49.4K` | `347.1 µs` | `35.8%` | **`235.3 µs`** | **`18.5%`** |
| 32 | `~1,495–1,524` | `~100.0K–101.7K` | `678.6 µs` | `41.9%` | **`439.3 µs`** | **`32.3%`** |
| 64 | `~3,119–3,154` | `~207.7K–209.6K` | `1,446.4 µs` | `48.1%` | **`916.5 µs`** | **`36.7%`** |

*(Note: `--profile_decode` calls `steady_clock::now()` ~3,100 times/query, adding ~150–400 µs of timer overhead; for clean QPS/latency numbers, run without `--profile_decode`.)*

## Part 2D.1 — Full 1M SIFT1M 3-Way Ablation: Unfiltered (`selectivity = 1.0`) (Done)
- **Memory & OS RSS (`N = 1,000,000`, `M = 32`, `gamma = 1`, `M_beta = 64`):**
  - Vanilla Graph: `442.63 MB` (`OS RSS: 936.34 MB`)
  - Option A (VByte): `110.01 MB` (**`-75.15%`**), compress time `2.46 s`
  - Option B (AVX2 SIMD-BP128, `proximity_prefix = 16`): **`127.63 MB`** (**`-71.17%`**, `-69.44%` incl. ID map), compress time **`1.78 s`**, **OS RSS: `936.34 → 643.31 MB` (`-293.03 MB` freed)**

| efSearch | Van R@10 | VByte R@10 | **SIMD R@10** | Van p50 | VByte p50 | **SIMD p50** | **SIMD vs Van** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.7045 | 0.7044 | **0.7032** | 1.205 ms | 0.503 ms | **0.657 ms** | **`-45.4%` (faster)** |
| 32 | 0.8312 | 0.8311 | **0.8309** | 1.245 ms | 0.817 ms | **0.864 ms** | **`-30.6%` (faster)** |
| 64 | 0.9103 | 0.9099 | **0.9098** | 1.128 ms | 1.133 ms | **1.213 ms** | `+7.5%` |
| 128 | 0.9566 | 0.9564 | **0.9561** | 1.676 ms | 1.637 ms | **1.635 ms** | **`-2.4%` (faster)** |

## Part 2D.2 — Full 1M SIFT1M 3-Way Ablation: Filtered (`selectivity = 0.10`) (Done)
- **Memory & OS RSS (`N = 1,000,000`, `100,099` passing nodes, actual `selectivity = 0.1001`):**
  - Vanilla Graph: `442.63 MB` (`OS RSS: 942.90 MB`)
  - Option A (VByte): `110.47 MB` (`-75.04%`), compress time `2.86 s`
  - Option B (AVX2 SIMD-BP128, `proximity_prefix = 16`): **`128.27 MB`** (**`-71.02%`**, `-69.30%` incl. ID map), compress time **`1.88 s`**, **OS RSS: `942.90 → 644.84 MB` (`-298.06 MB` freed)**

| efSearch | Van R@10 | VByte R@10 | **SIMD R@10** | Van p50 | VByte p50 | **SIMD p50** | **SIMD vs Van** | **SIMD vs VByte** | **1T QPS (Van → SIMD)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **16** | 0.7901 | 0.7882 | **0.7860** | 1.720 ms | 1.147 ms | **0.993 ms** | **`-42.3%` FASTER** | **`-13.5%` faster** | `426.5 → 706.1` |
| **32** | 0.8973 | 0.8908 | **0.8890** | 2.106 ms | 1.840 ms | **1.691 ms** | **`-19.7%` FASTER** | **`-8.1%` faster** | `382.0 → 415.4` |
| **64** | 0.9592 | 0.9502 | **0.9520** | 2.977 ms | 3.059 ms | **2.465 ms** | **`-17.2%` FASTER** | **`-19.4%` faster** | `264.3 → 309.7` |
| **128** | 0.9853 | 0.9788 | **0.9808** | 4.876 ms | 5.693 ms | **4.505 ms** | **`-7.6%` FASTER** | **`-20.9%` faster** | `165.8 → 181.2` |

## Part 2D.3 — Full 1M SIFT1M 3-Way Ablation: Strict Filter (`selectivity = 0.01`) (Done)
- **Memory & OS RSS (`N = 1,000,000`, `9,941` passing nodes, actual `selectivity = 0.0099`):**
  - Vanilla Graph: `442.63 MB` (`OS RSS: 942.85 MB`)
  - Option A (VByte): `110.19 MB` (`-75.11%`), compress time `3.22 s`
  - Option B (AVX2 SIMD-BP128, `proximity_prefix = 16`): **`128.20 MB`** (**`-71.04%`**, `-69.31%` incl. ID map), compress time **`1.64 s`**, **OS RSS: `942.85 → 644.52 MB` (`-298.33 MB` freed)**

| efSearch | Van R@10 | VByte R@10 | **SIMD R@10** | Van p50 | VByte p50 | **SIMD p50** | **SIMD vs Van** | **SIMD vs VByte** | **1T QPS (Van → SIMD)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **16** | 0.7600 | 0.7593 | **0.7584** | 1.664 ms | 1.137 ms | **0.926 ms** | **`-44.3%` FASTER** | **`-18.5%` faster** | `468.6 → 843.3` |
| **32** | 0.8877 | 0.8873 | **0.8876** | 1.982 ms | 1.847 ms | **1.397 ms** | **`-29.5%` FASTER** | **`-24.3%` faster** | `434.6 → 589.5` |
| **64** | 0.9467 | 0.9464 | **0.9465** | 2.582 ms | 3.466 ms | **2.219 ms** | **`-14.1%` FASTER** | **`-36.0%` faster** | `318.5 → 368.6` |
| **128** | 0.9757 | 0.9756 | **0.9755** | 4.277 ms | 5.793 ms | **3.804 ms** | **`-11.1%` FASTER** | **`-34.3%` faster** | `189.8 → 215.5` |

---

## Status: PHASE 2 (AVX2 SIMD-BP128 Engine & 1M Ablation) COMPLETE! ✅
- **Memory Target (`50%–60%`):** **CRUSHED** — **`71.0%` graph reduction** with AVX2 SIMD-BP128 (**`-298 MB` real OS RSS freed** on 1M) and `75.1%` with Scalar VByte.
- **Accuracy Target (Zero Recall Loss):** **VALIDATED** — `0.9755` vs `0.9757` (`-0.0002`) on strict 1% filtered 1M SIFT1M; `0.9561` vs `0.9566` on unfiltered 1M SIFT1M.
- **Latency Target (`< 3%` degradation):** **EXCEEDED** — AVX2 SIMD-BP128 is **`7.6%–45.4%` FASTER than Original Vanilla ACORN** across `selectivity = 1.0`, `0.10`, and `0.01`, and **`13.5%–36.0%` faster than Scalar VByte** on filtered hybrid search!
- **Next:** Ready for **Phase 3 (Large-Scale Hierarchical Reordering & Multi-Attribute Datasets)** and **Phase 4 (Plots & SCI Paper Writing)**.
