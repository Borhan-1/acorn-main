# Master Paper Roadmap & Progress Tracker (ACORN-SIMD)

**Paper Title:** *Scalable and Memory-Efficient Hybrid Vector Search: Hierarchical Graph Reordering and Proximity-Preserving SIMD Delta Compression for ACORN*  
**Public Repository:** `https://github.com/Borhan-1/acorn-main` (Branch: `main`)  
**Target Venues:** *Information Systems* (Elsevier, SCI Q1) / *Future Generation Computer Systems* (Elsevier, SCI Q1) / *VLDB Journal* / *IEEE TKDE*

> **How to use this file:** This is the single master checklist of **everything completed so far** (small and big) and **every planned future upgrade** to maximize SCI Q1 acceptance probability. Whenever any future task is completed, its status here is updated to `✅ Completed` with exact commit hash and measured numbers.

---

## Overall Status Dashboard

| Track / Phase | Scope | Hardware Needed | Status |
| :--- | :--- | :--- | :---: |
| **Phase 0–1** | Baseline ACORN, Flat/Hierarchical Reordering, Scalar VByte (Option A), Upstream Bug Fixes | 8 GiB Laptop | ✅ **Completed** |
| **Phase 2** | Base-Separated 16-Delta AVX2 `SimdBP128Codec`, Two-Segment Split, Zero-Alloc Stack Decode, `l0_freed` | 8 GiB Laptop | ✅ **Completed** |
| **Phase 3** | `1M SIFT1M`, `GloVe-400K`, Dense `ACORN-γ`, 4T OpenMP QPS, Cluster-Gapped Dynamic `add()` | 8 GiB Laptop | ✅ **Completed** |
| **Phase 3.5** | Real Correlated / Compound `AND` / Zipfian Predicates, 6-Selectivity Crossover, `P` Sweep, `DBpedia-OpenAI` (`1,536-d`) | 8 GiB Laptop | ✅ **Completed** |
| **Phase 4** | 6 Camera-Ready Figures (PNG+PDF), Formal Proofs (Props 1–4), LaTeX Tables, Full Manuscript (`Rev 1–4`) | 8 GiB Laptop | ✅ **Completed** |
| **Step A** | Live GitHub Repo Sync, Section 7 URL Fix, Peak Permutation RAM Clarification, Fixed-Point README Table | 8 GiB Laptop | ✅ **Completed** |
| **Track B** | **High-ROI Reviewer-Proof Upgrades:** `B.1` Elias-Fano (✅ Done), `B.2` 5-Run 95% CI (✅ Done); `B.3` Memory–Latency Pareto, `B.4` In-Place Permutation, `B.5` Index Serialization | 8 GiB Laptop | 🔄 **In Progress (2/5)** |
| **Track C** | **Large-Scale Server Validation:** `10M` & `100M` Datasets, Exact Server Ground Truth, `4–64` Thread Scaling | 128 GiB+ Linux Server | ⏳ **Planned** |
| **Track D** | **Submission & Dissemination:** Journal LaTeX Template, arXiv Preprint, Zenodo DOI, Journal Submission | Any | ⏳ **Planned** |

---

## Part I — Completed Work (Everything Done So Far) ✅

### 1. Core C++ Engine & Upstream Bug Fixes (Phases 0–2)
- [x] **Upstream Memory-Safety Bug Fixes (`faiss/impl/ACORN.cpp`):**
  - Fixed heap buffer overflow `ep_per_level(max_level)` $\to$ `ep_per_level(max_level + 1)` in `ACORN::add_with_locks` (`ACORN.cpp:1078`).
  - Isolated upper-layer traversal (`l0_freed = true`) to avoid upstream `add_link` boundary spillover (`ACORN.cpp:695`).
- [x] **Hierarchical Multi-Layer Graph Reordering (`faiss/IndexACORN.cpp`):**
  - Implemented `reorder_graph_bfs()` (flat BFS baseline) and `reorder_graph_hierarchical()` (`O(N/M + B_loc)` queue workspace).
  - Measured on `1M SIFT1M`: peak BFS queue dropped **`32.0×`** (`3,906.25 KiB → 122.09 KiB`), reordering speed improved **`1.93×`** (`2.59 s → 1.34 s`), and 1-byte deltas (`<128`) jumped **`5.89×`** (`5.83% → 34.31%`).
- [x] **Scalar VByte Ablation Codec (`faiss/impl/codec/vbyte_codec.h`):**
  - Implemented Option A baseline (`compress_neighbors_vbyte`) to isolate compression benefits from SIMD decoding overhead (`7.94 ns/edge`).
- [x] **Base-Separated 16-Delta Mini-Block AVX2 Codec (`faiss/impl/codec/simd_bp128.h`):**
  - Decoupled VByte base $z_0$ + 16-delta / 8-delta AVX2 bit-packing (`Avx2Lut` with `_mm256_shuffle_epi8`, `_mm256_srlv_epi32`, `_mm256_and_si256`, `avx2_prefix_sum`) + bit-compatible scalar fallback.
  - Exhaustive unit test suite (`tests/test_simd_bp128.cpp`): **`3,799/3,799`** bit-exact cases passed on both AVX2 and scalar builds; `104M`-edge microbenchmark achieved **`1.58 ns/edge`** (**`5.02×`** faster than scalar `VByte`).
- [x] **Proximity-Preserving Two-Segment Split & Zero-Alloc Execution (`faiss/impl/ACORN.cpp`):**
  - Split Level-0 lists at $s(u) = M_\beta$ (`ACORN-γ`) or $s(u) = P$ (`ACORN-1`, default `P=4..8`) stored in `uint16_t compressed_l0_prefix_counts[N]`.
  - Replaced heap allocations in 2-hop search with stack buffers (`dec_buf1[1024]`, `dec_buf2[1024]`).
  - Implemented `l0_freed = true` heap reclamation via vector swap, returning `298.38 MiB` OS RSS on `1M SIFT1M`.
- [x] **Cluster-Gapped Dynamic `add()` & Batch Re-Compaction (`faiss/IndexACORN.cpp`):**
  - Implemented `enable_dynamic_gaps()`, `add_dynamic()`, and `compact_dynamic_l0()` with 1 B/node `has_overflow` guard.
  - Measured on `1M SIFT1M` (`+50,000` live inserts, `ρ=0.05`): **`1,897.8 inserts/s`**, `75.36%` primary BFS cluster hit rate, `R@10 = 0.9800`, and **`1.953 s`** re-compaction restoring `-69.72%` compression.

### 2. Multi-Domain Experimental Suite (Phases 3 & 3.5)
- [x] **SIFT1M (`1M × 128-d`, Vision, Euclidean $L_2$):**
  - Graph RAM: `442.63 MiB → 130.46 MiB` (**`-70.53%`** vs. ACORN, **`-49.73%`** vs. HNSW `259.52 MiB`, `2.20 B/edge`).
  - `σ=1%, ef=128`: `4.277 → 3.804 ms` (**`-11.1%`**, `ef=32`: **`-29.5%`**), `R@10 = 0.9755` (`7.56×` HNSW).
  - Full 6-selectivity Pre-Filter (`Bitmap-Scan` & `Inverted-List`) crossover map across `σ ∈ {0.1%, 1%, 5%, 10%, 20%, 100%}`.
  - Multi-thread OpenMP QPS (`σ=10%, ef=128`): **`+21.4%`** at 4 threads (`297.2 → 360.8 QPS`).
- [x] **SIFT100K Dense `ACORN-γ` (`γ=12`, 77% Slot Occupancy):**
  - Graph RAM: `48.68 MiB → 20.85 MiB` (**`-57.16%`** vs. ACORN, **`-19.68%`** vs. HNSW, `1.50 B/edge`); `σ=5%, ef=64`: `0.851 → 0.839 ms` (`-1.3%` parity, `ef=32`: `-24.7%`).
- [x] **GloVe-400K (`399K × 100-d`, NLP, Cosine):**
  - Graph RAM: `176.61 MiB → 46.92 MiB` (**`-73.43%`** vs. ACORN, **`-54.69%`** vs. HNSW, `1.99 B/edge`).
  - Evaluated 5 real/skewed predicates (`word_len >= 16`, `word_len >= 14`, compound `word_len >= 10 AND zipf_rank < 20000`, Zipfian tail `0.36%`, Zipfian head `23.41%`).
- [x] **DBpedia-OpenAI (`100K × 1,536-d`, LLM RAG, Cosine):**
  - Graph RAM: `44.27 MiB → 11.92 MiB` (**`-73.07%`** vs. ACORN, **`-54.08%`** vs. HNSW, `1.72 B/edge`).
  - `σ=1%, ef=128`: `4.236 → 4.149 ms` (`-2.0%` parity, `ef=32`: **`-35.0%`**), `R@10 = 0.9783` (`7.91×` HNSW); `4.4×–7.0×` faster than Pre-Filter at `σ=10%`; real range `entity_text_len >= 500`.
- [x] **Multi-Prefix Sensitivity Study (`--sweep_prefix 0,4,8,12,16,32`):**
  - Empirically validated Proposition 3's closed-form $k^*(\sigma)$ optimum on `1M SIFT1M` (`σ=20%`, `4.1×` recall-gap reduction at `P=4`) and `GloVe-400K` Zipf Head (`5.3×` recall-gap reduction at `P=4`).

### 3. Theory, Figures, Manuscript & Public Artifact (Phase 4 & Step A)
- [x] **6 Camera-Ready Figures (`paper_figures/fig1..fig6.(png|pdf)`):** Generated at 300-DPI PNG + vector PDF via `benchs/generate_thesis_plots.py`.
- [x] **Formal Proofs (`planing/THEORETICAL_ANALYSIS.md`):** Propositions 1–4 (Space & Base-Separation, Chernoff Queue Bound, Wald Stopping Horizon & Unimodal Capture $\Phi(P;r)$, Dynamic Compaction) + Cache-Line Bandwidth Model.
- [x] **Camera-Ready LaTeX Tables (`paper_figures/tables.tex`):** Tables 1–6 formatted with `booktabs` + `multirow`.
- [x] **Full SCI Q1 Manuscript Draft (`planing/MANUSCRIPT_DRAFT.md`):** `9,400+` words, Sections 1–8 + 24 verified peer-reviewed references, hardened across `Rev 1–4` (academic tone, formal proofs, variance transparency, reproducibility section).
- [x] **Step A Polish & Public GitHub Release (`bf5b0cc`):**
  - Clean public repo synced at `https://github.com/Borhan-1/acorn-main` (datasets/binaries excluded via `.gitignore`).
  - Fixed `<author>` placeholder in Section 7, clarified $O(N/M + B_{\text{loc}})$ queue workspace vs. out-of-place `new_codes` RAM in Sections 4.2 & 6.4, and updated `README.md` Headline Table to fixed `(σ, efSearch)` operating points.

---

## Part II — Future Roadmap to Make the Paper Stronger 🚀

### Track B — High-Impact Upgrades (Runnable on Current 8 GiB Laptop 💻)
These upgrades directly neutralize the top reviewer objections **without needing a 128 GiB server**:

- [x] **B.1 — Elias-Fano Compressed-Adjacency Baseline (`✅ Completed`):**
  - **Implemented:** `faiss/impl/codec/elias_fano_codec.h` (Base-Separated Elias-Fano + 64-bit `ctzll` decoder), `tests/test_elias_fano.cpp` (`3,275` exhaustive cases + `16` two-segment splits passed; `5.89×` slower decode than AVX2 SIMD-BP128 at `19.94` vs. `3.38 ns/edge`), wired into `ACORN.h`, `IndexACORN.h/.cpp`, and `bench_acorn_sift1m.cpp` (`--codec all`).
  - **Measured 4-Way Results (`σ=1%`):** Across `SIFT1M` (`126.80` vs. `130.66 MiB`), `GloVe-400K` (`45.82` vs. `46.90 MiB`), and `DBpedia-OpenAI` (`11.58` vs. `11.94 MiB`), Elias-Fano saves only `0.6–0.9%` more graph RAM while running **`15.8%–25.3%` slower** than AVX2 SIMD-BP128 across every dataset and `efSearch` setting (`ef=128` `p50`: `5.040 → 4.066 ms`, `4.590 → 3.429 ms`, `4.209 → 3.319 ms`).
  - **Manuscript Updated:** Added to Section 5.1, Section 5.3 (4-way comparison table), and Reference `[25]` (Ottaviano & Venturini, SIGIR 2014).
- [x] **B.2 — Multi-Run Statistical Variance & 95% Confidence Intervals (`✅ Completed`):**
  - **Implemented:** Added `--runs <K>` with exact Student-$t$ 95% CI ($t_{0.975, 4} = 2.776$ for 5 runs) to `demos/bench_acorn_sift1m.cpp`.
  - **Measured 5-Run Results (`--runs 5 --codec all`, `σ=1%, ef=128`):**
    - `SIFT1M`: Vanilla `4.177 ± 0.047 ms` vs. VByte `6.054 ± 0.036 ms` vs. Elias-Fano `4.982 ± 0.040 ms` vs. **SIMD-BP128 `4.052 ± 0.054 ms`** (`-3.0%` vs. Van, `-18.7%` vs. EF).
    - `GloVe-400K`: Vanilla `3.257 ± 0.044 ms` vs. VByte `5.065 ± 0.102 ms` vs. Elias-Fano `4.521 ± 0.040 ms` vs. **SIMD-BP128 `3.460 ± 0.023 ms`** (`-23.5%` vs. EF, `-31.7%` vs. VByte).
    - `DBpedia-OpenAI`: Vanilla `3.427 ± 0.095 ms` vs. VByte `4.253 ± 0.133 ms` vs. Elias-Fano `4.354 ± 0.016 ms` vs. **SIMD-BP128 `3.311 ± 0.035 ms`** (`-3.4%` vs. Van, `-23.9%` vs. EF).
  - **Docs Updated:** `MANUSCRIPT_DRAFT.md` (Sec 5.1 & 5.3) and `README.md` Headline Results table updated with 5-run 95% CIs.
- [ ] **B.3 — Dedicated Memory vs. Latency Pareto Frontier Plot:**
  - **Why:** Visually demonstrates the exact space–time dominance of `AVX2 SIMD-BP128` over `Vanilla ACORN`, `Standard HNSW`, `Scalar VByte`, and `Elias-Fano` at a fixed recall target.
  - **Scope:** Add `fig7_memory_latency_pareto.(png|pdf)` to `benchs/generate_thesis_plots.py` and embed in Section 5.
- [ ] **B.4 — In-Place Cycle-Leader Permutation (`apply_acorn_permutation`):**
  - **Why:** Eliminates the temporary `new_codes` copy (`+47.7 GiB` peak RAM at `100M`), reducing extra vector RAM during reordering from $O(N \cdot d)$ to $O(1)$ and strengthening the memory-efficiency claim.
  - **Scope:** Replace out-of-place `new_codes` allocation in `faiss/IndexACORN.cpp` with in-place cycle-leader permutation using a compact `std::vector<uint8_t> visited(n)` (`N` bytes = `100 MB` at `100M`), and verify bit-exactness on `1M SIFT1M`.
- [ ] **B.5 — Index Serialization (`Save / Load` to Disk):**
  - **Why:** Essential prerequisite for `10M–100M` experiments so the graph is built once, saved to disk, and queried across multiple selectivities/predicates without re-building.
  - **Scope:** Implement binary save/load for `IndexACORNFlat` (uncompressed + compressed L0 + permutation maps) with `--save_index` / `--load_index` flags in `demos/bench_acorn_sift1m.cpp`.

---

### Track C — Large-Scale Server Experiments (`10M–100M` on High-RAM Linux Server 🖥️)
Requires a Linux server with **128 GiB+ RAM** and **32–64 CPU cores** (plus BLAS/GPU for exact ground truth):

- [ ] **C.1 — 10M Scale Validation (`SIFT10M` or `DEEP10M`):**
  - **Why:** Closes the "only tested up to 1M" critique with manageable build time (`10×` faster than `100M`) and `~10 GiB` total RAM.
  - **Scope:** Run all baselines (`Vanilla ACORN`, `HNSW`, `VByte`, `Elias-Fano`, `AVX2 SIMD-BP128`) on the same server, log exact build time + peak RSS, and report 3–5 run CIs.
- [ ] **C.2 — 100M Scale Validation (`SIFT100M` / `BIGANN-100M`):**
  - **Why:** Upgrades Table 6.1's `100M` row from *Projected* to **Measured** (`~91 GiB → ~60 GiB` total RAM), a major win for *VLDB Journal* / *IEEE TKDE*.
  - **Scope:**
    - Build once with `32–64` threads and serialize to disk (`B.5`); log exact build time & peak RAM.
    - Compute exact filtered ground truth via BLAS/GPU for `1,000` queries across selectivities (`1%, 5%, 10%`).
    - Explicitly frame claims as *"real correlated predicates at 1M scale + synthetic predicates at 100M scale"*, keeping `200M` and `500M` labeled as analytical extrapolations.
    - Re-run `1M` baselines on the same server if directly comparing latency across scales.
- [ ] **C.3 — Multi-Core Server QPS Scaling (`4, 8, 16, 32, 64` Threads):**
  - **Why:** Proves that our memory-bandwidth QPS advantage (`+21.4%` at 4T) holds or widens on modern 32–64-core server memory controllers.
  - **Scope:** Generate a thread-scaling curve (`4, 8, 16, 32, 64` threads) comparing `Vanilla ACORN` vs. `AVX2 SIMD-BP128` and add to Figure 6.

---

### Track D — Submission & Dissemination Package 📄
- [ ] **D.1 — Journal LaTeX Camera-Ready Conversion:**
  - Convert `planing/MANUSCRIPT_DRAFT.md` + `paper_figures/tables.tex` + vector PDFs into the target journal template (`elsarticle` for *Information Systems* / *FGCS*, or `IEEEtran` for *TKDE*), verify zero LaTeX warnings/overfull boxes.
- [ ] **D.2 — arXiv Preprint Upload:**
  - Compile clean single-column/two-column PDF and upload to arXiv (`cs.DB` / `cs.IR`) to establish priority and gather early community feedback.
- [ ] **D.3 — Zenodo Archival DOI Release:**
  - Tag `v1.0` on `https://github.com/Borhan-1/acorn-main`, archive on Zenodo, and insert the permanent DOI into Section 7.
- [ ] **D.4 — Journal Submission & Peer-Review Revision Log:**
  - Submit to primary target venue (*Information Systems* / *FGCS* / *VLDBJ*) and track reviewer responses in a dedicated rebuttal document.

---

## Reference Index of Existing Detailed Trackers
- [THESIS_PLAN.md](THESIS_PLAN.md) — Original MSc thesis blueprint
- [PART2_BASELINE_TRACKER.md](PART2_BASELINE_TRACKER.md), [PART3_4_REORDER_TRACKER.md](PART3_4_REORDER_TRACKER.md), [PART6_9_VBYTE_TRACKER.md](PART6_9_VBYTE_TRACKER.md) — Phase 1 logs
- [PHASE2_SIMD_TRACKER.md](PHASE2_SIMD_TRACKER.md) — AVX2 `SimdBP128Codec` development & unit tests
- [PHASE3_SCALING_TRACKER.md](PHASE3_SCALING_TRACKER.md) — 1M SIFT1M, GloVe-400K, Dense ACORN-γ, 4T QPS, Dynamic `add()`
- [PHASE3_5_Q1_POLISH_TRACKER.md](PHASE3_5_Q1_POLISH_TRACKER.md) — Real/Zipf predicates, Pre-Filter crossover, `P` sweep, DBpedia-OpenAI
- [PHASE4_PAPER_TRACKER.md](PHASE4_PAPER_TRACKER.md) — Figures, proofs, tables, manuscript, `Rev 1–4`, `Step A.1–A.3`
