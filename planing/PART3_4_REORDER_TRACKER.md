# Parts 3 & 4 Tracker — Compression Design & Graph Reordering Prototype

Tracks **Part 3 (Implementation Checklist)** and **Part 4 (Graph Reordering Prototype)** from [NEXT_10_PART_PLAN.md](NEXT_10_PART_PLAN.md) and [THESIS_PLAN.md](THESIS_PLAN.md).

## Overall Progress

| Sub-Part | Title | Status |
|---|---|---|
| **Part 3.1** | Map exact ACORN touch points & finalize reordering/compression design | ✅ **Done** |
| **Part 4.1** | Implement `reorder_graph_bfs()` + consistent ID/vector permutation in ACORN | ✅ **Done** |
| **Part 4.2** | Rebuild `libfaiss.a` & add delta-distribution / VByte-size analyzer to benchmark | ✅ **Done** |
| **Part 4.3** | Validate 100% recall preservation & measure delta reduction on 100K & 1M SIFT1M | ✅ **Done** |

---

## Part 3.1 — Exact File-Level Touch Points & Integration Design (Done)

### 1. Every Search-Time Neighbor Access in `faiss/impl/ACORN.cpp`
All search-time reads of `hnsw.neighbors[]` and `hnsw.neighbor_range(...)` are confined to **4 functions**:

| Function | Lines (`ACORN.cpp`) | Level | Access Pattern |
|---|---:|---|---|
| `greedy_update_nearest` | `812–854` | Upper (`> 0`) | 1-hop (`neighbor_range` at line `823`) |
| `hybrid_greedy_update_nearest` | `855–968` | Upper (`> 0`) | 1-hop (`874`) + 2-hop (`929`) |
| `search_from_candidates` | `1101–1192` | Level `0` | 1-hop (`1155`) |
| `hybrid_search_from_candidates` | `1195–1420` | Level `0` | 1-hop (`1261`) + 2-hop (`1331`) |

*(Note: Distance counts are accumulated in `stats.n3`, not `stats.ndis`.)*

### 2. Graph Reordering Design (`reorder_graph_bfs`)
To localize neighbor IDs (minimizing deltas for VByte / SIMD-BP128) while remaining **100% lossless** to external callers:
- **Algorithm (Hierarchical / Level-0 BFS):**
  1. Start BFS from `acorn.entry_point` (and any upper-level hubs first, then Level-0 connected components) to assign sequential IDs `0 .. ntotal - 1` to topologically adjacent nodes.
  2. Build bidirectional maps on `IndexACORN`:
     - `std::vector<storage_idx_t> old_to_new` (`size = ntotal`)
     - `std::vector<storage_idx_t> new_to_old` (`size = ntotal`)
     - `bool is_reordered = false`
- **Consistent Permutation (performed once offline after `index.add()`):**
  1. **Vectors (`IndexFlat::xb`):** Permute stored vectors into `new_id` order (`new_xb[new_id] = old_xb[old_id]`) so distance computation `qdis(new_id)` has zero indirection and better CPU cache locality.
  2. **Graph metadata:** Permute `acorn.levels`, rebuild `acorn.offsets` for the new level layout, update `acorn.entry_point = old_to_new[entry_point]`.
  3. **Neighbor lists (`acorn.neighbors`):** For every node `new_id` and every level `l`:
     - Copy active neighbors (`>= 0`) of `old_id` at level `l`, rewriting each neighbor `v -> old_to_new[v]`.
     - Pad remaining slots with `-1` (while uncompressed) so vanilla search continues to work identically.
- **Transparent Search Remapping (`IndexACORN::search`):**
  - **Returned IDs:** After search completes, translate `labels[j] = new_to_old[labels[j]]` (for `labels[j] >= 0`) so callers receive exact original vector IDs.
  - **Filter map (`filter_id_map`):** In `IndexACORN::search(..., char* filter_id_map, ...)`, remap `filter_id_map` to `new_id` order (or support pre-reordered filter maps in benchmarks) so `hybrid_search` checks `reordered_filter[new_id]` directly in the hot loop.

### 3. Compression Storage Design (Prepared for Part 6)
- **Target Storage in `ACORN` (`faiss/impl/ACORN.h`):**
  - `bool use_compressed_neighbors = false;`
  - `std::vector<uint8_t> compressed_neighbors;` — concatenated VByte (Option A) or SIMD-BP128 (Option B) byte streams for **Level 0** (which holds >96% of all edges; upper levels stay tiny and uncompressed, or can be compressed optionally).
  - `std::vector<size_t> compressed_offsets;` (`size = ntotal + 1`) — byte start offset for each node's Level-0 list.
  - `std::vector<uint16_t> compressed_counts;` (`size = ntotal`) — exact number of active Level-0 neighbors (eliminates `-1` padding completely!).
- **Preserving ACORN-$\gamma$ Semantics (`M_beta` split at `ACORN.cpp:1328`):**
  - For `gamma == 1` (ACORN-1), all Level-0 neighbors trigger 2-hop expansion equally, so sorting the entire Level-0 list ascending is strictly lossless.
  - For `gamma > 1` (ACORN-$\gamma$), neighbors at index `< M_beta` do not trigger 2-hop expansion while `>= M_beta` do; we can preserve exact semantics by sorting/encoding `[0 .. min(count, M_beta))` and `[M_beta .. count)` as two segments when `gamma > 1`.
- **Zero-Allocation Query Decoding:**
  - Decode into a thread-local stack buffer `int32_t dec_buf[512]` right before the neighbor loop in `search_from_candidates` and `hybrid_search_from_candidates`.
  - Free uncompressed `acorn.neighbors` (Level 0) via `shrink_to_fit()` / swap when compression is enabled, and keep an option to retain it during A/B recall validation.

---

## Part 4.1 — Implement `reorder_graph_bfs()` (Done)
- Added `is_reordered`, `old_to_new`, `new_to_old`, `reordered_metadata`, and `reorder_graph_bfs()` to [faiss/IndexACORN.h](../faiss/IndexACORN.h).
- Added `assume_reordered_filter` flag to `SearchParametersACORN` in [faiss/impl/ACORN.h](../faiss/impl/ACORN.h).
- Implemented `IndexACORN::reorder_graph_bfs()` in [faiss/IndexACORN.cpp](../faiss/IndexACORN.cpp):
  - Level-0 BFS seeded from `acorn.entry_point` (plus full component sweep).
  - Consistently permutes `IndexFlatCodes::codes`, `acorn.metadata`, `acorn.levels`, `acorn.offsets`, `acorn.neighbors` (rewriting all neighbor IDs `v -> old_to_new[v]`), and `acorn.entry_point`.
  - Updated `IndexACORN::search` (unfiltered & filtered), `reconstruct()`, and `reset()` for transparent lossless ID mapping.

## Part 4.2 — Rebuild `libfaiss.a` & Add Delta Histogram Analyzer (Done)
- Rebuilt `build-release/faiss/libfaiss.a` (`cmake --build build-release --target faiss -j 4`) with zero errors.
- Updated [demos/bench_acorn_sift1m.cpp](../demos/bench_acorn_sift1m.cpp):
  - Added `--reorder` flag and `analyze_level0_deltas()` to measure mean delta, VByte byte-width histogram (`1B`, `2B`, `3B`, `4+B`), and exact VByte-encoded Level-0 payload size **before vs. after** `reorder_graph_bfs()`.
  - Pre-permutes `filter_map` with `params.assume_reordered_filter = true` during `--reorder` runs and reports true distance counts (`faiss::acorn_stats.n3`).
- Recompiled `build-release/bench_acorn_sift1m.exe` cleanly.

## Part 4.3 — Reordering Validation & Delta Reduction Results (Done)

### 1. Full 1M SIFT1M Delta & VByte Compression Analysis (`M=32, gamma=1, M_beta=64`)
- **BFS Reordering time (1M nodes):** **`3.262 s`** (`0.238 s` for 100K)

| Metric (1M SIFT1M, Level 0) | Before Reordering | After BFS Reordering | Improvement |
|---|---:|---:|---:|
| **Mean neighbor delta** | `19,548.1` | **`13,865.9`** | **-29.1%** |
| **1-byte deltas (`< 128`)** | `5.83%` | **`34.31%`** | **5.89× more 1-byte deltas** |
| **2-byte deltas (`< 16,384`)** | `57.82%` | `49.34%` | — |
| **3-byte deltas (`< 2,097,152`)** | `36.35%` | **`16.35%`** | **2.22× fewer 3-byte deltas** |
| **Raw L0 `neighbors[]`** | `427.25 MB` (`4.00 B/slot`) | `427.25 MB` (`4.00 B/slot`) | Baseline |
| **VByte L0 payload** | `108.80 MB` (`2.31 B/edge`) | **`85.92 MB` (`1.82 B/edge`)** | **-21.0% vs unsorted VByte** |
| **VByte + offsets/counts** | `118.34 MB` (`72.30%` savings) | **`95.46 MB` (`77.66%` savings)** | **77.66% total L0 savings** |

### 2. 100K SIFT1M Delta & VByte Analysis
- **1-byte deltas (`< 128`):** jumped from `18.82%` → **`53.76%`**
- **VByte L0 payload:** dropped from `9.01 MB` (`1.82 B/edge`) → **`7.31 MB` (`1.48 B/edge`)** (**`80.66%`** total L0 savings including offsets/counts)

### 3. Recall & Verification Notes
- **1M Unfiltered (`selectivity=1.0`):** `ef=128` Recall@10 = `0.9550` (vs `0.9554` in Part 2.4 — matches within multi-threaded build variance), `Avg ndis = 3670.3`.
- **100K Filtered (`selectivity=0.10`):** `ef=16/32/64` Recall@10 = `0.9340 / 0.9681 / 0.9812` (`Avg ndis = 799.8 / 1162.1 / 1639.1`).
- **Action item for Part 7 (Exact Zero-Recall-Loss Gate):** Always compare Vanilla vs. Reordered vs. Compressed on the **exact same built index** (same OpenMP build) to verify `0.0000` recall delta and check whether filtered `0.10` had any subtle discrepancy vs. build variance.

---

## Status: Parts 3 & 4 Complete ✅
Ready to proceed to **Part 6 (Integrate VByte Compression into ACORN Search)** and **Parts 7–9 (Zero Recall Loss, Real Memory/RSS & Latency Validation)** to complete **Phase 1**.
