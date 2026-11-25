# ACORN: Paper & Codebase Analysis

> **Paper:** *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data*  
> **Authors:** Liana Patel, Peter Kraft, Carlos Guestrin, Matei Zaharia (Stanford)  
> **Venue:** SIGMOD '24 — [arXiv:2403.04871](https://arxiv.org/abs/2403.04871)  
> **Code:** Built on [FAISS](https://github.com/facebookresearch/faiss) (C++)

---

## 1. Problem Statement

Modern applications increasingly store **mixed-modality data**: vector embeddings (images, text, video) *plus* structured attributes (categories, prices, tags, dates). The core challenge is **hybrid search**: find the top-*k* nearest vectors *subject to a predicate filter* (e.g., "find similar images where `color == red`").

### Why Existing Methods Fail

| Approach | Problem |
|---|---|
| **Post-filtering** (HNSW + filter after search) | Recall collapses under high-selectivity predicates |
| **Pre-filtering** (filter first, then brute-force ANN) | Slow — defeats the purpose of an ANN index |
| **Attribute-aware graph indices** (HQANN, ADBV, NHQ) | Restricted to *equality* predicates or *small cardinality* predicate sets; don't generalize |
| **IVF with selective scan** | Poor recall/throughput tradeoff at all selectivities |

> [!IMPORTANT]
> No prior method was both **performant** AND **predicate-agnostic** — supporting arbitrary predicates (equality, range, regex, bitset, complex boolean expressions) across varying selectivity levels.

---

## 2. Key Contributions

1. **ACORN** — a hybrid search index that is:
   - *Predicate-agnostic*: works with any predicate expressible as a boolean filter map
   - *Performant*: 2–1,000× higher throughput than prior methods at fixed recall
   - Built on top of HNSW, so it inherits its scalability

2. **Predicate Subgraph Traversal** — a new *search strategy* concept that traverses only the subgraph of nodes satisfying the predicate, emulating an ideal (but impractical) index

3. **Two algorithm variants**:
   - **ACORN-1**: predicate-agnostic (works for any selectivity without tuning)
   - **ACORN-γ**: tuned for low-selectivity predicates, uses larger neighborhoods

4. **ACORN Construction Algorithm** — builds a modified HNSW graph where each node stores γ·M neighbors at level 0 (instead of M), making the predicate subgraph better connected

---

## 3. Core Technical Design

### 3.1 The Predicate Subgraph Idea

The key insight: given a predicate *P*, define the **predicate subgraph** *G_P* = subgraph induced by all nodes satisfying *P*. 

- The ideal (impractical) approach: build a separate HNSW index for every possible predicate subgraph → exponential space
- ACORN's insight: if the **base HNSW graph is dense enough**, traversal through non-matching nodes can still navigate to matching ones efficiently

### 3.2 ACORN-1 (Predicate-Agnostic Baseline)

**Construction**: Standard HNSW construction, but during the level-0 neighbor search, it **expands to neighbors-of-neighbors** to find γ·M candidates, then selects M best. This pre-computes 2-hop reachability.

**Search** (level 0 `hybrid_search_from_candidates`):
```
For each candidate v0 popped from queue:
  For each neighbor v1 of v0:
    if v1 passes filter → compute distance, add to results
    if fewer than M neighbors found → expand to v1's neighbors (2-hop)
      For each v2 in neighbors(v1):
        if v2 passes filter → compute distance, add to results
```
This is the **predicate subgraph traversal**: if no direct neighbors satisfy the filter, it looks 1 hop further to find ones that do.

### 3.3 ACORN-γ (High-Selectivity Tuning)

**Construction**: Each level-0 node stores **γ·M neighbors** (instead of M). The γ parameter multiplies the neighborhood to cover more of the graph locally.

**Construction key parameter** (from `ACORN.cpp` line 125–134):
```cpp
efConstruction = M * gamma;  // search wider during construction
// Level 0 nodes get M_beta + 1.5*M neighbor slots (base slots)
// Level 0 also includes γ·M = gamma * M effective neighbors
```

**Neighbor Selection** — a modified `shrink_neighbor_list`:
- First M_beta neighbors: standard closest-first selection
- After M_beta: use **neighbors-of-neighbors pruning** — skip a candidate if it's already a neighbor of an accepted neighbor (promotes graph diversity/connectivity)

### 3.4 Key Parameters

| Parameter | Role |
|---|---|
| `M` | Base number of neighbors per node (like HNSW's M) |
| `gamma` | Neighborhood multiplier (ACORN-γ: γ > 1; ACORN-1: γ = 1) |
| `M_beta` | Threshold after which neighbor-of-neighbor pruning kicks in (level 0) |
| `efSearch` | Search expansion factor (higher = better recall, lower = faster) |
| `efConstruction` | Construction expansion = M × γ |

---

## 4. Implementation Architecture

```
ACORN (repo, forked from FAISS)
├── faiss/
│   ├── IndexACORN.h / .cpp      ← Public index interface (extends faiss::Index)
│   └── impl/
│       ├── ACORN.h              ← Core data structure + algorithm declarations
│       └── ACORN.cpp            ← Full implementation (1,694 lines)
└── benchs/                      ← Benchmarking scripts
```

### Class Hierarchy

```
faiss::Index
    └── faiss::IndexACORN         (IndexACORN.h/.cpp)
            └── faiss::IndexACORNFlat   (stores raw float vectors)

faiss::ACORN                      (impl/ACORN.h/.cpp)
    ├── MinimaxHeap               (bounded priority queue for search)
    ├── NodeDistCloser/Farther    (pair sorting helpers)
    └── ACORNStats                (search/construction telemetry)
```

### Data Structures (`ACORN` struct)

```cpp
vector<int>        levels;          // level assigned to each node
vector<size_t>     offsets;         // offsets[i]: start of node i's neighbor list
vector<NeighNode>  neighbors;       // flat adjacency list (int32_t node IDs)
const int*         metadata;        // pointer to attribute array (one int per node)
int                gamma;           // neighborhood multiplier
int                M, M_beta;       // capacity parameters
```

> [!NOTE]
> The neighbor list is stored as a single flat array with `offsets[]` for O(1) random access — same as FAISS's HNSW. Metadata is stored as a separate `int[]` array indexed by node ID.

### Filter Interface

Predicates are represented as a **flat boolean bitmap** (`char* filter_id_map`), one byte per database vector per query:

```cpp
// Caller pre-computes which vectors satisfy the predicate:
filter_ids_map[query_idx * N + vector_idx] = (bool)(metadata[vector_idx] == aq[query_idx]);

// Then calls:
acorn_gamma.search(nq, xq, k, distances, labels, filter_ids_map.data());
```

This is what makes ACORN **predicate-agnostic** — the filter evaluation is completely external. Any predicate computable as a boolean mask is supported.

### Search Flow

```
hybrid_search() [ACORN.cpp:1508]
  │
  ├── Upper levels (l > 0):
  │     hybrid_greedy_update_nearest()
  │       → greedy walk, checking filter_map
  │       → if γ=1: also checks neighbors-of-neighbors (2-hop)
  │
  └── Level 0:
        hybrid_search_from_candidates() [ACORN.cpp:1194]
          → BFS from nearest entry point
          → For each candidate v0:
              → scan neighbors v1:
                  if filter_map[v1] → compute dist, push to heap
                  if j >= M_beta (or γ=1) → scan v1's neighbors v2:
                      if filter_map[v2] → compute dist, push to heap
              → stopping condition: efSearch candidates processed
```

---

## 5. Experimental Results (from paper abstract + paper context)

| Method | Throughput (QPS) | Recall@10 | Notes |
|---|---|---|---|
| **ACORN-γ** | Highest | ≥ 0.9 | All datasets, all selectivities |
| **ACORN-1** | 2–10× better than baselines | ≥ 0.9 | Predicate-agnostic, no tuning |
| HQANN | ~100-1000× lower QPS | same recall | Only equality predicates |
| NHQ | Lower QPS | same recall | Small predicate sets only |
| Post-filter HNSW | Fails at low selectivity | recall → 0 | — |
| Pre-filter (brute-force) | Very low QPS | 1.0 | Always correct, always slow |

> **Key result**: ACORN achieves **2–1,000× higher throughput at fixed recall** compared to prior methods, across both low and high selectivity benchmarks.

### Datasets tested:
- **Filtered-DiskANN benchmarks** (standard, low-cardinality equality predicates)
- **Complex multi-modal datasets** (LAION image-text with arbitrary predicates)
- Prior methods fail on complex datasets entirely; ACORN handles them natively

---

## 6. Strengths

- ✅ **Truly predicate-agnostic**: filter is a black-box boolean function; no algorithm changes needed
- ✅ **HNSW-based**: inherits logarithmic construction and search complexity, proven scalability
- ✅ **Simple integration**: extends FAISS, minimal API changes from standard HNSW usage
- ✅ **No index rebuild**: same index works for different predicates at query time
- ✅ **Parallelizable**: uses OpenMP for construction and search

---

## 7. Limitations & Design Notes

> [!WARNING]
> **Memory overhead**: ACORN-γ stores γ times more neighbors at level 0. With γ=12 and M=32, level 0 stores ~384 neighbors per node vs. 64 in standard HNSW. This significantly increases index size.

> [!CAUTION]
> **Filter bitmap cost**: The `filter_id_map` is sized `nq × N` (queries × dataset size). For 10M vectors and 1000 queries, this is 10 GB — not practical without batching. The current implementation appears to assume small query batches.

- **Single-integer metadata only** in the current code (`const int* metadata`). The paper discusses arbitrary predicates, but the reference implementation uses a pre-computed bitmap to achieve predicate agnosticism at the cost of precomputation.
- **Code has TODOs**: Several `// TODO` comments remain (e.g., `"TODO change to hybrid"` in `IndexACORN.h`, `"TODO figure out what do with this"` in `IndexACORN.cpp`). The code appears research-grade.
- **No Python bindings** for ACORN specifically (FAISS Python bindings exist, but ACORN's overloaded `search(filter_id_map)` isn't wrapped in the Python layer).
- **`fill_with_random_links`** is unimplemented (`throw FaissException("UNIMPLEMENTED")`).
- **Unbounded queue search path** is also unimplemented (`"UNIMPLEMENTED search unbounded queue"`).

---

## 8. Code Quality Notes

| Aspect | Observation |
|---|---|
| **Style** | Follows FAISS conventions; clang-format config present |
| **Debugging** | Extensive `debug()`/`debug_search()` macros with env-var controlled levels |
| **Profiling** | `ACORNStats` tracks timing for `candidates_loop`, `neighbors_loop`, `tuple_unwrap`, `skips`, `visits` |
| **Thread safety** | Uses `omp_lock_t` per-node for concurrent construction |
| **Commented code** | Many commented-out blocks (e.g., old regex-based predicate support) suggest iterative development |

---

## 9. Summary

ACORN is an elegant engineering solution to a hard practical problem. The core insight — that increasing neighborhood size during construction makes the predicate subgraph navigable without pre-building per-predicate indices — is simple and powerful. The key design choices are:

1. **Externalize predicate evaluation** → predicate agnosticism via filter bitmap
2. **Over-provision neighborhoods** (γ·M) → denser graph → better subgraph connectivity
3. **2-hop expansion during search** → compensate for sparse matching at query time
4. **HNSW as backbone** → logarithmic complexity, industrial-grade implementation base

The tradeoff is **memory** (γ× larger index) and **filter bitmap computation cost** (O(N) per query), which is acceptable for batch workloads but needs attention for real-time single-query applications.
