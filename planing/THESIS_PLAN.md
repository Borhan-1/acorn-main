# 🎓 MSc Thesis Research & Execution Plan

**Topic:** Scalable & Memory-Efficient Hybrid Vector Search: Graph Reordering and SIMD-Accelerated Delta Compression for ACORN  
**Target Venue:** SCI-Indexed Journal (IEEE TKDE / VLDB Journal / Information Systems) or Top Tier Conference (SIGMOD / VLDB / ICDE)  
**Author:** Muhammad Borhan  
**Framework:** ACORN (FAISS C++ Backend)  

---

## 1. Executive Summary & Research Motivation

### 1.1 Background & The Hybrid Search Challenge
Modern enterprise search engines, recommendation systems, and Retrieval-Augmented Generation (RAG) pipelines require **Hybrid Vector Search** — retrieving top-$k$ nearest high-dimensional embeddings subject to arbitrary structured metadata predicates (e.g., `price < 50 AND category == 'electronics'`).

In SIGMOD 2024, Stanford researchers introduced **ACORN** (*Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data*). ACORN achieves orders-of-magnitude higher throughput than prior methods by over-provisioning neighbors at Level 0 ($\gamma \cdot M$ edges) and employing **Predicate Subgraph Traversal** with 2-hop search expansions.

### 1.2 The Critical Bottleneck (Memory Explosion)
While ACORN solves the recall collapse of hybrid search, it incurs an **extreme memory penalty**:
* In standard HNSW ($M=32$), Level 0 stores $2M = 64$ neighbors per node.
* In ACORN ($M=32, \gamma=1, M_\beta=64$), Level 0 stores $M_\beta + 1.5M = \mathbf{112}$ neighbors per node. For higher $\gamma$ (e.g. $\gamma=12$), it stores up to $384+$ neighbors per node.
* Each neighbor ID is stored as a 32-bit integer (`int32_t` = 4 bytes).

**For 500 Million (50 কোটি) vectors:**
$$\text{Storage} = 500,000,000 \times 112 \times 4 \text{ bytes} \approx \mathbf{224\text{ GB}}$$
This graph footprint exceeds the RAM capacity of standard enterprise instances, dramatically driving up infrastructure costs.

### 1.3 Core Thesis Hypothesis
> *"By exploiting graph topology locality via Graph-Aware Reordering and applying SIMD-accelerated delta compression on neighbor lists, we can compress ACORN's graph storage by **50% to 60%** (reducing 4 bytes/neighbor to ~1.8 bytes) while maintaining **100% search accuracy (Zero Recall Loss)** and keeping search latency degradation within **< 3%**."*

---

## 2. Theoretical Methodology & System Design

The proposed system introduces an end-to-end compression and search pipeline seamlessly integrated into ACORN's C++ core:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Standard ACORN Index Construction                        │
│    Builds base multi-layer graph with predicate metadata    │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 2. Graph-Aware ID Reordering (Topology Localization)        │
│    HNSW Hierarchical Traversal / Localized BFS              │
│    Transforms scattered IDs into clustered sequential IDs   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 3. Neighbor Sorting & Delta Transformation                  │
│    Ascending sort: [v_1, v_2, ..., v_k]                     │
│    Compute deltas: Δ_1 = v_1, Δ_i = v_i - v_{i-1}           │
└──────────────────────────────┬──────────────────────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
┌────────────▼───────────────┐     ┌─────────────▼──────────────┐
│ Option A (Baseline)        │     │ Option B (Proposed Core)   │
│ Byte-Aligned VByte Encode  │     │ SIMD-BP128 / FastPFor      │
│ (Compact, Scalar Decoded)  │     │ (Bit-packed, AVX2 Decoded) │
└────────────┬───────────────┘     └─────────────┬──────────────┘
             │                                   │
             └─────────────────┬─────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 4. Search-Time On-The-Fly L1-Cache Decompression            │
│    Decompresses candidate neighbors into stack buffer       │
│    Zero heap allocation during distance calculations        │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Graph-Aware ID Reordering (Tackling the 500M Scale)
On 500M nodes, a single global BFS queue causes Out-of-Memory (OOM) and cache thrashing. We employ **Hierarchical HNSW-Guided Reordering**:
1. Leverage HNSW's natural hierarchy: Upper layers (Level 1, 2, ...) already act as geometric centroids.
2. Top layers are ordered first.
3. Level-0 subgraphs are traversed using localized BFS bounded by parent centroids.
4. **Result:** Neighbors stay tightly clustered within local ranges ($[1, 4000]$), minimizing delta sizes.

### 2.2 Delta Encoding & Dual Compression Engines
* **Option A: Variable-Byte (VByte) — Comparative Baseline:**
  * Byte-aligned format. Deltas $< 128$ use 1 byte; $< 16,384$ use 2 bytes.
  * Easy to implement, serves as the scalar ablation baseline.
* **Option B: SIMD-BP128 (Bit-Packing 128 integers) — The SCI Contribution:**
  * Packs chunks of 128 integer deltas into the exact minimum bit-width required:
    $$b = \lceil \log_2(\max(\Delta_1, \dots, \Delta_{128}) + 1) \rceil$$
  * Utilizes Intel/AMD AVX2 / AVX-512 vector registers to decompress 128 neighbor IDs in **< 15 nanoseconds**.

### 2.3 On-The-Fly Query Decompression
* Decompression is triggered **only** when an unvisited node is popped from the search candidate priority queue.
* The decoded neighbors are written into a tiny thread-local stack buffer (`uint32_t neighbor_buf[128]`).
* No dynamic `malloc`/`free` calls during search $\implies$ near-zero overhead.

---

## 3. Four-Phase Execution Roadmap

```
[Phase 1: Proof of Concept (SIFT1M)] ──► [Phase 2: SIMD & Acceleration Engine]
                    │                                      │
                    ▼                                      ▼
[Phase 4: Paper Writing & SCI Submission] ◄── [Phase 3: Scaling to 500M (SIFT1B)]
```

### Phase 1: Proof of Concept on SIFT1M (1 Million Scale)
* **Objective:** Validate correctness, measure initial compression ratio, and establish the C++ baseline.
* **Deliverables:**
  1. Download and preprocess `SIFT1M` (1M vectors, 128 dimensions).
  2. Implement `reorder_graph_bfs()` in `faiss/IndexACORN.cpp`.
  3. Implement `compress_neighbors_vbyte()` (Option A) and verify lossless reconstruction.
  4. Integrate on-the-fly decoding in `hybrid_search_from_candidates()`.
  5. Validate: **Recall@10 must match Vanilla ACORN 100% exactly**.

### Phase 2: High-Performance SIMD-BP128 Engine (Option B)
* **Objective:** Maximize search QPS and achieve > 55% memory compression.
* **Deliverables:**
  1. Implement SIMD-BP128 bit-packing codec with AVX2 intrinsics.
  2. Add benchmark instrumentation (`ACORNStats`) to track decoding latency per query.
  3. Conduct ablation study: Uncompressed vs. Option A (VByte) vs. Option B (SIMD-BP128).

### Phase 3: Large-Scale Scaling & Hierarchical Partitioning (Up to 500M)
* **Objective:** Enable the method to process up to 500 Million (50 crore) nodes without OOM.
* **Deliverables:**
  1. Implement Hierarchical/Chunked graph reordering.
  2. Benchmark on `SIFT1B` (10M subset $\to$ 100M $\to$ 500M scale).
  3. Test on multi-attribute datasets: `Paper` and `TripClick`.
  4. Measure memory consumption under various predicate selectivities ($0.001$ to $1.0$).

### Phase 4: Thesis Writing, Visualization & SCI Paper Submission
* **Objective:** Produce high-impact research plots and complete the manuscript.
* **Target Journals:**
  * *IEEE Transactions on Knowledge and Data Engineering (TKDE)* (IF: 8.9)
  * *VLDB Journal* (IF: 4.2)
  * *Information Systems (Elsevier)* (IF: 3.8)
* **Key Visualizations to Include:**
  * Pareto Frontier: Throughput (QPS) vs. Recall@10.
  * Memory Breakdown: Index size vs. Dataset Scale (1M to 500M).
  * Delta Distribution Histogram: Before vs. After Graph Reordering.
  * Decompression Overhead analysis across CPU thread counts.

---

## 4. Comprehensive Experimental Design (Ablation Matrix)

To guarantee acceptance in an SCI venue, the evaluation must include the following exhaustive matrix:

| Evaluation Dimension | Variations & Settings |
|---|---|
| **Datasets** | 1. `SIFT1M` (1M vectors, 128-dim)<br>2. `Paper` (2M vectors, attribute predicates)<br>3. `TripClick` (Health click log embeddings)<br>4. `SIFT1B` (10M to 500M scale) |
| **Methods Compared** | 1. Post-filter HNSW<br>2. Vanilla ACORN-1 & ACORN-$\gamma$<br>3. ACORN + Scalar VByte (Option A)<br>4. ACORN + SIMD-BP128 (Option B - **Proposed**)<br>5. Graph Reordering without compression (Ablation)<br>6. Compression without Graph Reordering (Ablation) |
| **Predicate Selectivity** | $0.001$ (0.1%), $0.01$ (1%), $0.05$ (5%), $0.20$ (20%), $1.0$ (Unfiltered) |
| **Primary Metrics** | 1. Memory footprint (GB) & Average bytes per neighbor slot<br>2. Search throughput (Queries Per Second / QPS)<br>3. Search Latency ($p50, p95, p99$ in milliseconds)<br>4. Recall@10 and Recall@100 accuracy<br>5. Index construction & reordering time (minutes) |

---

## 5. Potential Research Novelties & Paper Title Suggestions

### Title Alternatives for Submission:
1. **Option 1 (Recommended for IEEE TKDE / VLDB):**  
   *“TopoCompress: Scalable and Topology-Aware Graph Compression for Performant Hybrid Vector Search”*
2. **Option 2:**  
   *“Memory-Efficient Predicate-Agnostic Vector Search via Graph Reordering and SIMD Delta Encoding”*
3. **Option 3:**  
   *“CompactACORN: Halving Graph Footprint in Billion-Scale Hybrid Vector Retrieval with Zero Accuracy Loss”*

### Key Claims to Defend in the Paper:
1. **First Graph-Compression Study on Predicate-Agnostic ANN:** While compression has been studied for pure vector graphs (e.g. DiskANN), this is the first work addressing the unique $\gamma$-expansion memory explosion in hybrid predicate search.
2. **Strictly Lossless:** Graph topology is preserved identically; recall is mathematically identical ($1.00 \times$) to uncompressed ACORN.
3. **Hardware-Efficient:** Exploits modern SIMD execution pipelines, eliminating the traditional decoding latency penalty.

---

## 6. Project Directory Organization

```text
ACORN/
├── planing/
│   └── THESIS_PLAN.md          <-- [This comprehensive thesis plan]
├── Datasets/
│   ├── sift1M/                 <-- [Base dataset for Phase 1]
│   └── ...
├── faiss/
│   ├── IndexACORN.h            <-- [Modified with compression flags & structs]
│   ├── IndexACORN.cpp          <-- [Integrated Reordering & Compression APIs]
│   └── impl/
│       ├── ACORN.h             <-- [Compressed neighbor buffer definitions]
│       ├── ACORN.cpp           <-- [Hybrid search on-the-fly decompression]
│       └── codec/
│           ├── vbyte_codec.h   <-- [Option A: Scalar VByte encoder/decoder]
│           └── simd_bp128.h    <-- [Option B: AVX2 SIMD Bit-packing engine]
└── benchs/
    └── bench_compact_acorn.py  <-- [Automated evaluation & plotting scripts]
```

---
*Created on: 2026-11-12 | Master's Thesis Research Framework*

