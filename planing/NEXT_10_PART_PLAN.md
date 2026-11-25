# Next 10+ Part Execution Plan for ACORN Thesis Work

This is the actionable next-stage plan derived from the thesis roadmap in [THESIS_PLAN.md](THESIS_PLAN.md). The goal is to finish the proof-of-concept compression pipeline before moving to SIMD optimization and paper writing.

## Part 1 — Confirm the baseline ACORN repo state
- Open the project and verify the repository builds in the current environment.
- Check whether this workspace is the main ACORN repo or a variant copy.
- Confirm the build path and required dependencies (CMake, BLAS/OpenBLAS, compiler flags).
- Record the exact build command used for later benchmarking.
- Save a short note: baseline build status, compiler version, and platform details.

Deliverable: one-page build status note.

---

## Part 2 — Reproduce the vanilla ACORN baseline on SIFT1M
- Download or locate the SIFT1M dataset in the project data folders.
- Build a minimal ACORN run using the existing demo/test code.
- Run a small query set to establish baseline recall, latency, and graph memory footprint.
- Collect baseline stats: recall@10, time per query, and memory.
- Compare the results against the expected behavior described in the analysis report.

Deliverable: baseline benchmark results table.

---

## Part 3 — Write a precise implementation checklist for the compression feature
- Document the exact functions to modify: likely in [ACORN/faiss/IndexACORN.cpp](../faiss/IndexACORN.cpp) and [ACORN/faiss/IndexACORN.h](../faiss/IndexACORN.h).
- Define the target abstraction layer: graph reorder, delta encode, decode, and query-time use.
- Clarify where the compressed storage should live in the index structure.
- Decide how the original uncompressed graph is preserved during validation.

Deliverable: design note with exact file-level touch points.

---

## Part 4 — Implement the graph reordering prototype
- Create a prototype function to reorder neighbor IDs in a topology-aware manner.
- Start with a simple local BFS or hierarchical ordering strategy.
- Verify that the resulting neighbors are more localized than the original graph.
- Measure delta sizes before and after reordering.
- Confirm the ordering does not change graph semantics.

Deliverable: graph reordering function + delta histogram summary.

---

## Part 5 — Implement the scalar VByte delta compression path
- Sort neighbor IDs within each node.
- Compute deltas: first value as-is, then differences to the previous neighbor.
- Encode deltas using a simple VByte implementation.
- Add decode logic to reconstruct the original neighbor list exactly.
- Validate with random nodes and a full index subset.

Deliverable: lossless VByte encode/decode utility.

---

## Part 6 — Integrate compressed storage into ACORN search flow
- Add a compressed-neighbor path in the query-time search logic.
- Ensure the decompressor runs only when needed.
- Keep memory allocation minimal: prefer stack buffers or preallocated buffers instead of malloc during query.
- Verify that standard search still returns the same neighbors as the uncompressed version.

Deliverable: working compressed-search path with minimal extra allocations.

---

## Part 7 — Run the first correctness gate: zero recall loss
- Compare vanilla ACORN vs. reordered + VByte compressed ACORN on the same dataset and query set.
- Measure recall@10 and recall@100.
- Confirm the compressed pipeline is mathematically and operationally lossless.
- Record any differences in retrieved IDs and explain them if present.

Deliverable: correctness validation report with pass/fail criteria.

---

## Part 8 — Measure memory and compression ratio
- Compute index size for uncompressed and compressed graphs.
- Measure average bytes per neighbor slot.
- Compare with the thesis target of roughly 50%–60% reduction.
- Plot or tabulate memory vs. graph size.
- Check whether the reordering step is material to the gain.

Deliverable: compression ratio summary table.

---

## Part 9 — Benchmark latency and QPS impact
- Measure search latency before and after compression.
- Measure p50, p95, and p99 if feasible.
- Confirm that the overhead stays within the thesis target (< 3% or close to it).
- Identify whether decode overhead dominates the search cost.

Deliverable: latency/QPS benchmark output.

---

## Part 10 — Build the SIMD-BP128 design document
- Once the scalar path is working, plan the AVX2/AVX-512 implementation.
- Define how 128-delta blocks are packed and unpacked.
- Write the exact algorithm for bit-width selection and vectorized decode.
- Decide the fallback path for unsupported CPUs.

Deliverable: SIMD design note with pseudo-code and CPU assumptions.

---

## Part 11 — Prepare the experimental section plan for the thesis
- Write the evaluation matrix and ablation list.
- Define the dataset splits and selectivity scenarios.
- Prepare the metrics to report: memory, recall, QPS, latency, construction time.
- Plan the figures for the final paper.

Deliverable: experimental design section draft.

---

## Part 12 — Prepare the final thesis transition
- After the proof-of-concept passes, write the result summary.
- Turn the successful VByte result into a clear thesis contribution.
- Create the next-step list for SIMD optimization and scaling work.
- Decide whether the final paper should report both scalar and SIMD compression variants.

Deliverable: thesis transition note and publication roadmap.

---

## Recommended priority order
If you want the minimum-risk path, do these first in order:
1. Part 2 — baseline SIFT1M
2. Part 4 — graph reordering prototype
3. Part 5 — scalar VByte compression
4. Part 6 — integrate compressed decoding
5. Part 7 — zero recall loss validation
6. Part 8 — memory ratio
7. Part 9 — latency/QPS

This is the exact path that turns the research concept into a defendable thesis contribution.

---

## Ask for the next part
When you are ready, I will continue with the next single part in this sequence and wait for your confirmation before moving to the next one.
