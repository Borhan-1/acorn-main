# Phase 4 Tracker — Publication Assets, Formal Proofs & SCI Q1 Manuscript

**Target Venues:** *Information Systems* (Elsevier, SCI Q1) / *Future Generation Computer Systems* (Elsevier, SCI Q1) / *IEEE TKDE* (SCI Q1)

## Overall Progress
| Sub-Part | Title | Status |
|---|---|---|
| **Part 4A** | Create `benchs/generate_thesis_plots.py` & generate 6 camera-ready 300-DPI PNG + PDF figures in `paper_figures/` | ✅ **Done** |
| **Part 4B** | Write formal mathematical proofs (`planing/THEORETICAL_ANALYSIS.md`) & camera-ready LaTeX tables (`paper_figures/tables.tex`) | ✅ **Done** |
| **Part 4C.1** | Manuscript Draft: Title, Abstract, Keywords, Sec 1 (Introduction), Sec 2 (Background & Related Work) | ✅ **Done** |
| **Part 4C.2** | Manuscript Draft: Sec 3 (System Architecture & Methodology — Reordering, AVX2 Codec, Split, Dynamic `add()`) | ✅ **Done** |
| **Part 4C.3** | Manuscript Draft: Sec 4 (Theoretical Analysis) & Sec 5 (Experimental Evaluation across 3 Domains & 6 Figures) | ✅ **Done** |
| **Part 4C.4** | Manuscript Draft: Sec 6 (Sub-Billion Scaling, Discussion & Limitations), Sec 7 (Conclusion) & Full Bibliography | ✅ **Done** |

---

## Part 4A — Camera-Ready Figures (`paper_figures/`) (DONE ✅)
Generated via [benchs/generate_thesis_plots.py](../benchs/generate_thesis_plots.py) in both **300-DPI PNG** and **vector PDF**:
- [x] `fig1_memory_scaling.(png|pdf)` (`297.9 KB` / `42.7 KB`) — 3-Dataset Graph Memory Comparison + `1M → 500M` Analytical RAM Scaling
- [x] `fig2_reordering_and_simd.(png|pdf)` (`289.9 KB` / `41.8 KB`) — Level-0 Delta Byte-Width Shift (`5.89×` 1-byte boost) + AVX2 Microbenchmark (`5.02×` faster) & Hierarchical Queue RAM (`32×` smaller)
- [x] `fig3_selectivity_crossover.(png|pdf)` (`359.5 KB` / `43.0 KB`) — `1M SIFT1M` 6-Selectivity Hybrid Search Regime Map (Pre-Filter vs. Post-Filter HNSW vs. Vanilla vs. VByte vs. Ours)
- [x] `fig4_prefix_sensitivity.(png|pdf)` (`318.8 KB` / `42.9 KB`) — `proximity_prefix` Sensitivity Curve (`P = 0..32`) validating closed-form $k^*(\sigma)$ optimum on `1M SIFT1M` & `GloVe-400K`
- [x] `fig5_multidomain_pareto.(png|pdf)` (`331.5 KB` / `38.2 KB`) — 3-Domain (`100-d NLP`, `128-d Vision`, `1,536-d LLM`) Recall@10 vs. Latency Pareto Frontier
- [x] `fig6_mt_qps_and_dynamic.(png|pdf)` (`297.0 KB` / `39.5 KB`) — Multi-Thread QPS Scaling (`1T/2T/4T`) + Live Dynamic `add()` (`+50K` inserts & `1.95s` re-compaction)

---

## Part 4B — Formal Proofs & Camera-Ready LaTeX Tables (DONE ✅)
1. **[planing/THEORETICAL_ANALYSIS.md](THEORETICAL_ANALYSIS.md):**
   - [x] **Proposition 1:** Exact Space Upper Bound & Base-Separation Savings (`~30 MB` saved on 1M vs. monolithic BP128).
   - [x] **Proposition 2:** Sublinear $O(N/M + B_{\text{loc}})$ Queue Bound of Hierarchical Reordering (`31,250` predicted vs. `31,254` measured on 1M — `32.00×` smaller!).
   - [x] **Proposition 3:** Closed-Form 2-Hop Early-Termination Horizon $k^*(\sigma) = \min(\bar{d}_0, \lceil 2M / (\sigma(1+\bar{d}_0)) \rceil)$ proving why $P \in [4, 8]$ is Pareto-optimal.
   - [x] **Proposition 4:** Bounded Intra-Cluster Deltas & $O(\bar{d}_0 \log \bar{d}_0 / \rho)$ Amortized Dynamic Compaction (`39.1 µs/insert`).
   - [x] **Analytical Cache-Line & Bandwidth Model:** Proving why `4 → 2` cache lines + `1.58 ns/edge` AVX2 decode yields net `-90 ns/hop` speedup while Scalar VByte incurs `+230 ns/hop` slowdown.
2. **[paper_figures/tables.tex](../paper_figures/tables.tex):**
   - [x] **Tables 1–6** formatted with `booktabs` + `multirow` for Elsevier / IEEE two-column templates (brace-balanced and verified).

---

## Part 4C — Complete SCI Q1 Journal Manuscript Draft (DONE ✅)
- [x] **[planing/MANUSCRIPT_DRAFT.md](MANUSCRIPT_DRAFT.md)** — Complete end-to-end SCI Q1 manuscript (`Parts 4C.1–4C.4`):
  - **Part 4C.1:** Title, Abstract, Keywords, Sec 1 (Introduction & 4 Contributions), Sec 2 (Background & Related Work + Architectural Comparison Table)
  - **Part 4C.2:** Sec 3 (System Architecture & Methodology — Mermaid Pipeline, Algorithm 1 Pseudocode, Base-Separated AVX2 `SimdBP128Codec`, Two-Segment Split, Zero-Alloc Stack Execution, Cluster-Gapped Dynamic `add()`)
  - **Part 4C.3:** Sec 4 (Theoretical & Complexity Analysis — Propositions 1–4 + Cache-Line Model) & Sec 5 (Experimental Evaluation across `SIFT1M`, `GloVe-400K`, `DBpedia-OpenAI-1536d`, Figures 1–6 & Tables 1–6)
  - **Part 4C.4:** Sec 6 (`1M→500M` Sub-Billion Scaling Projection, Vector Quantization Synergy, Upstream `ACORN` Bug Diagnostics, Limitations), Sec 7 (Conclusion), & 25 Verified Peer-Reviewed References.

---

## Status: ALL PHASES (PHASES 1, 2, 3, 3.5, AND 4) 100% COMPLETE! 🎉✅

---

## Peer-Review Hardening Revisions (Split into 4 Tracked Sub-Parts)
| Sub-Part | Scope | Status |
|---|---|---|
| **Rev 1** | Polish Abstract, Sec 1 & Sec 2 (plain academic tone, soften overclaims, sharpen 3-obstacle novelty & baseline scope) | ✅ **Done** |
| **Rev 2** | Formalize Proofs in Sec 4 & `THEORETICAL_ANALYSIS.md` (explicit Chernoff tail bound in Prop. 2, explicit i.i.d./correlation $\kappa$ & capture-probability proof in Prop. 3) | ✅ **Done** |
| **Rev 3** | Polish Sec 3 & Sec 5 tone (remove `!`, `"critical reviewer question"`, excessive bolding, add explicit note on single-run variance & CI protocol) | ✅ **Done** |
| **Rev 4** | Polish Sec 6 & Sec 7, add mandatory **Reproducibility & Artifact Availability** section, remove uncited Ref `[25]` (`LAION-5B`) & verify 1-to-1 citations | ✅ **Done** |
