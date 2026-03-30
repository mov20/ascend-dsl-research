# Research Log

*Chronological summary of all research discussions and decisions.*

---

## 2026-03-30 — Project Kickoff

### Project Definition

**Context:** Oleg initiated a research project to design an alternative Python DSL for Huawei Ascend NPU.

**Goals established:**
- Goal 1 — **Simplicity:** minimal lines of code, clean syntax, low barrier to entry
- Goal 2 — **Performance:** ≥90% of peak hardware potential on Ascend NPU

### Architecture Decisions

**DSL level:** Triton-like kernel DSL (operator/kernel level), with a roadmap toward graph-level later.
- Rationale: Modern neural network optimization requires more than just operator-level control (graph-level needed for fusion/scheduling), but kernel DSL is the right starting point.

**IR strategy:** MLIR (Python DSL → MLIR → AscendC codegen)
- Considered: direct CANN CCE-C generation, TVM TIR → Ascend target
- Chosen: MLIR — more flexible, reusable optimizations, extensible with custom Ascend dialect
- Key insight: Triton itself moved to MLIR (Triton IR → TritonGPU IR → LLVM)

**Code generation target:** AscendC (not abstract CANN backend)
- AscendC is Huawei's official C++-like API for custom operators on Ascend NPU
- Generates verifiable, debuggable code
- Full pipeline: Python DSL → MLIR → AscendC → CANN compiler → NPU binary

### Languages Selected for Analysis

The following languages were selected for comparative analysis:

| Language | Author | Why included |
|----------|--------|--------------|
| Triton | OpenAI | Primary reference; industry standard kernel DSL |
| Triton-Ascend | Huawei | Direct Ascend port of Triton — key competitor/reference |
| TileLang | tile-ai | Tile-based, Ascend support — key competitor/reference |
| TileLang-Ascend | tile-ai | TileLang adapter for Ascend A2/A3, AscendC codegen |
| Pallas | Google/JAX | Tile/grid model, TPU + GPU |
| Gluon | OpenAI | Warp-level control, exceeds Triton performance |
| cuTile | NVIDIA | Minimal LOC, compiler-automated performance |
| Mojo | Modular | Python superset, systems-level |
| AscendCraft | Research | LLM-driven DSL → AscendC auto-generation |
| Helion | Meta/PyTorch | PyTorch-native tile DSL, beats hand-written Triton |

### Key Findings from Language Analysis

**Dominant trend:** Tile-based abstractions dominate all new DSLs (TileLang, cuTile, Helion, Pallas). Tile naturally fits accelerator memory hierarchies (HBM → L2 → SRAM → registers).

**Performance ceiling insights (from Triton community meetups):**
- Triton out-of-the-box: only ~80% peak (Jeff Niu, OpenAI, Jul 2025 meetup)
- Flash Attention without warp specialization: 45% compute throughput on H100; with WS: 69% (Meta, Mar 2025 meetup)
- Gluon (warp-level) needed to exceed Triton performance, but FMHA on B200 still slower than cuDNN
- Helion achieves 1.85x vs hand-written Triton on H100 GEMM with PyTorch-level abstraction

**For 90% peak on Ascend — unique requirements not present in GPU DSLs:**
1. Explicit **Cube Unit** (matrix) vs **Vector Unit** (elementwise) vs **Scalar Unit** routing
2. **L0→L1→L2→HBM** pipeline — deeper memory hierarchy than GPU
3. **Multi-AI-core scheduling** — analogous to warp specialization but at core level
4. **CopyIn→Compute→CopyOut** — explicit pipeline model specific to Ascend

**Syntax patterns identified for our DSL:**
- `tile` loop as core primitive (from Helion, TileLang) — intuitive, minimal code
- Automatic scheduling + autotuning (from Helion, cuTile) — key to simplicity
- Explicit memory hints (from Pallas, Gluon) — required for 90% peak
- PyTorch-compatible syntax (from Helion) — low barrier to entry
- Warp/wave-level escape hatch (from Gluon) — for edge cases

### Ascend-Specific Findings

**Triton-Ascend** (gitcode.com/Ascend/triton-ascend):
- Huawei's official Triton fork for Ascend NPU
- Primary active development on gitcode.com (not gitee.com)
- Requirements: torch==2.6.0, torch-npu==2.6.0rc1
- Related ops repo: github.com/Ascend/triton-ascend-ops

**TileLang-Ascend** (github.com/tile-ai/tilelang-ascend):
- Released September 2025, open source
- Only third-party Python DSL with Ascend NPU backend (A2/A3)
- Two backends: AscendC & PTO route, AscendNPU IR route
- Supports: GEMM, batch GEMM, conv, Flash Attention, softmax, normalization, activation, GEMV, grouped GEMM, sparse FA, positional embedding, cross-entropy, linear attention/RNN, top-k
- No published TFLOPS benchmark numbers vs AscendC (as of Mar 2026)
- Active: pip install support added March 2026, T.Parallel added Dec 2025

**AscendCraft** (arxiv 2601.22760):
- LLM-driven pipeline: compact DSL → LLM generates DSL code → transpile to AscendC
- Results: 98% compilation rate, 90% correctness, but only 46% reach PyTorch eager perf
- Relevant for: validation of intermediate DSL concept between Python and AscendC
- Authors: academic-industrial collaboration (Nanjing University + Huawei ecosystem)
- **Not a competitor** — different niche (auto-generation, not manual programming)

### Process Rules Established

1. **Plan first** — write action plan, get approval before executing
2. **Batch work** — avoid one-by-one token-heavy iterations
3. **Documents in English**
4. **Cite sources** — every data claim needs a reference

### Repository Setup

- GitHub repo created: https://github.com/mov20/ascend-dsl-research (public)
- Files: PROJECT.md, README.md, ascend-dsl-comparison.md, ascend-dsl-syntax-perf.md
- Git credentials configured for automated pushes
- PR workflow established for review before merge

---

## Open TODOs

- [ ] Deep dive into AscendCraft paper (arxiv 2601.22760) — DSL design, host/kernel split, UB/L1 buffer model
- [ ] Deep dive into TileLang-Ascend — get actual benchmark numbers vs AscendC
- [ ] Start designing syntax for our DSL (informed by analysis above)
- [ ] Determine access to Ascend hardware for benchmarks
- [ ] Decide target audience: ML engineers vs kernel developers
- [ ] Decide licensing/open-source strategy
