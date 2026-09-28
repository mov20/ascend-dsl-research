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

**Dominant trend:** Tile-based abstractions dominate all new DSLs. Tile naturally fits accelerator memory hierarchies (HBM → L2 → SRAM → registers).

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
- `tile` loop as core primitive (Helion, TileLang) — intuitive, minimal code
- Automatic scheduling + autotuning (Helion, cuTile) — key to simplicity
- Explicit memory hints (Pallas, Gluon) — required for 90% peak
- PyTorch-compatible syntax (Helion) — low barrier to entry
- Warp/wave-level escape hatch (Gluon) — for edge cases

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
- Active: pip install support added March 2026, T.Parallel added Dec 2025
- No published TFLOPS benchmark numbers vs AscendC (as of Mar 2026)

**AscendCraft** (arxiv 2601.22760):
- LLM-driven: compact DSL → LLM generates DSL code → transpile to AscendC
- Results: 98% compilation, 90% correctness, only 46% reach PyTorch eager perf
- **Not a competitor** — different niche (auto-generation, not manual programming)
- Relevant as: validation of intermediate DSL concept; host/kernel split design reference

### Process Rules Established

1. **Plan first** — write action plan, get approval before executing
2. **Never push to main directly** — always branch + PR
3. **Batch work** — avoid one-by-one token-heavy iterations
4. **Documents in English**
5. **Cite sources** — every data claim needs an inline reference

### Repository Setup

- GitHub repo: https://github.com/mov20/ascend-dsl-research (public)
- Files: PROJECT.md, README.md, ascend-dsl-comparison.md, ascend-dsl-syntax-perf.md
- PR workflow established for review before merge

---

## 2026-09-28 — Trends Doc Split Into H1 / H2 Editions

**Context:** H1 2026 has passed, so Oleg moved the trends doc to an H2 edition rather than keep
extending a doc titled H1.

**Decisions:**

- [`docs/python-dsl-trends-2026H1.md`](python-dsl-trends-2026H1.md) is **frozen and marked complete**,
  ending at §2.6. Its §1 Highlights, §2.7–§2.10 and §3 stubs are replaced by pointers to the H2 edition.
- [`docs/python-dsl-trends-2026H2.md`](python-dsl-trends-2026H2.md) is the **active edition**: full
  carry-forward of §2.0–§2.6 verbatim, plus §2.7–§2.10. All future work goes here.
- Reference numbering is shared: `[1]`–`[102]` mean the same in both editions; `[103]`–`[131]` are
  H2-only.
- #45 (§2.4) and #46 (§2.5–§2.6) were merged into the H1 file first, so the frozen edition is genuinely
  complete through §2.6. #47 was **closed**, not merged; its §2.7–§2.10 content was re-targeted to the
  H2 file.

**Open point:** three §2 items still carry "(post-H1, 2026-07)" caveats that read oddly in an H2-titled
edition. Left unchanged for now — rewording them alters how cited claims are framed, so it needs Oleg's
call.

---

## 2026-08-10 → 2026-09-19 — Python DSL Trends Doc, §2.0–§2.10

**Context:** The Python DSL trends doc built stage by stage.
§2.0–§2.3 merged in August (#37, #38, #41, #43, #44). §2.4–§2.10 drafted 2026-09-19 (#45, #46, #47).
Oleg's decisions on 2026-09-18: keep the H1 title and include Jul–Sep 2026 items flagged "post-H1";
finish §2 before §3 Strategy, which needs his positioning input. On 2026-09-28 the doc was split into
editions (see next entry): §2.0–§2.6 landed in both, §2.7–§2.10 in the H2 edition only.

### Findings That Change Our Picture

**1. Triton-Ascend now lives in the `triton-lang` org** (repo 2026-01-05, gitcode frozen 2026-05-18),
under Triton-community governance. Its latest release tracks upstream 3.2 while upstream is at 3.8; a
weekly AI-assisted merge workflow chases upstream.

**2. The only public Triton-vs-Ascend C number** is one GroupGEMM chart on 950: near parity on average,
0.70× on FP8 backward.

**3. Every vendor with a Triton path added a native tier below it:** NVIDIA CuTe DSL, AMD FlyDSL, Triton's
own Gluon. PyAsc2 fits that tier, complementary to Triton-Ascend.

**4. On Ascend, Ascend C owns the hot path.** vLLM-Ascend has 61 Ascend C ops and 140 Triton kernels.
PyAsc2's entry point is the Ascend C ops.

**5. LLMs write Ascend C at 2.5% Pass@1 (MultiKernelBench), but 90.4% correct through AscendCraft's DSL.**
This is an argument for PyAsc2 as a model-generable intermediate.

**6. AWS NKI is the only vendor DSL with in-kernel collectives** (stable 2026-04). The §2.0 claim that
no tile DSL has them was corrected to "no GPU tile DSL".

**7. Huawei has five public Python front-ends**, more than any vendor, with no published tier map.
AscendNPU-IR is open, but it is not yet a contract: it has no spec, stability promise or conformance suite.

### Corrections Made

- TT-Lang is a Python tile DSL, verified from two sources. It is neither Triton nor C++; the C++ layer is TT-Metalium.
- The §2.2 table cited NKI to the vLLM-Ascend reference by mistake; it now has its own reference.

### Follow-ups Added

See Open TODOs below.

---

## 2026-08-07 — CATLASS TLA DSL Analysis

**Context:** Oleg requested a full analysis of the `dsl` branch of Huawei's CATLASS repository
(`gitcode.com/cann/catlass`), which carries a Python frontend for Ascend kernels. Result:
[`docs/catlass-dsl-analysis.md`](catlass-dsl-analysis.md). Static source and git-history analysis
only — no build or execution (requires CANN ≥ 9.1.0 and Ascend 950 hardware).

### New Project Identified

**CATLASS TLA DSL** is a first-party Huawei Python DSL for Ascend, not previously in our tracking
tables. ~45k LOC (26k Python + 19k C++/MLIR), 13 authors, 151 commits in 12 weeks with accelerating
cadence. It belongs alongside Triton-Ascend and TileLang-Ascend in the Ascend Python-DSL landscape.

### Notable Findings

**1. It does not target AscendC.** Pipeline is Python → TLA MLIR dialect (79 ops) → ~23 lowering
passes → HIVM/HACC (AscendNPU-IR) → LLVM → device binary, bypassing AscendC source generation
entirely. Notable contrast with TileLang-Ascend, which does generate AscendC — the vendor's own DSL
team chose the AscendNPU-IR path instead.

**2. Explicit synchronization is a usability dead end, and they know it.** The DSL Flash Attention
example needs ~60 hand-declared sync flags and comes out *longer* than the C++ template version
(1,527 vs 1,401 LOC). Basic matmul is 334 LOC vs 148 LOC in C++. Their answer is
`@tla.kernel(auto_sync="v0")` — compiler-inferred synchronization via `TlaInsertAutoMutexPass`,
which cuts basic matmul from 334 → 259 LOC and removes all 15 flags.

**3. SIMT-on-AIV is viable.** `tla.vec.func(mode="simt", thread_block_dim=N)` with `thread_idx()`
compiles a CUDA-shaped kernel onto Ascend vector cores — `gm_c[i] = gm_a[i] + gm_b[i]`, no UB
staging, no tiles, no flags. If this holds up, the "Ascend cannot do SIMT" assumption behind much
tile-first DSL design deserves re-examination. Caveat: landed 2026-08-07 with exactly 3 supporting
ops (`simt_add`, `simt_load`, `simt_store`).

**4. Cross-core AIC↔AIV sync is the hard part nobody abstracts.** Both CATLASS DSL and
TileLang-Ascend expose it explicitly (`cross_flag` / `cross_core_set_flag` / `cross_core_wait_flag`,
with sync topology `mode` 1/2/4). Whichever DSL hides it first while keeping performance wins the
usability argument on Ascend.

**5. No communication primitives at all.** Zero HCCL / all-reduce / all-gather / reduce-scatter
matches across the entire repository, both branches. Single-device kernel DSL only — anything
distributed needs a separate layer.

### Maturity Assessment

**Recommendation: TRACK, do not adopt.** Reasons:

- **Never released** — no git tag (v1.0.0 through v1.6.3) contains `python/tla_dsl`.
- **Ascend 950PR/950DT only** — `SUPPORTED_ARCH_SCOPES = ("aiv.c310", "aic.c310")`. Nothing runs on
  A2/A3, which is what most hardware access looks like today.
- **No CI** — ~23,500 LOC of tests with no automated gate. "DSL CI" is a Q3 2026 roadmap goal.
- **~3 months old**, self-labeled beta (first commit 2026-05-16).
- **License** is CANN Open Software License Agreement v2.0 — Huawei-authored, not OSI-approved.
- **Branch is diverging** — `dsl` is 154 ahead / 143 behind `master`, forked 2026-05-13, while
  mainline restructures for v2.0.0.

Counterweight: engineering quality is genuinely good. Test LOC ≈ source LOC, MLIR-based architecture,
healthy bus factor (top contributor 18% of commits), and a well-written English syntax-constraints
document added during the analysis window.

### Their Q3 2026 Roadmap (issue #399, opened 2026-08-04)

Still outstanding by their own account: 40+ SIMD ops needed for Matmul/FA, NZ data format,
MxFP8/MxFP4/FP8/int8/int32 dtypes, tensor subscript access, L0C2UB and UB2L1 paths, cross-core sync
modes 1/2/4, DSL CI, and a `dsl-gen` backend for torch.inductor. Planned operators: SplitK, StreamK,
fullLoad, GroupMatmulSliceM, PFA, KDA, BSA — all Ascend 950.

### Follow-ups Added

See Open TODOs below. The highest-value experiment is measuring what `auto_sync="v0"` costs in
performance versus hand-placed flags — that number bounds how much synchronization any Ascend DSL
can hide without paying for it.

---

## Open TODOs

- [ ] Write trends doc §3 Strategy (positioning, pillars, risks, milestones) — needs Oleg's positioning input
- [ ] Write trends doc §1 Highlights — after §3
- [ ] Resolve trends doc Appendix A.1 (broader kernel set) and A.2 (dynamic shapes)
- [ ] Update `docs/asic-landscape.md` DSL matrix — Trainium has NKI; Tenstorrent has TT-Lang; Cambricon and Moore Threads have Triton backends
- [ ] Propose a public multi-DSL Ascend benchmark: Triton-Ascend, TileLang-Ascend, PyAsc2, Ascend C
- [ ] Deep dive into AscendCraft paper — DSL design, host/kernel split, UB/L1 buffer model
- [ ] Deep dive into TileLang-Ascend — get actual benchmark numbers vs AscendC
- [ ] Start designing syntax for our DSL
- [ ] Determine access to Ascend hardware for benchmarks
- [ ] Decide target audience: ML engineers vs kernel developers
- [ ] Decide licensing/open-source strategy
- [ ] Compare codegen targets across Ascend DSLs: AscendC (TileLang-Ascend) vs AscendNPU-IR/HIVM (CATLASS DSL)
- [ ] **Benchmark `auto_sync="v0"` overhead** in CATLASS DSL vs hand-placed flags (needs 950 hardware)
- [ ] Study `TlaInsertAutoMutexPass` as a reference implementation for automatic sync insertion
- [ ] Add CATLASS TLA DSL to `ascend-dsl-comparison.md` (Table 1) and `ascend-dsl-syntax-perf.md` (Table 2)
- [ ] Re-evaluate CATLASS DSL when: it ships in a tagged release, A2/A3 support lands, or CI goes live
