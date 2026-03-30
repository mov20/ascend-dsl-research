# Alternative Programming Models for Huawei Ascend

**Status:** 🟢 Active
**Started:** 2026-03-30
**Horizon:** 3–12 months
**Owner:** Oleg

## Summary

Research and development of alternative programming languages and DSLs for the Huawei Ascend NPU/AI accelerator platform.

## Context

- Huawei Ascend — AI processor lineup (Ascend 310, 910, etc.)
- Primary stack: MindSpore, CANN (Compute Architecture for Neural Networks), AscendCL
- **Focus: design a custom Python DSL** for programming Ascend NPU
- **Goal 1 — Simplicity:** minimal lines of code, clean syntax, low barrier to entry
- **Goal 2 — Performance:** ≥90% of peak hardware potential on Ascend

## Architecture

- **Abstraction level:** Triton-like kernel DSL (operator/kernel level)
- **Roadmap:** later — move to graph-level for full pipeline optimization (fusion, scheduling)
- **Compilation pipeline:** Python DSL → MLIR → AscendC code generation
- **Target hardware:** Ascend NPU (Cube Unit for matrix ops, Vector Unit for elementwise ops)

## Languages Under Analysis

| Language | Author | Focus |
|----------|--------|-------|
| **Triton** | OpenAI | Kernel DSL, Python, MLIR, GPU |
| **Triton-Ascend** | Huawei | Triton fork for Ascend NPU — **key reference/competitor** |
| **TileLang** | tile-ai (open-source) | Tile-based abstractions, GPU |
| **TileLang-Ascend** | tile-ai (open-source) | TileLang for Ascend NPU (A2/A3), AscendC codegen — **key reference/competitor** |
| **Pallas** | Google/JAX | Kernel DSL for TPU/GPU via JAX |
| **Gluon** | ? | TBD |
| **cuTile** | NVIDIA | Tile-level abstractions for CUDA |
| **Mojo** | Modular | Python superset, systems-level performance |
| **AscendCraft** | Research | LLM-driven DSL → AscendC kernel auto-generation (arxiv 2601.22760) |
| **Helion** | Meta/PyTorch | Python DSL, tile-based, built on top of Triton |

**Analysis goal:** syntactic constructs, abstractions, approaches to near-peak performance. What to adopt, what to skip, and why.

## Open Questions

- [ ] Access to Ascend hardware for benchmarks?
- [ ] Target audience for the DSL? (ML engineers, kernel developers?)
- [x] IR strategy: **MLIR** (decided)
- [ ] Licensing / open-source plan?

## Process Rules

1. **Plan first** — always write an action plan and get approval before executing
2. **Batch work** — avoid one-by-one token-heavy iterations; group tasks
3. **Documents in English**

## Trigger Phrases

- **"Ascend summary"** — get a summary of all discussions and decisions
- **"Ascend summary for [period]"** — summary for a specific time range
- **"Ascend news"** — digest of recent industry news (Huawei Ascend, CANN, MindSpore, alternative NPU languages)
- **"Ascend news for [period]"** — news for a specific time range

## Log

### 2026-03-30
- Project track created
- Focus defined: Python DSL → MLIR → AscendC codegen
- Goals: simplicity (min LOC) + ≥90% peak performance
- Level: Triton-like kernel DSL, with a path to graph-level later
- High-level comparison table created: `ascend-dsl-comparison.md`
- Detailed syntax/perf table created: `ascend-dsl-syntax-perf.md`
- Ascend-specific projects added: Triton-Ascend, TileLang-Ascend, AscendCraft
- GitHub repo created: https://github.com/mov20/ascend-dsl-research
- TODO: deep dive into AscendCraft (DSL design, host/kernel split, UB/L1 buffer model)
