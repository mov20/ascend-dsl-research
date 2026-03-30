# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project on **alternative Python DSLs for Huawei Ascend NPU**.

**Goal:** Design a Python DSL for Ascend NPU that achieves:
- **Simplicity** — minimal lines of code, low barrier to entry
- **Performance** — ≥90% of peak hardware potential

**Compilation pipeline:**
```
Python DSL → MLIR → AscendC → CANN compiler → NPU binary
```

**Owner:** Oleg | **GitHub:** https://github.com/mov20/ascend-dsl-research
**Status:** Active | **Started:** 2026-03-30

---

## Trigger Phrases

When Oleg uses these, execute accordingly:

- **"Ascend summary"** — read `docs/research-log.md` + `PROJECT.md`, produce a full summary of all discussions and decisions
- **"Ascend summary for [period]"** — same, filtered to that time range
- **"Ascend news"** — web-search for recent news on Huawei Ascend, CANN, NPU DSLs, and summarize
- **"Ascend news for [period]"** — news for a specific time range

---

## Process Rules

> **Always follow these rules. No exceptions.**

1. **Plan first** — before making any changes, write a clear action plan and get approval
2. **Show before commit** — show proposed changes (diff or content preview) before committing
3. **Never push to main directly** — always use a feature branch + PR
4. **Never merge without explicit approval** from Oleg
5. **Batch work** — group related tasks, avoid one-by-one iterations
6. **English only** — all documents, comments, and code in English
7. **Cite sources** — every data claim must have an inline reference `[[N]](#ref-N)` linking to Sources section

---

## Git Workflow

- **Main branch:** `main` — stable, reviewed content only
- **Feature branches:** create for any non-trivial change, open PR for review

```bash
git checkout -b <branch-name>
git push -u origin <branch-name>
# Then create PR via gh or web UI
git checkout main && git pull  # after merge
```

---

## Key Decisions (Architecture)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| DSL type | Kernel-level (Triton-like) | Focus on operator-level first; graph-level later |
| IR strategy | MLIR | Flexibility, reusable optimizations, ecosystem |
| Code generation target | AscendC | Official Huawei interface, debuggable, ~90% peak realistic |
| Syntax inspiration | Helion + cuTile style | Minimal LOC, PyTorch-compatible |
| Path to graph-level | Future roadmap | Current focus: kernel DSL |

---

## Languages Under Analysis

| Language | Author | Notes |
|----------|--------|-------|
| Triton | OpenAI | Kernel DSL, Python, MLIR, GPU — baseline reference |
| Triton-Ascend | Huawei | Triton fork for Ascend NPU — **primary competitor** |
| TileLang | tile-ai | Tile-based abstractions, GPU |
| TileLang-Ascend | tile-ai | TileLang for Ascend NPU (A2/A3), AscendC codegen — **primary competitor** |
| Helion | Meta/PyTorch | Python DSL, tile-based, built on Triton — **primary syntax reference** |
| cuTile | NVIDIA | Tile-level abstractions for CUDA — syntax reference |
| Pallas | Google/JAX | Kernel DSL for TPU/GPU via JAX |
| Mojo | Modular | Python superset, systems-level performance |
| AscendCraft | Research | LLM-driven DSL → AscendC auto-generation — see arxiv 2601.22760 |
| Gluon | ? | TBD |

---

## Key References

| Project | What it is | Link |
|---------|-----------|------|
| Triton-Ascend | Huawei's Triton fork for Ascend | https://gitcode.com/Ascend/triton-ascend |
| TileLang-Ascend | tile-ai's TileLang adapter for Ascend (A2/A3) | https://github.com/tile-ai/tilelang-ascend |
| AscendCraft | LLM-driven DSL → AscendC auto-generation | https://arxiv.org/abs/2601.22760 |
| Triton community meetups | Official meetup notes (performance data, etc.) | https://github.com/triton-lang/triton/tree/main/docs/meetups |
| Helion (Meta) | Key syntax reference — PyTorch-style tile DSL | https://pytorch.org/blog/helion |

---

## Key Files

- `PROJECT.md` — project tracker, open questions, decisions log
- `ascend-dsl-comparison.md` — Table 1: high-level DSL comparison
- `ascend-dsl-syntax-perf.md` — Table 2: syntax constructs & performance impact
- `docs/research-log.md` — running log of all research discussions and decisions

---

## Open Questions

See `PROJECT.md` for the full list. Key ones:

- [ ] Access to Ascend hardware for benchmarks?
- [ ] Target audience: ML engineers or kernel developers?
- [ ] Licensing / open-source plan?
