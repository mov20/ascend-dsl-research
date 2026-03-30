# CLAUDE.md — Project Guide for Claude Code

## Project Overview

This repository is a research project on **alternative Python DSLs for Huawei Ascend NPU**.

**Goal:** Design a Python DSL for Ascend NPU that achieves:
- **Simplicity** — minimal lines of code, low barrier to entry
- **Performance** — ≥90% of peak hardware potential

**Compilation pipeline:**
```
Python DSL → MLIR → AscendC → CANN compiler → NPU binary
```

**Owner:** Oleg  
**Status:** 🟢 Active  
**Started:** 2026-03-30

---

## Repository Structure

```
ascend-dsl-research/
├── CLAUDE.md                    # This file — read first
├── README.md                    # Public project description
├── PROJECT.md                   # Project tracker, open questions, decisions log
├── ascend-dsl-comparison.md     # Table 1: High-level DSL comparison (7 languages)
├── ascend-dsl-syntax-perf.md    # Table 2: Syntax constructs & performance impact
└── docs/
    └── research-log.md          # Summary of all research discussions and decisions
```

---

## Process Rules

> **Always follow these rules. No exceptions.**

1. **Plan first** — before making any changes, write a clear action plan and get approval
2. **Show before commit** — show proposed changes (diff or content preview) before committing
3. **Never push without explicit "ok" or "push"** from Oleg
4. **Batch work** — group related tasks, avoid one-by-one iterations
5. **English only** — all documents, comments, and code in English
6. **Cite sources** — every data claim must have an inline reference `[[N]](#ref-N)` linking to Sources section

---

## Git Workflow

- **Main branch:** `main` — stable, reviewed content only
- **Feature branches:** create for any non-trivial change, open PR for review
- **PR process:** push branch → create PR → Oleg reviews/comments → apply feedback → merge
- **Commit messages:** short and descriptive (English)

```bash
# Start new feature
git checkout -b <branch-name>

# Push and create PR
git push -u origin <branch-name>
# Then create PR via GitHub API or web UI

# After merge, sync local main
git checkout main && git pull
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

## Key References

| Project | What it is | Link |
|---------|-----------|------|
| Triton-Ascend | Huawei's Triton fork for Ascend (primary active dev) | https://gitcode.com/Ascend/triton-ascend |
| TileLang-Ascend | tile-ai's TileLang adapter for Ascend (A2/A3) | https://github.com/tile-ai/tilelang-ascend |
| AscendCraft | LLM-driven DSL → AscendC auto-generation | https://arxiv.org/abs/2601.22760 |
| Triton community meetups | Official meetup notes (performance data, etc.) | https://github.com/triton-lang/triton/tree/main/docs/meetups |
| Helion (Meta) | Key syntax reference — PyTorch-style tile DSL | https://pytorch.org/blog/helion |

---

## Trigger Phrases

When working in this repo, recognize these commands from Oleg:

- **"Ascend summary"** — produce a summary of all discussions and decisions (read `docs/research-log.md` + `PROJECT.md`)
- **"Ascend summary for [period]"** — summary for a specific time range
- **"Ascend news"** — search for recent industry news on Huawei Ascend, CANN, NPU DSLs, and summarize
- **"Ascend news for [period]"** — news for a specific time range

---

## Open Questions

See `PROJECT.md` for the full list. Key ones:

- [ ] Access to Ascend hardware for benchmarks?
- [ ] Target audience: ML engineers or kernel developers?
- [ ] Licensing / open-source plan?
