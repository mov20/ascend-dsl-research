# Ascend DSL Research

Research on alternative programming models and Python DSLs for Huawei Ascend NPU.

## Goal

Design a Python DSL for Ascend NPU that satisfies:
- **Simplicity**: minimal LOC, low barrier to entry
- **Performance**: ≥90% of peak hardware potential

## Pipeline

```
Python DSL → MLIR → AscendC → CANN → NPU binary
```

## Files

- `pyasc2-design.md` — primary design document (requirements, challenges, programming-model analysis, API spec)
- `PROJECT.md` — project tracker, open questions, decisions log
- `ascend-dsl-comparison.md` — high-level comparison of existing DSLs
- `ascend-dsl-syntax-perf.md` — detailed syntax constructs & performance impact
- `docs/research-log.md` — running log of research discussions and decisions
- `CLAUDE.md` — guidance for Claude Code when working in this repo
