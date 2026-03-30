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

- `PROJECT.md` — project tracker, open questions, decisions log
- `ascend-dsl-comparison.md` — high-level comparison of existing DSLs
- `ascend-dsl-syntax-perf.md` — detailed syntax constructs & performance impact
