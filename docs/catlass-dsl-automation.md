# CATLASS TLA DSL — What Is Automated, What Is Not

*Last updated: 2026-08-07*

> Focused follow-up to [`catlass-dsl-analysis.md`](catlass-dsl-analysis.md), answering one question:
> in Huawei's CATLASS TLA DSL, are **double buffering**, **UB / on-chip memory management**, and
> **synchronization insertion** handled by the compiler, or must the programmer write them by hand?
>
> This is the question that separates a *productivity* DSL from a *notation* for hardware the
> programmer still has to schedule personally. The answer here is mostly the latter, with one
> partial exception and one escape hatch.

**Analyzed at commit `c511c43`**, branch `dsl`, 2026-08-07. <sup>[[1]](#ref-1)</sup>
**Method:** static reading of the MLIR compiler passes and all 44 end-to-end examples. Nothing built
or executed — the toolchain needs CANN ≥ 9.1.0 and Ascend 950 hardware.

---

## Verdict

| Concern | Automated? | What the compiler does | What the programmer still writes |
|---|---|---|---|
| **Double buffering** | **No** — fully manual | Nothing. No pass creates or manages multi-buffering. | Allocate every buffer half separately, name them (`_ping`/`_pong`, `l1a0`/`l1a1`), maintain the toggle index, write the select expression |
| **UB / on-chip memory** | **Partial** — offset assignment only | Bump-allocates static byte offsets per address space | Declare every buffer with a **statically known** size, explicit address space, explicit byte alignment. Fit everything into capacity yourself |
| **Sync insertion** | **Opt-in, intra-core only** | `auto_sync="v0"` inserts intra-core mutexes | All cross-core (AIC↔AIV) sync, always. Everything, if you don't opt in — and it's all-or-nothing |

**One escape hatch:** SIMT mode (`tla.vec.func(mode="simt")`) requires none of the three — no buffer
allocation, no flags, no double buffering. It is also three ops old and vector-only. <sup>[[2]](#ref-2)</sup>

**Headline evidence:** across the 44 end-to-end examples, 31 use manual flag synchronization and only
**2** use `auto_sync`. The flagship Flash Attention example hand-writes 33 buffer allocations, 33
flag declarations, 7 cross-core flags, and **200 `set_flag`/`wait_flag` calls** in 1,400
lines. <sup>[[3]](#ref-3)</sup> <sup>[[4]](#ref-4)</sup>

---

## 1. Double Buffering — Fully Manual

No compiler pass creates, infers, or manages multi-buffering. A search for the usual machinery
(`double_buffer`, `pingpong`, `multibuffer`, software pipelining) across the entire Python and C++
tree returns hits in exactly two files, both of them *examples* describing what the programmer did by
hand — not implementation. <sup>[[5]](#ref-5)</sup>

The programmer allocates each half, maintains the index, and writes the select:

```python
l1a0_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)
l1a1_ptr = tla.allocate(l1_tm * l1_tk, DTYPE_A, tla.AddressSpace.l1, 512)   # second half, by hand
...
l1_buf_idx = c0
for ...:
    l1_a = tla.make_tensor_like(
        l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1)          # select, by hand
    ...
    l1_buf_idx = c1 - l1_buf_idx                                           # toggle, by hand
```

**This is unchanged under `auto_sync="v0"`.** The auto-sync variant of the basic matmul still
allocates all nine buffers, still declares both ping-pong halves for L1A/L1B/L0A/L0B, still keeps
`l1_buf_idx` and `l0_buf_idx`, and still toggles them with `c1 - idx`. Only the *flags* disappear —
the buffering structure is untouched. <sup>[[6]](#ref-6)</sup>

The Flash Attention example makes the cost visible: buffers are named `l1_k_ping_ptr`,
`l1_k_pong_ptr`, `l1_v_ping_ptr`, `l1_v_pong_ptr`, `l0a_ping_ptr`, `l0a_pong_ptr` — the ping/pong
distinction lives in *identifier names*, which is as manual as it gets. 56 lines of that file deal
with buffer-index bookkeeping. <sup>[[4]](#ref-4)</sup>

What the compiler *does* understand is the ping-pong **select pattern**, but only in order to attach
mutexes correctly. `TlaInsertAutoMutexPass` models pointer provenance as a `StorageExpr` tree with a
`Select` node carrying `condition` / `thenValue` / `elseValue` / `materializedMutexes[2]`, so it can
tell which physical buffer a pointer refers to on each branch. That is dependence *analysis over*
programmer-written double buffering — not synthesis of it. <sup>[[7]](#ref-7)</sup>

**Nothing on the Q3 2026 roadmap changes this.** The roadmap's memory-related items are new *data
paths* (L0C2UB, UB2L1) and cross-core sync *modes*, not buffering automation. <sup>[[8]](#ref-8)</sup>

---

## 2. UB / On-Chip Memory — Offsets Only

### What the programmer must supply

`tla.allocate(shape, dtype, mem_scope, byte_alignment)` requires all four explicitly, and the shape
must be **fully static** — the docstring states "It must be fully static", and the alternative
byte-oriented API raises `TlaLoweringError` on a non-constant size: *"requires a static size_bytes
(compile-time constant); dynamic sizes are not supported."* <sup>[[9]](#ref-9)</sup> <sup>[[10]](#ref-10)</sup>

So the programmer chooses: which memory (L1 / L0A / L0B / L0C / UB), how big, what alignment, and how
many buffers. 19 of 44 example files call `tla.allocate` directly. <sup>[[3]](#ref-3)</sup>

### What the compiler supplies

`planTlaScratchAllocations` walks every `AllocPtrOp` in the module and assigns byte offsets: <sup>[[11]](#ref-11)</sup>

```cpp
FailureOr<uint64_t> base = alignUpCheckedU64(nextOffsetByAddrspace[addressSpaceKey], alignment);
...
nextOffsetByAddrspace[addressSpaceKey] = *base + *alignedSize;
```

That is a **monotonic bump allocator**, and it is the whole of the memory management. The header
comment describes the result as "One statically assigned, whole-kernel scratch allocation."

Concretely, what is **absent**:

- **No liveness analysis.** A grep for `liveness` / `lifetime` / `reuse` / `free` / `dealloc` across
  the allocator and the pointer-lowering pass returns nothing. <sup>[[11]](#ref-11)</sup>
- **No buffer reuse.** Offsets only ever increase. Two buffers with disjoint live ranges still
  occupy distinct bytes for the whole kernel. If you declare five buffers and only two are ever live
  at once, you pay for five.
- **No capacity fitting or spilling.** `LocalmemAllocator.capacity_in_bytes()` exists so you can
  *query* the limit, but fitting within it is your problem. <sup>[[10]](#ref-10)</sup>
- **No dynamic allocation.** Everything is static, whole-kernel.

### `LocalmemAllocator` is not an allocator

The Flash Attention example calls `tla.utils.LocalmemAllocator()`, which reads like a real allocator.
It is not. The class is declared `__slots__ = ()` — **stateless** — and its `allocate()` emits the
same `tla.alloc_ptr` op that `tla.allocate` does. The differences are cosmetic: it takes **bytes**
instead of elements and returns an `i8` pointer you must then `recast_ptr` to your dtype, which is
why every allocation in that example is two lines instead of one. <sup>[[10]](#ref-10)</sup> <sup>[[4]](#ref-4)</sup>

Real offset assignment still happens later, in the same bump allocator.

---

## 3. Synchronization — Opt-In, Intra-Core Only

### The default: everything by hand

By default, `catlass/dsl.py` states plainly: *"local synchronization remains explicit."* The
programmer declares each flag as a producer→consumer pipe pair and places every set/wait: <sup>[[12]](#ref-12)</sup>

```python
l1a0_data_ready = tla.flag("l1a0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
l0c_data_ready  = tla.flag("l0c_data_ready",  tla.arch.CUBE, tla.arch.FIX)
...
tla.wait_flag(l1a0_available); tla.copy(l1_a, gm_a_by_l1); tla.set_flag(l1a0_data_ready)
```

Basic matmul needs 15 such flags. Flash Attention needs 33 flags plus 7 cross-core flags, driving 200
`set_flag`/`wait_flag` calls. <sup>[[4]](#ref-4)</sup> <sup>[[6]](#ref-6)</sup>

### `auto_sync="v0"` — what it actually covers

`@tla.kernel(auto_sync="v0")` sets a `tla.auto_sync` function attribute that enables
`TlaInsertAutoMutexPass`, which infers intra-core mutexes from pointer provenance. On basic matmul it
removes all 15 flag declarations and every set/wait, cutting 334 → 259 lines. <sup>[[6]](#ref-6)</sup> <sup>[[7]](#ref-7)</sup>

Four limits matter, all read directly from the pass source:

**1. Cross-core sync is never automated.** The pass's own diagnostic is explicit:

> `auto_sync='v0' cannot be combined with local mutex, mutex_guard, or local flag synchronization;`
> **`cross_core_* remains explicit`**

The pass has two mutex ID spaces, `Cube = 0` and `Vector = 1`, kept separate — it reasons within one
core type at a time and never emits a `cross_core_set_flag` / `cross_core_wait_flag`. For a mixed
AIC+AIV kernel, the hardest synchronization is still yours. <sup>[[7]](#ref-7)</sup>

**2. It is all-or-nothing.** The same diagnostic rejects mixing. You cannot let the compiler handle
the routine pipeline and hand-place the two flags you care about — you take the whole pass or none of
it. That makes incremental adoption on an existing kernel impossible.

**3. It bails out on a long list of patterns.** The pass carries 23 distinct error paths. The
substantive ones:

| Constraint | Diagnostic |
|---|---|
| Buffers must resolve to a static `alloc_ptr` root | *"requires every accessed on-chip tensor to resolve to a static tla.alloc_ptr capacity/root; bare on-chip addresses and changing loop-carried pointers are unsupported"* |
| Ping-pong must be a two-branch `scf.if` | *"automatic mutex requires a two-branch scf.if selector"* |
| Restricted `unit_flag` values | *"supports copy unit_flag values 0 or 3"*; *"supports tla.mmad unit_flag values 0, 2, or 3"* |
| `mmad` unit_flag must be statically decidable | *"requires tla.mmad unit_flag to be provably always zero or always enabled with value 2/3"* — a runtime choice "cannot safely determine L0C locking" |
| MIX kernels must be regioned | *"instructions in a MIX kernel must be inside tla.cube or tla.vector"* |
| Hard cap | `kMaxAutoMutexIds = 32` |

"Changing loop-carried pointers are unsupported" is the sharpest one: the pointer-rotation idiom that
deep software pipelining wants is exactly what the pass refuses. <sup>[[7]](#ref-7)</sup>

**4. Almost nobody uses it.** 2 of 44 examples, against 31 using manual flags. Notably the flagship
Flash Attention kernel does **not** use it — consistent with the constraints above ruling out its
7-cross-flag, multi-stage pipeline. <sup>[[3]](#ref-3)</sup>

Test coverage is nonetheless real: one pytest module (`test_local_autosync.py`) plus 10 lit tests
covering diagnostics, ID spaces, dynamic select, mixed split, and vec-func. This is a maintained
feature, not an abandoned prototype. <sup>[[13]](#ref-13)</sup>

---

## 4. The SIMT Exception

`tla.vec.func(mode="simt")` sidesteps all three concerns at once. The entire kernel body is: <sup>[[2]](#ref-2)</sup>

```python
with tla.vector():
    with tla.vec.func(mode="simt", thread_block_dim=VECTOR_ELE):
        tid, _, _ = tla.arch.thread_idx()
        thread_block_dim, _, _ = tla.arch.thread_block_dim()
        for i in tla.range(tid, VECTOR_ELE, thread_block_dim):
            gm_c[i] = gm_a[i] + gm_b[i]
```

Zero `tla.allocate`, zero `tla.flag` — verified by grep against that file. There is no UB staging to
manage and no producer/consumer pipe pair to order, because operations lower onto GM memrefs
directly; the vector-region pass converts `simt_load`/`simt_store` onto memref parameters. The
example's own comment notes the consequence: buffers "are statically shaped memrefs, since only a
pointer crosses the launch ABI."

The caveats are severe: **three ops** exist (`simt_add`, `simt_load`, `simt_store`), the mode is
vector-only (no cube/mmad), and it landed 2026-08-07. It demonstrates the model works on AIV
hardware; it does not yet do useful work.

---

## 5. Reading of the Trajectory

The direction is real but early, and it is narrower than the automation vocabulary suggests:

- **Sync automation exists and is maintained** — `auto_sync="v0"` with 11 tests. But the version
  string is honest: intra-core only, all-or-nothing, 23 bail-out paths, 2 users among the examples.
- **Memory automation does not exist** beyond offset assignment. A bump allocator with no liveness is
  the least an MLIR compiler can do here; buffer reuse under UB pressure is where the real difficulty
  lives, and it is untouched.
- **Buffering automation does not exist at all**, and is not on the published roadmap.

The pattern across all three is consistent: the compiler analyzes what the programmer wrote and fills
in bookkeeping; it does not make scheduling decisions. That places CATLASS DSL closer to a typed
Python notation for Ascend C than to a scheduling compiler — which matches the project's own README
statement that CATLASS-template-style higher-level wrapping remains future work. <sup>[[14]](#ref-14)</sup>

**Open question requiring hardware:** what `auto_sync="v0"` costs against hand-placed flags. If the
inferred mutexes are conservative, the 75-line saving on basic matmul buys a performance regression,
and that trade determines whether the feature can grow beyond 2 examples. No benchmark exists in the
repo. This is already logged as a TODO in [`research-log.md`](research-log.md).

---

## 6. References

Source paths are relative to the CATLASS repository root on branch `dsl` at commit `c511c43`.
Statistics marked "git" are reproducible from a clone with the command shown.

| # | Description | URL / Command |
|---|-------------|---------------|
| <a name="ref-1"></a>[1] | CATLASS repository, branch `dsl` @ `c511c43` | https://gitcode.com/cann/catlass/tree/dsl |
| <a name="ref-2"></a>[2] | SIMT vector-add example — no `tla.allocate`, no `tla.flag`; `thread_idx()` / `thread_block_dim`; comment on statically shaped memrefs and pointer-only launch ABI. SIMT lowering onto memref parameters in `TlaVectorRegionPass.cpp:2534` | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/examples/end_to_end/simt/basic_vadd_simt.py · git: `grep -c 'tla\.allocate(\|tla\.flag(' examples/end_to_end/simt/basic_vadd_simt.py` → 0 |
| <a name="ref-3"></a>[3] | Example-corpus counts: 19 of 44 example files call `tla.allocate`; 31 use manual `tla.flag`/`set_flag`/`wait_flag`; 2 use `auto_sync` | git: `grep -rl 'tla\.allocate(' examples/ \| wc -l`; `grep -rl 'tla\.flag(\|set_flag\|wait_flag' examples/ --include='*.py' \| wc -l`; `grep -rl auto_sync examples/ \| wc -l` |
| <a name="ref-4"></a>[4] | Flash Attention example — 1,400 lines; 33 `LocalmemAllocator.allocate` calls, 33 `tla.flag` declarations, 7 `tla.cross_flag`, 200 `set_flag`/`wait_flag` calls, 56 lines of buffer-index bookkeeping; hand-named `l1_k_ping_ptr` / `l1_k_pong_ptr` / `l0a_ping_ptr` / `l0a_pong_ptr`; two-line allocate+`recast_ptr` idiom | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/examples/end_to_end/flash_attention_infer/flash_attention_infer.py |
| <a name="ref-5"></a>[5] | Absence of buffering automation: a case-insensitive search for `double_buf` / `pingpong` / `ping_pong` / `multibuf` / `software pipelin` across `.py`, `.cpp`, `.h`, `.md` (excluding `3rdparty/`) matches only two example files describing hand-written buffering — no implementation | git: `grep -ril 'double.\?buf\|pingpong\|ping_pong\|multibuf\|software.\?pipelin' --include='*.py' --include='*.cpp' --include='*.h' --include='*.md' . \| grep -v 3rdparty` |
| <a name="ref-6"></a>[6] | Basic matmul examples — `basic_matmul.py` 334 LOC with 15 hand-declared flags; `basic_matmul_auto_sync.py` 259 LOC with zero flags but **unchanged** manual buffering: 9 `tla.allocate` calls, ping-pong halves `l1a0`/`l1a1`, `l0a0`/`l0a1`, `l0b0`/`l0b1`, indices `l1_buf_idx`/`l0_buf_idx` toggled via `c1 - idx` | https://gitcode.com/cann/catlass/tree/dsl/python/tla_dsl/examples/end_to_end/basic_mmad |
| <a name="ref-7"></a>[7] | `TlaInsertAutoMutexPass` (967 LOC) — `kAutoSyncAttrName = "tla.auto_sync"`; `kMaxAutoMutexIds = 32`; `MutexIdSpace{Cube=0, Vector=1}`; `StorageExpr` provenance tree with `Select` node and `materializedMutexes[2]`; 23 error paths including the cross-core, all-or-nothing, static-root, two-branch-`scf.if`, and `unit_flag` constraints quoted in §3 | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/csrc/mlir/lib/Passes/TlaInsertAutoMutexPass.cpp |
| <a name="ref-8"></a>[8] | CATLASS 2026 Q3 RoadMap (issue #399) — memory/sync items are new data paths (L0C2UB, UB2L1) and cross-core sync modes 1/2/4; no buffering or allocation automation listed | https://gitcode.com/cann/catlass/issues/399 |
| <a name="ref-9"></a>[9] | `tla.allocate(shape, dtype, mem_scope, byte_alignment)` — docstring "It must be fully static"; rejects `generic`/`gm` scopes; alignment and address space are required arguments | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/catlass/core_api.py#L5858 |
| <a name="ref-10"></a>[10] | `LocalmemAllocator` — declared `__slots__ = ()` (stateless); `allocate()` emits the same `tla.alloc_ptr` op, taking bytes and returning an `i8` pointer requiring `recast_ptr`; raises "requires a static size_bytes (compile-time constant); dynamic sizes are not supported"; `capacity_in_bytes()` queries the limit but performs no fitting | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/catlass/utils/localmem_allocator.py |
| <a name="ref-11"></a>[11] | `planTlaScratchAllocations` — monotonic bump allocator over all `AllocPtrOp`, per address space (`nextOffsetByAddrspace[key] = base + alignedSize`); header comment "One statically assigned, whole-kernel scratch allocation". No liveness, reuse, free, or spilling anywhere in the allocator or `TlaLowerPtrPass` | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/csrc/mlir/lib/Passes/TlaScratchAllocation.cpp · git: `grep -in 'liveness\|reuse\|free\|dealloc\|lifetime' csrc/mlir/lib/Passes/TlaScratchAllocation.cpp csrc/mlir/lib/Passes/TlaLowerPtrPass.cpp` → no matches |
| <a name="ref-12"></a>[12] | `tla.kernel` decorator — "By default, local synchronization remains explicit. `auto_sync=\"v0\"` enables the first version of automatic local mutex insertion."; accepts only `None` or `"v0"`; attribute set at `execution_lowering.py:251` | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/catlass/dsl.py#L109-L123 |
| <a name="ref-13"></a>[13] | auto_sync test coverage — `tests/test_local_autosync.py` plus 10 lit tests: `auto-mutex-alias-root.mlir`, `auto-mutex-diagnostics.test`, `auto-mutex-dynamic-select.mlir`, `auto-mutex-id-spaces.test`, `auto-mutex-instruction-pipes.mlir`, `auto-mutex-mixed-split.mlir`, `auto-mutex-vec-func.mlir`, `mutex-auto-id-error.mlir`, `mutex-control-flow-lowering.mlir`, `mutex-roundtrip.mlir` | https://gitcode.com/cann/catlass/tree/dsl/python/tla_dsl/tests |
| <a name="ref-14"></a>[14] | CATLASS DSL README — states that CATLASS-template-style higher-level wrapping, to match the C++ template development experience, remains future work | https://gitcode.com/cann/catlass/blob/dsl/python/tla_dsl/README.md |
