# PyAsc2 API Design

## 1. Requirements

- Express kernels in terms of **tensors** (ND-arrays in global memory) and **tiles** (fixed-shape chunks in on-chip memory). Buffer addresses, TPipe/TQue lifecycles, and synchronization barriers are not exposed to the user.
- Provide NumPy-like syntax for arithmetic, reductions, shape manipulation, masking, and atomics.
- **[Main engineering challenge]** Automate synchronization insertion, UB memory allocation, and ping-pong optimization through compiler passes and compiler hints.
- Target hardware: Ascend 910B/C, 950.

## 2. Key Challenges for Ascend NPU

Ascend 910B AI Core has two types of compute cores:
- **AIC** (Cube) — matrix/tensor operations
- **AIV** (Vector) — vector operations; 1 AIC : 2 AIV ratio on 910B

Each core has its own MTE (DMA), Scalar Unit, and Compute Unit.
AIC and AIV have no direct interconnect — data exchange goes through L2/Global Memory.

### 2.1 Synchronization Insertion

MTE and Compute Unit within each core run in parallel. Explicit `set_flag`/`wait_flag`
barriers must be inserted between them. Missing barriers cause silent data hazards;
redundant barriers cause stalls. Must be fully automated by the compiler.

### 2.2 Ping-Pong (Double Buffering)

MTE loads for tile N+1 must overlap with Compute on tile N.
Requires the compiler to split the loop body, hoist loads, and insert correct barriers —
without user annotation.

### 2.3 UB Memory Allocation and Reuse

On-chip UB (Unified Buffer) is ~256 KB per Da Vinci core (910B).
After loop unrolling, many tile SSA values can be live simultaneously.
Compiler must compute liveness and reuse freed UB regions.
Tile shapes are statically known at JIT time; total live footprint must be
validated at compile time.

### 2.4 Hardware Differences: 910B/C vs 950

> *Partial information — to be updated as more documentation becomes available.*

**910C** — two 910B dies in one package. Same Da Vinci architecture, AIC/AIV model unchanged.

**950 (Ascend A5)** — new architecture with changes relevant to DSL design:

- **AIC↔AIV data transfer via TPUSH/TPOP**: hardware instructions implementing
  a tag-based dual-channel FIFO between Cube and Vector cores. On 910B/C
  (PLATFORM_A2A3), ring buffer lives in Global Memory — DMA in and out.
  On 950 (PLATFORM_A5), ring buffer lives in consumer's on-chip SRAM —
  **zero-copy**. This fundamentally changes the cost of Cube↔Vector pipelines.
- **Memory access granularity**: 512 bytes → 128 bytes.
- **New data formats**: MXFP4, MXFP8, HiF8 (in addition to FP16/BF16/INT8).
- **Pipeline tuning**: different buffer allocation strategy and pipeline depth
  vs. A2/A3 (confirmed in PTO FA kernel for A5).
- **Sync model**: `set_flag`/`wait_flag` between PIPE_MTE2 and PIPE_V are still
  explicit even on A5 (confirmed in PTO engram kernel source).

## 3. Programming Model Analysis

### 3.1 AscendC

AscendC is the official C++ kernel language for Ascend NPU and the compilation
target for PyAsc2. Understanding how it handles the three key challenges defines
the baseline that PyAsc2 must improve upon.

#### Sync Insertion

AscendC exposes synchronization directly to the user via `TPipe` and `TQue`.
Every data transfer requires explicit `EnQue`/`DeQue` calls; every pipeline
stage boundary requires explicit `SetEventId`/`WaitEventId`:

```cpp
TPipe pipe;
TQue<QuePosition::VECIN, 2> inQueue;
TQue<QuePosition::VECOUT, 1> outQueue;
pipe.InitBuffer(inQueue, 2, TILE_SIZE);    // TPipe allocates UB memory for TQue buffers
pipe.InitBuffer(outQueue, 1, TILE_SIZE);

// MTE2 stage: load
LocalTensor<half> tile = inQueue.AllocTensor<half>();
DataCopy(tile, gm_src[offset], TILE_SIZE);
inQueue.EnQue(tile);              // signal: load done

// Vector stage: compute
LocalTensor<half> tile = inQueue.DeQue<half>();   // wait: load done
Add(out, tile, tile2, TILE_SIZE);
outQueue.EnQue(out);

// MTE3 stage: store
LocalTensor<half> out = outQueue.DeQue<half>();
DataCopy(gm_dst[offset], out, TILE_SIZE);
outQueue.FreeTensor(out);

// Cross-unit sync (e.g. between independent pipelines)
pipe.SetEventId(EVENT_ID0);       // producer signals
pipe.WaitEventId(EVENT_ID0);      // consumer waits
```
Source: Ascend C Operator Development Guide, CANN 8.0 https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html

Missing or misplaced EnQue/DeQue / SetEventId/WaitEventId causes silent data hazards. AscendC does not insert any barriers automatically.

**Conclusion**: AscendC doesn't solve the challenge — it exposes it. The user is responsible for every barrier manually. This is a source of bugs and boilerplate.

#### Ping-pong

```cpp
// Queue depth=2: two UB slots (ping + pong)
// Depth is set in TWO places — they must match:
constexpr int QUEUE_DEPTH = 2;

TQue<QuePosition::VECIN, QUEUE_DEPTH> inQueue;   // (1) template param
pipe.InitBuffer(inQueue, QUEUE_DEPTH, TILE_SIZE); // (2) runtime init

for (int i = 0; i < num_tiles; i++) {
    // MTE2: load tile i+1 while Vector computes tile i
    LocalTensor<half> tile = inQueue.AllocTensor<half>();
    DataCopy(tile, gm_src[(i+1) * TILE_SIZE], TILE_SIZE);
    inQueue.EnQue(tile);

    // Vector: compute tile i
    LocalTensor<half> cur = inQueue.DeQue<half>();
    Add(out, cur, cur2, TILE_SIZE);
    inQueue.FreeTensor(cur);
}
```
Depth 2 = two slots in UB. While Vector works on ping, MTE2 loads into pong. But:

User chooses queue depth manually
User structures the loop manually to achieve overlap
Compiler doesn't help

**Conclusion**: AscendC supports ping-pong, but requires manual orchestration. No automatic loop body partitioning.

#### UB Memory Allocation and Reuse
TBD


### 3.2 Triton

### 3.3 cuTile

### 3.4 TileLang-Ascend

### 3.5 Triton-Ascend

### 3.6 PyAsc v3

## 4. Key Design Decisions

## 5. API Specification

## 6. References

| # | Description | URL |
|---|-------------|-----|
| 1 | PyAsc2 HLD (internal) | https://gitcode.com/compiler-team/pyasc/pull/99 |
| 2 | PyPTO — tile-based framework for Ascend (official) | https://github.com/hw-native-sys/pypto |
| 3 | PyPTO official repo (GitCode/CANN) | https://gitcode.com/cann/pypto/ |
| 4 | PTO ISA — TPUSH/TPOP protocol | https://github.com/hw-native-sys/pypto/blob/main/docs/en/reference/pto-isa/01-tpush_tpop.md |
| 5 | PTO Tile Library (pto-isa) | https://github.com/hengliao1972/pto-isa |
| 6 | PTO A5 kernel examples | https://github.com/hengliao1972/pto-isa/tree/main/kernels/manual/a5 |
| 7 | PTO A5 Flash Attention kernel | https://github.com/hengliao1972/pto-isa/blob/main/kernels/manual/a5/flash_atten/README.md |
| 8 | PTO engram SIMT kernel (A5) | https://github.com/hengliao1972/pto-isa/blob/main/kernels/manual/a5/engram_simt/engram-simt_kernel.cpp |
| 9 | Ascend 910B architecture overview | https://arxiv.org/html/2505.15112v1 |
