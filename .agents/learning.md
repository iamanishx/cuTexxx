# Learning CuTe DSL + NVIDIA GPU Internals — A Ground-Up Guide

This is a self-contained study guide that goes from **GPU architecture →
NVIDIA jargon → the CUDA execution & memory model → tensor cores → the CuTe
DSL (layouts, partitioning, tiled MMA)**, and ends with a concrete learning
path. It is written to pair with the runnable code in this repo, and most
abstract ideas are tied back to lines in those files.

> **Companion files in this folder / repo:**
> - `matmul_cute.py` — the tiled SIMT GEMM this guide annotates.
> - `softmax_cute.py` — simple one-thread-per-row softmax (done).
> - `softmax_reduce_cute.py` — the shared-memory reduction version (your current exercise).
> - `kernel_explained.md` — a slow, plain-language, line-by-line of the GEMM kernel.
>
> **Reading order:** skim Part 1-4 first (architecture + jargon), then study
> Part 5 (CuTe layout algebra, the genuinely new material), then read Part 6
> (the annotated kernel). **Part 10 captures the exact questions that made it
> click for you** in plain language, so jump there whenever a concept feels
> slippery. Part 8 + Part 11 are your practice plan and personal roadmap
> (you are training ~6 hours/day, so the timeline is tuned for that).

---

## Table of Contents
1. [Why GPUs look the way they do](#1-why-gpus-look-the-way-they-do)
2. [The hardware hierarchy (and the jargon)](#2-the-hardware-hierarchy-and-the-jargon)
3. [The execution model: grid → CTA → warp → thread](#3-the-execution-model-grid--cta--warp--thread)
4. [The memory hierarchy](#4-the-memory-hierarchy)
5. [Tensor Cores & MMA](#5-tensor-cores--mma)
6. [CuTe DSL: layouts are the whole game](#6-cute-dsl-layouts-are-the-whole-game)
7. [Annotated walkthrough of `matmul_cute.py`](#7-annotated-walkthrough-of-matmul_cutepy)
8. [How to actually learn this](#8-how-to-actually-learn-this)
9. [Glossary (quick reference)](#9-glossary-quick-reference)
10. [The questions that made it click](#10-the-questions-that-made-it-click)
11. [Your build log and roadmap](#11-your-build-log-and-roadmap)

---

## 1. Why GPUs look the way they do

A CPU is built for **latency**: a few fat cores, huge caches, branch
prediction — finish one task as fast as possible.

A GPU is built for **throughput**: thousands of small ALUs that do the *same*
operation on *different* data. The design bet is:

- Don't try to avoid memory latency — **hide** it by having thousands of
  threads ready, and instantly switching to another group when one stalls.
- Spend transistors on **math units**, not on caches/branch predictors.
- Make threads cheap so you can launch millions.

Everything below (warps, occupancy, coalescing, tensor cores) is a consequence
of that one design choice: **maximize useful math per byte read from memory.**

This is also *why tiling exists* and why CuTe is obsessed with layouts: the
bottleneck is almost never the math, it's **moving data**. CuTe is a language
for describing data movement and thread-to-data mapping precisely.

---

## 2. The hardware hierarchy (and the jargon)

From the chip down to one lane:

```
GPU (e.g. RTX 4050, "Ada Lovelace" / sm_89)
 └── GPC  (Graphics Processing Cluster)        ← coarse partition, ignore for compute
      └── SM  (Streaming Multiprocessor)        ← THE core unit. ~dozens per GPU
           ├── CUDA cores (FP32/INT ALUs)       ← scalar "SIMT" math
           ├── Tensor Cores                     ← matrix-multiply units
           ├── Warp schedulers (usually 4)      ← pick a ready warp each cycle
           ├── Register file (huge, e.g. 64K x 32-bit)
           ├── Shared memory / L1 (configurable split)
           └── Load/Store + Special Function Units
```

### Key terms
- **SM (Streaming Multiprocessor):** the fundamental compute engine. A GPU is
  basically "N copies of an SM" plus memory controllers. Your RTX 4050 has ~20
  SMs. **A thread block (CTA) runs entirely on one SM.**
- **CUDA core:** a scalar ALU (one FP32 multiply-add per cycle, roughly). The
  marketing "thousands of CUDA cores" = total ALUs across all SMs.
- **Tensor Core:** a dedicated unit that does a small **matrix** multiply-
  accumulate per instruction (e.g. 16×8×16) — orders of magnitude more FLOPs
  than CUDA cores for matmul. See Part 5.
- **Compute Capability / `sm_XX`:** the hardware feature/ISA version.
  `sm_89` = Ada (RTX 40-series). `sm_80/86` = Ampere. `sm_90` = Hopper.
  `sm_100` = Blackwell. CuTe code often branches on this because newer SMs add
  features (TMA, clusters, `tcgen05`, etc.).
- **Architecture codenames:** Volta (sm_70) → Turing (sm_75) → Ampere (sm_80/86)
  → Ada (sm_89) → Hopper (sm_90) → Blackwell (sm_100/120). Each adds tensor-core
  and memory features.

---

## 3. The execution model: grid → CTA → warp → thread

This is the mental model you launch into. From biggest to smallest:

```
Grid                         ← the whole kernel launch
 └── CTA (= thread block)     ← runs on 1 SM, shares SMEM, can __syncthreads()
      └── Warp (32 threads)   ← the REAL scheduling unit; runs in lockstep (SIMT)
           └── Thread (lane)  ← one execution stream; lane = threadIdx % 32
```

### CTA = Cooperative Thread Array = thread block
Two names for the same thing:
- **"thread block"** = CUDA programming term (`blockIdx`, `blockDim`).
- **"CTA"** = hardware/PTX/CUTLASS term — you'll see this everywhere in CuTe.

A CTA:
- runs on exactly **one SM** (never split across SMs),
- can **share data** via shared memory,
- can **synchronize** all its threads (`cute.arch.sync_threads()`),
- is internally chopped into **warps** for scheduling.

### Warp = 32 threads in lockstep (SIMT)
- The hardware schedules **warps**, not individual threads. All 32 lanes
  execute the *same instruction* each step (**SIMT** = Single Instruction,
  Multiple Thread).
- **Warp divergence:** if lanes take different `if` branches, the warp executes
  both paths serially with some lanes masked off → slow. Avoid data-dependent
  branching inside a warp.
- `lane = threadIdx.x % 32`, `warp_id = threadIdx.x // 32`.
- Warps are why memory **coalescing** matters: if the 32 lanes read 32
  consecutive addresses, the hardware does it in one transaction.

### Cluster (Hopper+ / sm_90+, optional)
A level *above* the CTA: a **cluster** is a group of CTAs that can access each
other's shared memory (distributed shared memory) and sync. You don't need it
for a basic GEMM; just know the modern hierarchy is:
`Grid → Cluster → CTA → Warp → Thread`.

### How it maps to a launch
```python
self.kernel(...).launch(
    grid =[grid_m, grid_n, 1],  # number of CTAs  (a 3D grid)
    block=[256,    1,    1],    # threads per CTA (a 3D block) = 8 warps
    stream=stream,
)
```
Total threads = `(grid_m*grid_n) * 256`. Inside the kernel you locate yourself:
```python
tidx, _, _   = cute.arch.thread_idx()  # threadIdx  (0..255 here)
bidx, bidy,_ = cute.arch.block_idx()   # blockIdx   (which CTA)
```
**CuTe does NOT change this** — it lowers to a normal `<<<grid, block>>>`
launch. Grid+block is exactly CUDA.

### Occupancy & latency hiding
- An SM holds many resident warps at once. When one warp stalls on a memory
  load, the scheduler instantly runs another ready warp → latency hidden.
- **Occupancy** = (resident warps) / (max warps the SM supports). Limited by
  registers/thread and shared memory/CTA. Higher isn't always better, but too
  low means latency isn't hidden.

---

## 4. The memory hierarchy

Speed and size are inversely related. Performance work = keeping hot data in
the fast levels.

```
Registers      per-thread     fastest, tiny     (e.g. up to 255/thread)
   │
Shared mem     per-CTA        very fast, ~tens-hundreds of KB, programmer-managed
(SMEM / L1)                   ← "scratchpad" you explicitly fill; bank-conflict aware
   │
L2 cache       whole GPU      fast-ish, MBs, automatic
   │
Global (HBM)   whole GPU      large (GBs), SLOW (hundreds of cycles)
```

### Terms you must know
- **Global memory (HBM/DRAM):** big, slow, where your input tensors live. Every
  access costs hundreds of cycles. The enemy.
- **Coalescing:** if the 32 lanes of a warp touch consecutive addresses, the
  hardware merges them into one wide transaction. Non-coalesced (strided/random)
  access wastes bandwidth. Layout choices control this.
- **Shared memory (SMEM):** on-chip scratchpad shared by a CTA. You *manually*
  stage tiles here (`SmemAllocator`) so threads reuse data instead of re-reading
  global memory. This is the core of fast GEMM.
- **Bank conflicts:** SMEM is split into 32 banks. If multiple lanes hit the
  same bank (different address), accesses serialize. **Swizzling** layouts avoids
  this — CuTe has `cute.make_swizzle(...)`.
- **Registers:** per-thread private storage, fastest. Your accumulator `acc`
  lives here. Too many registers/thread → lower occupancy ("register pressure").
- **Fragment (CuTe term):** a small per-thread tensor held in registers — e.g.
  the slice of A/B/C a single thread owns. `make_fragment_C`, `partition_A`.
- **`cp.async` (cp_async):** Ampere+ instruction to copy global→shared
  *asynchronously*, so threads compute while the next tile loads ("pipelining").
- **TMA (Tensor Memory Accelerator, Hopper+):** a hardware engine that bulk-
  copies whole tiles global↔shared with one instruction. Replaces hand-written
  copy loops on sm_90+.

### The standard GEMM data flow (what fast kernels do)
```
Global A,B ──cp.async/TMA──▶ Shared memory tiles ──▶ Registers (fragments)
                                                          │
                                                   Tensor Core / FMA
                                                          │
Registers (accumulator) ──────────────────────────▶ Global C
```
Your current `matmul_cute.py` is the *simple* version: it skips the shared-
memory stage (reads tiles straight into registers and uses FMA). That's the
right first step — get it correct, then add SMEM + tensor cores for speed.

---

## 5. Tensor Cores & MMA

### What they are
Tensor Cores do a **matrix** multiply-accumulate `D = A·B + C` per instruction,
across a **warp**, on small fixed shapes. Compare:
- A CUDA core: ~1 scalar fused-multiply-add (FMA) per cycle.
- A Tensor Core MMA: a whole `16×8×16` (M×N×K) tile of FMAs per instruction.

That's why tensor cores are ~10× the throughput for matmul — but they:
- work on **fixed shapes** (e.g. 16×8×16, 16×8×8),
- are **warp-level** (the 32 lanes cooperate; each lane holds a piece of the
  operands in a specific layout),
- support specific **dtypes**: fp16, bf16, tf32, fp8, int8, fp4 (newer SMs).

### MMA jargon
- **MMA:** Matrix Multiply-Accumulate. The operation `D = A·B + C`.
- **MMA atom:** the smallest hardware MMA instruction (a shape + dtype). In CuTe:
  - `MmaUniversalOp(Float32)` = a **SIMT FMA** "atom" (no tensor core) — what
    your file uses. Works on every GPU.
  - `MmaF16BF16Op(in_dtype, acc_dtype, (16,8,16))` = a real **tensor-core** atom.
- **Tiled MMA:** many atoms tiled across threads/warps to cover a bigger tile.
  Built with `cute.make_tiled_mma(op, atom_layout_mnk)`.
- **Accumulator:** where partial sums live (usually fp32 even for fp16 inputs,
  for accuracy). In code: `acc = tiled_mma.make_fragment_C(...)`.

### Why CuTe leans on layouts here
Tensor cores require operands to be in a *very specific* per-lane register
layout. Hand-coding that index math is miserable. CuTe's tiled-MMA + layout
algebra **derive** the correct per-thread fragment layout for you — that's the
big practical payoff of the layout abstraction.

---

## 6. CuTe DSL: layouts are the whole game

CuTe = "**Cu**da **Te**nsors." The DSL is a Python front-end (lowering to
MLIR → PTX/SASS) for writing kernels. Its one big idea:

> A **Layout** is a function from logical coordinates → memory offsets, and it's
> a first-class, composable object. Tiling, thread assignment, swizzling — all
> are layout operations.

Everything else (`local_tile`, `partition_A`, `make_tiled_mma`) is built on
layout algebra. **This is the part that's actually new** vs. CUDA C++; invest
your time here.

### 6.1 Shape, Stride, Layout
A `Layout` = a `Shape` (logical extents) + a `Stride` (steps in memory).
```
Layout (4, 8) : (8, 1)      # 4x8 row-major: element (i,j) lives at i*8 + j
Layout (4, 8) : (1, 4)      # 4x8 col-major: element (i,j) lives at i*1 + j*4
```
- `crd2idx(coord, layout)` maps a coordinate → linear offset.
- Shapes/strides can be **nested** (hierarchical), e.g. `((2,2),4):((1,2),4)`.
  This nesting is how CuTe represents tiles-of-tiles cleanly.
- **Static vs dynamic:** known-at-compile-time sizes (great for the optimizer)
  vs runtime sizes. CuTe lets you mix.

### 6.2 Tensor = Pointer + Layout
A `cute.Tensor` is just "where the data starts" + "a Layout describing it."
`from_dlpack(cupy_array)` builds one from a GPU array's pointer + shape/stride.
No data is copied — it's a *view*.

### 6.3 The layout operations you'll use
- **`local_tile(T, tiler, coord)`** — carve a big tensor into tiles and select
  one (by CTA coordinate). "Give CTA (bidx,bidy) its 128×128 chunk."
- **`zipped_divide` / `logical_divide`** — the general tiling primitives
  `local_tile` is built on.
- **`partition_A/B/C(tensor)`** (via `thr_mma`/`thr_copy`) — split a tile across
  threads and return **this thread's** slice. The thread→data map.
- **`make_fragment_A/B/C`** — allocate the per-thread register tensor matching a
  partition.
- **`make_tiled_mma(op, atom_layout_mnk)`** — describe how atoms (and thus
  threads/warps) tile to cover a block-level matmul.
- **`make_tiled_copy` / `make_copy_atom` / `cute.copy`** — describe and perform
  data movement (global↔shared↔register), again as layouts.
- **`make_swizzle`** — permute SMEM layout to kill bank conflicts.
- **`cute.gemm(tiled_mma, D, A, B, C)`** — run the (tiled) MMA: `D = A·B + C`.

### 6.4 TV layout (Thread-Value layout) — the key concept
A **TV layout** maps `(thread_id, value_id) → coordinate in the tile`. It is
*the* answer to your earlier question "can I control which thread/warp works on
what?" — **yes, you write it as a TV layout.**

- `atom_layout = make_layout((16,16,1), stride=(16,1,0))` in your file says:
  arrange 256 threads as a 16×16 grid. Combined with the tile size, this fixes
  exactly which output elements each thread (and therefore each warp) owns.
- `tiled_mma.get_slice(tidx)` = "I am thread `tidx`; resolve the TV layout and
  hand me my slice." Change the layout → change the assignment.

### 6.5 The decorators
- **`@cute.jit`** — host-side function; traced to IR and compiled on first call.
  Sets up layouts/MMA and launches the kernel. Args are annotated with types
  (`cute.Tensor`, `cuda.CUstream`) so they lower correctly.
- **`@cute.kernel`** — the device function one CTA runs (the GPU code).
- **`cute.compile(obj, *args)`** — pre-compile to cache the kernel across launches
  (skip re-tracing every call).

### 6.6 SIMT vs Tensor-Core path (when to use which)
| | SIMT (`MmaUniversalOp`) | Tensor Core (`MmaF16BF16Op`, …) |
|---|---|---|
| Hardware | any GPU | Volta+ (better each gen) |
| Dtypes | fp32/fp64/… | fp16/bf16/tf32/fp8/int8/… |
| Speed | baseline | ~10× for matmul |
| Complexity | low (your file) | higher (fixed shapes, fragment layouts) |
| Learn it | **first** | after SMEM staging |

---

## 7. Annotated walkthrough of `matmul_cute.py`

This is the SIMT GEMM in this repo. Map each piece to the concepts above.

```python
# ---- Host setup (@cute.jit): build the tiled MMA + launch ----
op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)        # SIMT FMA "atom" (no tensor core)
atoms_layout = cute.make_layout((16,16,1), (16,1,0))   # TV layout: 256 threads as 16x16
tiled_mma = cute.make_tiled_mma(op, atoms_layout)      # how threads tile the block matmul

grid_m = (M + bM - 1)//bM;  grid_n = (N + bN - 1)//bN  # one CTA per 128x128 output tile
self.kernel(mA, mB, mC, tiled_mma).launch(
    grid=[grid_m, grid_n, 1],   # 2x2 = 4 CTAs
    block=[256, 1, 1],          # 256 threads = 8 warps per CTA
    stream=stream)
```

```python
# ---- Device kernel (@cute.kernel): one CTA's work ----
tidx,_,_   = cute.arch.thread_idx()    # my lane within the CTA
bidx,bidy,_= cute.arch.block_idx()     # which output tile this CTA owns

gA = cute.local_tile(mA, (bM,bK), (bidx,None))  # my CTA's rows of A: (bM,bK,k_tiles)
gB = cute.local_tile(mB, (bN,bK), (bidy,None))  # my CTA's cols of B: (bN,bK,k_tiles)
gC = cute.local_tile(mC, (bM,bN), (bidx,bidy))  # my CTA's output tile: (bM,bN)

thr_mma = tiled_mma.get_slice(tidx)    # resolve TV layout → MY thread's view
tCgC = thr_mma.partition_C(gC)         # my output elements
acc  = tiled_mma.make_fragment_C(tCgC) # my accumulator in REGISTERS
acc.fill(0.0)

for k in cutlass.range(cute.size(gA, mode=[2])):   # K-loop (the contraction)
    tCrA = thr_mma.partition_A(gA[None,None,k])     # my slice of this A k-tile
    tCrB = thr_mma.partition_B(gB[None,None,k])     # my slice of this B k-tile
    cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)      # acc += A_tile · B_tile

copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
cute.copy(copy_atom, acc, tCgC)        # write registers → global C
```

**Three subtleties this file encodes (common beginner traps):**
1. `cute.gemm` contracts the **trailing mode of both A and B**, so B is passed
   as `(N,K)` — that's why `main()` does `from_dlpack(b.T)`.
2. The `stream` arg is a real `cuda.CUstream` (annotated in `@cute.jit`), not a
   raw int, or IR lowering fails.
3. Inputs are generated with NumPy then `cp.asarray`'d, because cupy's RNG needs
   `libcurand`, which the cutlass-dsl wheel doesn't ship.

**What's intentionally missing (the speed work, for later):**
- No shared-memory staging (`SmemAllocator` + `cp.async`) → re-reads global.
- No tensor cores (uses FMA, not `MmaF16BF16Op`).
- No pipelining (multi-stage SMEM double-buffering).

---

## 8. How to actually learn this

You do **not** need to master CUDA C++ first. If you already know GPU
architecture (SMs, warps, memory hierarchy), the expensive part is done. The
genuinely new subject is **CuTe layout algebra.**

### The project ladder (each step adds exactly ONE new idea)

Do them in order. Every step reuses the skills from the one before, so do not
skip. Keep a NumPy reference + `assert_allclose` on every single one.

1. **Tiled SIMT matmul** (done) — layouts, `local_tile`, `partition`, tiled MMA.
2. **Simple softmax** (done) — element access, per-thread row loops, stable formula.
3. **Reduction softmax** (in progress) — shared memory + `sync_threads` + a tree
   reduction. One CTA per row, threads cooperate. This is the skill that unlocks
   almost everything else.
4. **Warp-shuffle reduction** — replace the smem tree with
   `cute.arch.shuffle_sync_down` for the intra-warp part. Teaches warp-level ops.
5. **LayerNorm / RMSNorm** — same reduction skeleton, but two stats (mean + var).
   Cements the pattern so it becomes muscle memory.
6. **Matmul with shared-memory staging** — add an SMEM tile + `cp.async` to step 1.
   This is the big performance jump, and where coalescing/bank-conflicts matter.
7. **Tensor-core matmul** — swap `MmaUniversalOp` for `MmaF16BF16Op` (16x8x16),
   fp16 in / fp32 accumulator. This is where you start *managing warps* (the MMA
   atom is now warp-level, see Part 10).
8. **Fused softmax + matmul, a tiny attention** — combine everything. This is
   basically a baby FlashAttention. After this, the official examples read easily.

### Timeline at ~6 hours/day (your pace)

| Milestone | What you can do | Calendar time |
|---|---|---|
| **Comfortable** | write correct SIMT kernels (steps 3-5), read most examples | ~1 week |
| **Productive** | shared-memory + tensor-core GEMM, debug layouts confidently (steps 6-7) | ~3-4 weeks |
| **Proficient** | fuse ops, write attention, reason about perf with Nsight (step 8+) | ~6-8 weeks |
| **Mastery** | match/extend library kernels, pipelining, Hopper/Blackwell (TMA, clusters, wgmma) | ~4-6 months |

The single biggest unlock is fluency in **layout algebra** (the `Shape:Stride`,
`local_tile`, `partition`, TV-layout reasoning). Once layouts feel obvious,
usually around steps 3-6, everything else is incremental. At 6 hours/day that
fluency is roughly **2-3 focused weeks**.

### A concrete first two weeks (6 hr/day)

- **Days 1-2:** layouts in isolation on CPU (`make_layout`, `crd2idx`,
  `local_tile`, `cute.pretty_str`). Predict offsets by hand, then verify.
- **Days 3-4:** re-derive every line of `matmul_cute.py`; change block sizes and
  `atoms_layout` and explain each thing that breaks.
- **Days 5-7:** finish `softmax_reduce_cute.py`, then rewrite its reduction with
  warp shuffles (step 4).
- **Days 8-10:** LayerNorm/RMSNorm from scratch (step 5).
- **Days 11-14:** add shared-memory staging to the matmul (step 6); profile it
  with Nsight Compute and compare against the naive version.

### Resources
- **CUTLASS repo examples:** `examples/python/CuTeDSL/` (SIMT, Ampere, Hopper,
  Blackwell GEMMs) — the canonical, runnable references.
- **CuTe C++ docs** (`media/docs/cute/` in the CUTLASS repo): the layout-algebra
  writeups apply directly to the DSL — best material on layouts anywhere.
- **CUTLASS DSL docs site:** the "GEMM Kernel in CuTe DSL" and "Quickstart"
  pages.
- **NVIDIA CUDA C++ Programming Guide:** read the execution & memory model
  chapters only (skip syntax). Plus the PTX ISA doc for MMA/`cp.async`/TMA
  definitions when you need exact semantics.
- **Nsight Compute:** to *see* occupancy, memory throughput, bank conflicts.

### Practice principle
Always keep a **NumPy reference + `assert_allclose`** (like `matmul_cute.py`
does). Correctness first, then make it fast. Change one thing at a time.

---

## 9. Glossary (quick reference)

| Term | Meaning |
|---|---|
| **SM** | Streaming Multiprocessor — the core compute engine; a CTA runs on one SM |
| **CTA** | Cooperative Thread Array = thread block (hardware name) |
| **Warp** | 32 threads executed in lockstep (SIMT); the real scheduling unit |
| **Lane** | a thread's index within its warp (`threadIdx % 32`) |
| **SIMT** | Single Instruction, Multiple Thread — warp executes one instr across 32 lanes |
| **Cluster** | group of CTAs sharing memory (Hopper+/sm_90+) |
| **Grid** | all CTAs of one kernel launch |
| **Occupancy** | resident warps ÷ max warps an SM supports |
| **Divergence** | lanes in a warp taking different branches → serialized → slow |
| **Coalescing** | merging 32 consecutive lane accesses into one memory transaction |
| **Global / HBM** | large, slow off-chip DRAM where tensors live |
| **Shared mem / SMEM** | on-chip per-CTA scratchpad, programmer-managed |
| **Bank conflict** | multiple lanes hitting the same SMEM bank → serialized |
| **Swizzle** | layout permutation to avoid bank conflicts |
| **Register / Fragment** | per-thread fastest storage; fragment = per-thread tensor |
| **cp.async** | async global→shared copy (Ampere+) for pipelining |
| **TMA** | Tensor Memory Accelerator — bulk tile copy engine (Hopper+) |
| **Tensor Core** | warp-level matrix-multiply unit (fixed shapes, fp16/bf16/…) |
| **MMA** | Matrix Multiply-Accumulate (`D = A·B + C`) |
| **MMA atom** | the smallest hardware MMA instruction (shape + dtype) |
| **Tiled MMA** | atoms tiled across threads/warps to cover a bigger tile |
| **Accumulator** | register tensor holding partial sums (usually fp32) |
| **Layout** | `Shape:Stride` — function from logical coord → memory offset |
| **Shape / Stride** | logical extents / memory step per mode |
| **Tensor (CuTe)** | pointer + Layout (a view, no copy) |
| **TV layout** | Thread-Value layout: maps (thread, value) → tile coordinate |
| **local_tile** | carve a tensor into tiles and select one by coordinate |
| **partition_A/B/C** | return this thread's slice of A/B/C |
| **sm_XX / CC** | compute capability; sm_89 = Ada (RTX 40), sm_90 = Hopper, sm_100 = Blackwell |
| **CuTe** | "Cuda Tensors"; layout-algebra library + DSL inside CUTLASS |
| **CUTLASS** | NVIDIA's open template library for GEMM/conv kernels |
| **@cute.jit / @cute.kernel** | host (trace+launch) / device (per-CTA GPU code) functions |

---

## 10. The questions that made it click

These are the exact sticking points you hit, written the way that finally made
sense. If a concept feels slippery later, re-read the relevant one.

### Why does a 256x256 by 256x256 matmul need 4 CTAs, not 2?

Because the output is split **both ways**. We chose 128x128 output tiles, so the
256x256 result is cut into a 2x2 grid: 2 row-bands times 2 col-bands = 4 tiles,
one per CTA. The trap is thinking a CTA reads a 128x256 strip so 256/128 = 2.
That 256 is the **K dimension, which gets summed away** and never appears in the
output. Each CTA outputs 128x128, and `(256x256) / (128x128) = 4`.

The clean mantra: **rows from A times cols from B = your CTA's output tile.**
Number of CTAs = `(M / BLOCK_M) * (N / BLOCK_N)`. K never enters that count; it is
the loop *inside* each CTA.

### M, N, K is not a 3D matmul

It is a plain **2D** matmul. M, N, K are three *sizes* describing one 2D product
(`A` is M x K, `B` is K x N, `C` is M x N). A real 3D/batched matmul would be
`(B, M, K)` times `(B, K, N)`, and you would put the batch on the grid's z-axis.
The unused `1` in `grid=[grid_m, grid_n, 1]` is exactly where a batch count would
go if you ever wanted batching.

### M and N are parallel; K is a reduction

This is the whole shape of GEMM:
- **M and N** are spread across CTAs and threads (the 128x128 output tile).
- **K** is walked in a serial loop inside each CTA, in chunks of `BLOCK_K`, and
  accumulated. That is why the output tiler is `(128,128)` but the input tilers
  are `(128,8)`: they tile different axes, and K is kept small because you loop
  over it (256/8 = 32 steps).

### `local_tile` coord: integer vs None

`local_tile(T, tiler, coord)` scores the matrix into a grid of tiles, then
`coord` selects:
- an **integer** picks exactly one tile index (that axis collapses),
- **None** keeps *all* tiles along that axis, stacked as an extra dimension to
  loop over later.

That is why `gA` and `gB` are `(128, 8, 32)` (the None kept all 32 K-chunks for
the loop) but `gC` is `(128, 128)` (both coords fixed, written once, nothing to
loop). And the returned views are just pointer + layout, so **no data moves**
until `partition_*` reads it in the K-loop.

### What does `mode=[2]` mean in `cute.size(gA, mode=[2])`?

A **mode is just an axis** (numbered from 0). `gA` is `(128, 8, 32)`, so mode 0 =
128, mode 1 = 8, mode 2 = 32. So this reads the length of the third axis = 32 =
the number of K-chunks = how many times the K-loop runs. Reading it (instead of
hardcoding 32) keeps the loop correct if you change K or BLOCK_K. The word
"mode" (not "axis") is used because layouts can be nested, and `[2]` is a list so
you can index into nested modes.

### Possible values of `k` in the loop

`cutlass.range(32)` yields `k = 0, 1, ... , 31` (32 values, 0-based, 32
excluded). Each `k` selects the k-th 128x8 chunk, covering K columns
`k*8 .. k*8+7`. Together the 32 chunks cover all 256 of K.

### Are there two different "grids"? Yes.

This confused you and it is worth nailing:
- **Launch grid** `grid=[grid_m, grid_n, 1]` = how many **CTAs**. Comes from
  `M/BLOCK_M` and `N/BLOCK_N`. This is the CUDA grid.
- **Thread layout** `make_layout((16,16,1), ...)` = how **threads sit inside one
  CTA**. This is the TV layout, and `make_layout` only controls *this*, not the
  launch grid.

The only link between them: the thread layout's total size (16x16 = 256) must
equal the threads-per-block you launch (`block=[256,1,1]`).

### What do `make_layout` and `make_tiled_mma` actually do?

- `make_layout((16,16,1), stride=(16,1,0))` arranges 256 threads as a 16x16 grid
  and assigns `thread_id = 16*m + n` (ids run across N first, then jump by 16 per
  M row). This is the thread-to-position map.
- `make_tiled_mma(op, that_layout)` fuses the single-FMA atom (`op`) with that
  16x16 arrangement into the `tiled_mma` object. That object is what later lets
  `get_slice(tidx)` and `partition_*` hand each thread its exact slice of A, B,
  C. Since the CTA tile is 128x128 and the layout is 16x16, each thread covers
  `128/16 x 128/16 = 8x8 = 64` output cells.

### Do we manage warps in CuTe?

It depends on the **MMA atom**, because the atom sets the granularity:
- **SIMT atom** (`MmaUniversalOp`, your kernel) = one **thread** per atom, so you
  arrange *threads* (16x16) and warps are implicit (warp = threads 0-31, etc.).
  You never write warp-specific code.
- **Tensor-core atom** (`MmaF16BF16Op`) = one **warp** per atom, so the
  `atom_layout_mnk` you pass to `make_tiled_mma` is a layout of *warps*, e.g.
  `(2,2,1)` = 4 warps. Now you are managing warps, just declaratively.

CuTe also has direct tools when you need them: `cute.arch.warp_idx`,
`lane_idx`, `shuffle_sync_*`, `sync_warp`, `ldmatrix`/`stmatrix`, and warpgroup
ops on Hopper. The CuTe philosophy is: express warp structure as a *layout*
rather than hand-written `if (warp_id == ...)` branches.

### Why no `cluster=` in the launch?

Thread-block **clusters** are a Hopper+ (sm_90+) feature. Your RTX 4050 is Ada
(sm_89), so the hierarchy you can use is just `Grid -> CTA -> Warp -> Thread`,
and the launch has exactly two levels (grid and block). On an H100 you would add
a cluster level (and TMA), which changes the launch signature and the copy code.

---

## 11. Your build log and roadmap

### Done
- `matmul_cute.py` is a tiled SIMT GEMM (256^3, 128x128 tiles, 16x16 threads),
  verified against NumPy. You can now explain every line.
- `softmax_cute.py` is a stable row-wise softmax, one thread per row, verified.

### In progress
- `softmax_reduce_cute.py` is one CTA per row with shared memory + a tree
  reduction. This is step 3 of the ladder. Watch for the classic bug: a missing
  `cute.arch.sync_threads()` between writing and reading the shared buffer, or
  reusing the buffer for the sum phase before all threads finished the max phase.

### Next (in order)
4. Warp-shuffle reduction (`shuffle_sync_down`) instead of the smem tree.
5. LayerNorm / RMSNorm (two stats, same skeleton).
6. Matmul with shared-memory staging + `cp.async` (the big perf jump).
7. Tensor-core matmul (`MmaF16BF16Op`, fp16 in / fp32 acc), your first real warp work.
8. Fused softmax + matmul = tiny attention (baby FlashAttention).

### Habits that will make you fast
- Always keep a NumPy reference and `assert_allclose`. Correctness first.
- Change one thing at a time, then re-run.
- When a kernel is wrong only *sometimes* or only for *some* rows, suspect a
  missing sync before anything else.
- After each working kernel, profile with Nsight Compute and connect the code to
  occupancy / memory throughput / bank conflicts.
- Re-read Part 5 (layout algebra) and Part 10 whenever something feels magic.

---

*Start at Part 5, re-read Part 6 and Part 10 until layouts and the kernel feel
obvious, finish the reduction softmax, then climb the ladder in Part 8.*

```mermaid
   flowchart TD
       START(["thread tidx in CTA (bidx,bidy)"]) --> ID

       subgraph ID["1 - locate myself"]
           T["tidx = thread_idx()  (0..255)"]
           B["bidx,bidy = block_idx()  (which C tile)"]
       end

       ID --> TILE

       subgraph TILE["2 - slice GLOBAL views (no copy)"]
           GA["gA = local_tile(mA,(128,8),(bidx,None))<br/>(128,
 8, 32)"]
           GB["gB = local_tile(mB,(128,8),(bidy,None))<br/>(128,
 8, 32)"]
           GC["gC =
 local_tile(mC,(128,128),(bidx,bidy))<br/>(128, 128)"]
       end

       TILE --> PART

       subgraph PART["3 - take MY slice (TV layout)"]
           SL["thr_mma = tiled_mma.get_slice(tidx)"]
           PC["tCgC = partition_C(gC)  → my 8x8 of C (global)"]
           ACC["acc = make_fragment_C(tCgC)  →
 REGISTERS<br/>acc.fill(0.0)"]
       end

       PART --> LOOP

       subgraph LOOP["4 - K-loop  (k = 0 .. 31)"]
           direction TB
           RA["tCrA = partition_A(gA[:,:,k])  → my A piece"]
           RB["tCrB = partition_B(gB[:,:,k])  → my B piece"]
           MM["cute.gemm: acc += tCrA · tCrB"]
           RA --> MM
           RB --> MM
           MM -->|"next k"| RA
       end

       LOOP --> WB

       subgraph WB["5 - write back"]
           CA["copy_atom = make_copy_atom(CopyUniversalOp,
 fp32)"]
           CP["cute.copy(copy_atom, acc, tCgC)<br/>registers →
 global C"]
           CA --> CP
       end

       WB --> DONE(["my 8x8 of C is done"])

       classDef reg fill:#dfe,stroke:#5b5;
       classDef glob fill:#fde,stroke:#b55;
       class ACC,RA,RB,MM reg;
       class GA,GB,GC,PC,CP glob;
 ```