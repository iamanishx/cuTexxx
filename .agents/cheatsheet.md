# CuTe DSL Cheatsheet (fast revision)

Plain-language "what does this actually do" for every CuTe thing you have used.
Tied to your files: `matmul_cute.py`, `softmax_cute.py`, `softmax_reduce_cute.py`,
and `tutorials/`. No fluff, made for re-reading.

Mental model first: **a kernel function runs on the CPU at build time to WRITE
the GPU program.** Plain Python (ints, `for`/`while`/`if` over python values)
steers code generation and unrolls. CuTe operations on tensors/registers become
the actual GPU instructions. See "Host vs device vs build time" at the bottom.

---

## 1. Layout algebra (the foundation)

A **Layout = a function from a logical coordinate to a memory offset.** Written
`Shape:Stride`. `offset = sum(coord[i] * stride[i])`.

| Call | What it actually does | Example |
|---|---|---|
| `cute.make_layout(shape, stride=...)` | builds a layout (coord -> offset function). If stride omitted, it is generated (column-major by default). | `L = make_layout((4,8), stride=(8,1))` then `L((2,3)) = 2*8+3 = 19` |
| `cute.size(x)` | total number of coordinates = product of the shape | `size((2,3),4) = 24` |
| `cute.size(x, mode=[i])` | length of **axis i only** ("mode" = axis, 0-based; nestable) | `size(gA, mode=[2])` = number of K-chunks (e.g. 32) |
| `cute.coalesce(L)` | merges contiguous modes and drops size-1 modes. **Offsets unchanged**, just a simpler shape. Helps vectorized memory. | `((4,2),8)` contiguous -> `(8,8)` mapping unchanged |
| `cute.composition(A, B)` | chain two mappings: feed B's output into A. `A o B`. Removes the middle coordinate. | thread coord -> tile coord -> offset, fused to thread coord -> offset |
| `cute.logical_divide(L, tiler)` | split a coord space into `(local, tile)`. The math behind tiling a big matrix into blocks. | `12:1` by `4` -> coord `(inner, tile)` mapping |
| `cute.complement(L, K)` | given a layout covering part of size K, returns the layout that sweeps the **rest**. Used to split work into (thread, value). | `4:1` under `16` -> `4:4` (the leftover stride-4 sweep) |

Key terms:
- **mode** = one axis of the layout (numbered from 0; can be nested).
- **stride** = how far you jump in memory when a coordinate increments by 1.
- **shape** = the valid coordinate ranges (does NOT know memory).

---

## 2. Tensors and views (no data copied)

A CuTe **Tensor = pointer + Layout** (a view). Slicing/tiling makes new views,
nothing moves until you actually load with a copy or partition.

| Call | What it actually does |
|---|---|
| `from_dlpack(arr)` | wrap a cupy/torch array as a CuTe tensor (a view over the same memory) |
| `from_dlpack(b.T)` | view the transpose. **Needed for B in GEMM** because `cute.gemm` contracts the trailing mode of BOTH A and B, so B must be passed as `(N,K)`. |
| `T[i, j]` | load one scalar element (in a kernel) / index a view |
| `T[(tidx, None)]` | slice keeping a sub-tensor; `None` keeps that axis |

---

## 3. Tiling and partitioning (who-owns-what)

This is how a global matrix gets carved into per-CTA tiles, then per-thread slices.

| Call | What it actually does | In your matmul |
|---|---|---|
| `cute.local_tile(T, tiler, coord)` | cut `T` into tiles of shape `tiler`, then **select** by `coord`. An **integer** in coord picks one tile (axis collapses). **`None`** keeps ALL tiles on that axis as an extra loopable dimension. | `gA = local_tile(mA,(128,8),(bidx,None))` -> shape `(128,8,32)`: this CTA's rows, all 32 K-chunks. `gC = local_tile(mC,(128,128),(bidx,bidy))` -> `(128,128)`: one fixed output tile. |
| `tiled_mma.get_slice(tidx)` | resolve the TV layout to **THIS thread's** view (`thr_mma`) | `thr_mma = tiled_mma.get_slice(tidx)` |
| `thr_mma.partition_A(g)` | return this thread's slice of A from a tile | `tCrA = thr_mma.partition_A(gA[None,None,k])` |
| `thr_mma.partition_B(g)` | this thread's slice of B | `tCrB = thr_mma.partition_B(gB[None,None,k])` |
| `thr_mma.partition_C(g)` | this thread's slice of the output C | `tCgC = thr_mma.partition_C(gC)` -> this thread's 8x8 of C |

Rule: **integer coord = pick one tile; `None` = keep all (loop later).** That is
why gA/gB are 3D (kept the K axis) but gC is 2D (both picked).

---

## 4. MMA: the matrix-multiply machinery

| Call | What it actually does | Note |
|---|---|---|
| `cute.nvgpu.MmaUniversalOp(Float32)` | the **SIMT FMA atom**: one thread, one fused multiply-add. No tensor core. | portable, any GPU |
| `cute.nvgpu.warp.MmaF16BF16Op(ab_dtype, acc_dtype, shape_mnk)` | a **tensor-core atom**: one WARP does an MMA in hardware. Real call: `MmaF16BF16Op(Float16, Float32, (16,8,16))`. fp16/bf16 in, fp32 acc. | sm_70+ ; what you use for fast DL |
| `cute.make_tiled_mma(op, atom_layout=...)` | tile the atom across threads/warps. `atom_layout` says how many atoms along M,N,K. For a SIMT op it counts **threads**; for a warp op it counts **warps**. **Default with no atom_layout = 1 atom (i.e. 1 warp for a warp MMA)**. | `make_tiled_mma(op, make_layout((16,16,1),(16,1,0)))` = 256 threads as 16x16 |
| `tiled_mma.make_fragment_C(tCgC)` | allocate the per-thread **accumulator in REGISTERS** shaped like its C slice | `acc = tiled_mma.make_fragment_C(tCgC); acc.fill(0.0)` |
| `cute.gemm(tiled_mma, acc, a, b, acc)` | do `acc += a . b` for this thread's fragments | the K-loop body |

"Atom decides granularity": SIMT atom -> you arrange threads (warps implicit);
warp MMA atom -> you arrange warps (you are managing warps, declaratively).

---

## 5. Fragments and registers

| Term | Meaning |
|---|---|
| **fragment** | a small per-thread tensor that lives in **registers** (fastest storage). Shape is dictated by the MMA atom (e.g. for 16x8x16 fp16 MMA: 8 fp16 per lane for A, 4 for B, 4 fp32 for C). |
| `make_fragment_C(...)` | makes the register accumulator for C (partial sums, usually fp32) |
| `acc.fill(0.0)` | zero the accumulator before the K-loop |
| `make_fragment_A/B` (tensor-core path) | register buffers that hold A/B pieces fed to the tensor core (filled via `ldmatrix` or `autovec_copy`) |
| `cute.make_fragment_like(src)` | make a register tensor shaped EXACTLY like another tensor (typically a partition slice). Used for the `global -> registers -> global` pattern in tiled copy. |

Why registers: the K-loop accumulates into `acc` thousands of times; keeping it
in registers (not memory) is what makes the inner loop fast.

---

## 6. Copy: moving data between memory spaces

### Atoms (one copy instruction)
| Call | What it actually does |
|---|---|
| `cute.nvgpu.CopyUniversalOp()` | a generic copy atom (works any direction; synchronous) |
| `cute.nvgpu.cpasync.CopyG2SOp()` | **async** global->shared copy (Ampere+, your sm_89 has it) |
| `cute.make_copy_atom(op, dtype, num_bits_per_copy=32)` | build a copy atom; `num_bits_per_copy` is the vector width (32 = one fp32, 128 = four fp32 = vectorized load) |

### Tiling the atom across threads (the idiom)
| Call | What it actually does |
|---|---|
| `cute.make_tiled_copy_tv(atom, thr_layout, val_layout)` | spread the atom over threads x values-per-thread. Friendly default; produces a **coalesced** TV layout (consecutive threads -> consecutive addresses). E.g. `make_tiled_copy_tv(atom, make_layout(256), make_layout(4))` = 256 threads, 4 values each, 1024-element tile. |
| `cute.make_cotiled_copy(atom, atom_layout_tv, data_layout)` | advanced: build a tiled copy from an EXPLICIT TV layout. Use when you need per-thread vectorization (each thread owns contiguous elements). |
| `tiled_copy.get_slice(tidx)` | resolve TV layout to THIS thread's view |
| `thr.partition_S(src)` | this thread's slice of the **source** tensor (parallel to `partition_A` for MMA) |
| `thr.partition_D(dst)` | this thread's slice of the **destination** tensor |
| `cute.copy(tiled_copy, src, dst)` | emit the copy from `src` view to `dst` view |
| `cute.autovec_copy(src, dst)` | auto-vectorized copy WITHOUT a tiled_copy object; CuTe picks a sensible width. Used to fill MMA fragments from a partition. |

### The async dance (cp.async)
| Call | What it actually does |
|---|---|
| `cute.arch.cp_async_commit_group()` | bundle the in-flight async copies into a group |
| `cute.arch.cp_async_wait_group(n)` | wait until only `n` groups remain in flight (use 0 to wait for all) |
| `cute.arch.sync_threads()` | always follow `wait_group` with this; `wait_group` only finishes THIS thread's copies |

The full async load skeleton:
```python
cute.copy(tc_async, gSrc, sDst)        # 1) fire (non-blocking)
cute.arch.cp_async_commit_group()       # 2) bundle
cute.arch.cp_async_wait_group(0)        # 3) wait
cute.arch.sync_threads()                # 4) block barrier so all threads see smem
```

GEMM write-back example (registers -> global, sync):
```python
copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
cute.copy(copy_atom, acc, tCgC)
```

### Coalescing vs vectorization (the layout tradeoff)
Real performance rule: **the warp's 32 memory requests should hit a small contiguous span**.
- **Coalescing** (across threads): consecutive threads, consecutive addresses. The default `make_tiled_copy_tv` gives you this with 32-bit copies.
- **Vectorization** (per thread): each thread's elements are contiguous, so a 128-bit copy moves 4 fp32 in one instruction.
- A flat 1D layout gives you ONE of these. To get both, use a 2D TV layout (each thread owns a contiguous chunk; consecutive threads own adjacent chunks). That is what real GEMMs do.

---

## 7. Shared memory

| Call | What it actually does |
|---|---|
| `cutlass.utils.SmemAllocator()` | a per-CTA shared-memory allocator |
| `.allocate_tensor(dtype, layout, byte_alignment=...)` | carve a shared tensor out of SMEM | 
| `cute.arch.sync_threads()` | **block barrier**: all threads wait here. Required between writing shared mem and reading what others wrote. |

Golden rule: **write shared -> `sync_threads()` -> read shared.** Missing this
barrier is the #1 "sometimes wrong" bug.

---

## 8. Thread / warp / arch primitives

| Call | Returns / does |
|---|---|
| `cute.arch.thread_idx()` | `(x,y,z)` thread position in the block. With `block=[256,1,1]`, x = 0..255 |
| `cute.arch.block_idx()` | `(x,y,z)` which CTA in the grid (which output tile you own) |
| `cute.arch.block_dim()` | block size per axis |
| `cute.arch.grid_dim()` | grid size per axis |
| `cute.arch.lane_idx()` | lane within the warp (`tid % 32`) |
| `cute.arch.warp_idx()` | warp index within the block (`tid // 32`) |
| `cute.arch.sync_threads()` | block-wide barrier |
| `cute.arch.sync_warp()` | warp-wide barrier |
| `cute.arch.shuffle_sync_down(v, d)` | each lane gets lane `(self+d)`'s register `v`. Register-to-register, **within one warp only**, no memory. |

Reduction rule of thumb: **shuffle = within a warp (32 lanes); shared memory =
across the whole block.** Big reductions do both (shuffle per warp, then SMEM to
merge per-warp results).

Warp shuffle fold (sum over 32 lanes, result in lane 0):
```python
offset = 16
while offset > 0:
    v = v + cute.arch.shuffle_sync_down(v, offset)   # 5 steps: 16,8,4,2,1
    offset //= 2
```

---

## 9. Control flow and launching

| Call | What it actually does |
|---|---|
| `@cute.jit` | **host** function (CPU). Sets up and launches kernels. |
| `@cute.kernel` | **device** code; running its Python on the CPU BUILDS the GPU program. |
| `kernel(args).launch(grid=[...], block=[...], stream=stream)` | launch the kernel. `grid` = number of CTAs, `block` = threads per CTA. |
| `cutlass.range(n)` | a loop; with a **dynamic** `n` it becomes a real GPU loop, with constexpr it may unroll |
| `cutlass.range_constexpr(n)` | a compile-time loop (always unrolls) |
| plain `while`/`for` over a python int | runs at **build time on CPU**, unrolls into straight-line GPU code |
| `if` on a **dynamic** value (e.g. `tidx < s`) | becomes a real per-thread GPU branch |
| `stream: cuda.CUstream` annotation | the stream arg MUST be typed as a real `cuda.CUstream`, not a raw int, or IR lowering fails |

Grid math (matmul): `grid_m = (M + bM - 1)//bM`, `grid_n = (N + bN - 1)//bN`.
**Number of CTAs = (M/BLOCK_M) * (N/BLOCK_N). K never enters this (it is summed away).**

---

## 10. Gotchas you already hit (keep handy)

- **B must be transposed:** pass B as `from_dlpack(b.T)` for `cute.gemm` (contracts trailing mode of A and B).
- **Stream type:** annotate `stream: cuda.CUstream` in `@cute.jit __call__`.
- **cupy RNG:** the cutlass wheel ships no `libcurand`, so make inputs with NumPy then `cp.asarray`. `cp.zeros`/`asarray`/DLPack all work fine.
- **`num_bits_per_copy`:** `CopyG2SOp` needs it (32 for fp32).
- **two "grids":** launch grid = number of CTAs; thread layout (`make_layout`) = threads inside one CTA. Only link: layout size must equal threads-per-block.
- **coord None vs int** in `local_tile`: int picks one tile, None keeps all (loop later).
- **don't run python from inside the cutlass package dir** (a local `torch.py` shadows real torch -> circular import). Run from a neutral dir.

---

## 11. Host vs device vs build time (the big mental model)

```
BUILD TIME (CPU, once):  run @cute.kernel python  ->  emit the GPU program
                         python ints / loops / ifs steer code generation (unroll, constants)
                         cute ops on tensors record GPU instructions

RUN TIME (GPU, many threads):  the emitted program executes
                               dynamic ifs/loops and cute ops do the real work
```

- "host side" = CPU code (`@cute.jit`) that orchestrates and `.launch()`es.
- "device side" = the GPU program, generated by tracing `@cute.kernel`.
- A `while` over a python int does NOT run on the GPU; it unrolls at build time.

---

## 12. sm_89 (your RTX 4050) capability ceiling

Have: warp shuffles, shared memory, **cp.async**, tensor cores (fp16/bf16/tf32/**fp8**/int8), ldmatrix/stmatrix.
Do NOT have (Hopper/Blackwell): TMA, wgmma (warpgroup MMA), clusters/DSMEM, tcgen05.

`matmul_cute.py` and `softmax_cute.py` use only portable SIMT primitives, so they
run on basically any NVIDIA GPU. sm_89 is a *ceiling*, not a requirement; it only
matters once you reach for cp.async / tensor cores (use them) vs TMA / wgmma (skip).

---

## 13. File map (what's in this repo)

**Runnable kernels:**
- `matmul_cute.py` ............ tiled SIMT GEMM (256^3, 128x128 tiles), verified.
- `softmax_cute.py` ........... stable row-wise softmax, one thread per row.
- `softmax_reduce_cute.py` .... one CTA per row + smem tree reduction (scaffold).
- `layouts_playground.py` ..... CPU-side predict-then-verify of all 8 layout algebra ops.

**Tutorials (one new concept each, all NumPy-verified):**
- `tutorials/__shfl_down_sync.py` ... warp shuffles (registers, within-warp comms).
- `tutorials/__shared_memory.py` .... block-wide smem + sync_threads + tree reduction.
- `tutorials/__copy_atom.py` ........ tiled copy idiom (atom + partition_S/D + fragment), coalescing.
- `tutorials/__cp_async.py` ......... async global->shared, the commit/wait dance.
- `tutorials/__mma_tensorcore.py` ... ONE warp doing ONE tensor-core 16x8x16 fp16/fp32 MMA.

**Docs in `.agents/`:**
- `learning.md` ..... the full ground-up guide (architecture -> CuTe -> learning path).
- `cute_algebra.md` . the 8 layout algebra concepts in depth.
- `kernel_explained.md` line-by-line walkthrough of `matmul_cute.py`.
- `cheatsheet.md` ... this file.

Path to real DL kernels (combine the four primitives you now have):
staged matmul (cp.async + smem) -> tensor-core matmul -> fused softmax -> baby FlashAttention.
```
