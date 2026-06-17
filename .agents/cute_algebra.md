# Bridging Math and Code: CuTe Layout Algebra in CuTeDSL

In high performance GPU programming, managing how data is partitioned across threads, registers, and memory hierarchies is the primary challenge. CuTe is a layout centric library within NVIDIA CUTLASS designed to solve this by representing data layouts as mathematical functions. With CuTeDSL, NVIDIA provides a Python Domain Specific Language (DSL) that exposes these abstractions directly in Python.

This guide details the eight foundational concepts of CuTe layout algebra. It explains their definitions, provides intuitive analogies, and shows how they translate into Python code.

---

## 1. Layout = Coordinate -> Offset

### Definition
A layout is a function that maps a logical coordinate to a physical memory offset.

```text
Coordinate
    ↓
  Layout
    ↓
  Offset
```

### Explanation
In traditional CUDA kernels, developers write manual index calculations like `offset = row * stride + col`. In CuTe, this mapping is encapsulated inside a first class `Layout` object. The function first maps a 1D index to a multidimensional coordinate via a coordinate isomorphism, multiplies each coordinate by its stride, and sums the results.

### Example
Suppose we have the following layout:
* **Shape:** `(4, 8)`
* **Stride:** `(8, 1)`

The layout function is defined as:
$$L(i, j) = i \cdot 8 + j$$

If we pass the coordinate `(2, 3)`, the offset calculation is:
$$L(2, 3) = 2 \cdot 8 + 3 = 19$$

This means that the logical matrix coordinate `(2, 3)` maps directly to physical memory index `19`.

### CuTeDSL Code
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def layout_function_example():
    S = (2, 4)
    D = (2, 2)
    L = cute.make_layout(shape=S, stride=D)

    for i in cutlass.range_constexpr(cute.size(S)):
        # Evaluates the layout at 1D index i
        cute.printf("fL(%d) = %d\n", i, L(i))
```

---

## 2. Shape Defines Coordinates

### Definition
Shape tells us which logical coordinates are valid. It defines the coordinate space.

### Explanation
The shape represents the logical domain of the layout. It does not dictate where data resides in memory, nor does it know anything about strides or physical offsets. It only dictates coordinate ranges.

### Example
Consider a shape defined as:
* `shape = (4, 8)`

Valid coordinates:
* `(0, 0)`
* `(0, 1)`
* ...
* `(3, 7)`

Invalid coordinates:
* `(4, 0)` (out of bounds for rows)
* `(0, 8)` (out of bounds for columns)

This shape simply states that we have 4 rows and 8 columns of logical coordinates. It does not specify anything about memory layout.

### CuTeDSL Code
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def shape_example():
    # A 2D logical space
    S = (4, 8)
    layout_size = cute.size(S)
    cute.printf("Coordinate space size: %d\n", layout_size) # Output: 32
```

---

## 3. Stride Defines Memory Movement

### Definition
Stride tells us how much the physical memory offset changes when a logical coordinate changes. Stride is how far you jump in memory.

### Explanation
Strides control the memory layout, such as whether a layout is column-major, row-major, or custom (like swizzled layouts designed to prevent shared memory bank conflicts).

### Example
Suppose we have:
* **Shape:** `(4, 8)`
* **Stride:** `(8, 1)`

If we move coordinates:
* Moving `(0, 0)` -> `(1, 0)` increases the row index by 1, which changes the offset by `+8` (the row stride).
* Moving `(0, 0)` -> `(0, 1)` increases the column index by 1, which changes the offset by `+1` (the column stride).

Formula:
$$\text{offset} = \text{row} \cdot 8 + \text{col} \cdot 1$$

### CuTeDSL Code
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def stride_example():
    S = (4, 8)
    
    # Column-major layout (stride along rows is 1, stride along cols is 4)
    L_col = cute.make_layout(S, stride=(1, 4))
    
    # Row-major layout (stride along rows is 8, stride along cols is 1)
    L_row = cute.make_layout(S, stride=(8, 1))
```

---

## 4. Hierarchical Shapes Represent Tiles

### Definition
Instead of maintaining a single coordinate system, we can nest shapes and strides to create multiple levels of coordinates.

### Explanation
By nesting shapes and strides, you can reason about different hierarchies of data (like block-level tiles, warp-level tiles, and thread-level fragments) within the same layout object, avoiding manual coordinate translation boilerplate.

### Example
Suppose we have a matrix of size 128x128 and we want to process it in tiles of size 8x8. We can represent this with a hierarchical shape:
* **Hierarchical Shape:** `((16, 8), (16, 8))`

Interpretation:
* `16` tile rows, with `8` rows inside each tile.
* `16` tile columns, with `8` columns inside each tile.

Coordinate format:
`(tile_r, tile_c, inner_r, inner_c)`

For example, coordinate `(2, 3, 4, 5)` means:
* We are at Tile row 2, Tile column 3.
* Inside that tile, we are at row 4, column 5.

This maps to the global coordinate:
$$\text{row} = 2 \cdot 8 + 4 = 20$$
$$\text{col} = 3 \cdot 8 + 5 = 29$$

This hierarchical coordinate system is how CuTe models matrix tiles for GEMMs.

### CuTeDSL Code
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def hierarchical_example():
    # Nested tuple structure representing hierarchical tiling
    nested_shape = ((16, 8), (16, 8))
    nested_stride = ((8, 1), (1024, 128))
    L = cute.make_layout(nested_shape, nested_stride)
```

---

## 5. Composition Combines Mappings

### Definition
Composition takes two coordinate mappings and chains them together.

### Explanation
Ignore the formal definition for a second. Think of composition as plugging one function into another. You already know how to map a Coordinate -> Offset. Suppose we have a matrix tiled into 8x8 blocks.

Without composition, we have a two-step process:
1. **Function A:** Converts tile coordinates into global coordinates:
   $$(tile\_r, tile\_c, inner\_r, inner\_c) \to (global\_r, global\_c)$$
   For coordinate `(2, 3, 4, 5)`, this yields `global_r = 2*8 + 4 = 20` and `global_c = 3*8 + 5 = 29`, giving `(20, 29)`.
2. **Function B:** Converts global coordinates into memory offsets:
   $$(global\_r, global\_c) \to \text{offset}$$
   Formula for row-major matrix: `offset = global_r * 128 + global_c`.
   For `(20, 29)`, we get `20 * 128 + 29 = 2589`.

Composition combines these two mappings so that you plug the output of Function A directly into Function B. The middle coordinate is removed:
$$(tile\_r, tile\_c, inner\_r, inner\_c) \to 2589 \text{ directly}$$

```text
Without Composition:
Tile Coord -> Matrix Coord -> Offset  (Two steps)

With Composition:
Tile Coord -> Offset                  (One step directly)
```

### Real-World Analogy
Suppose you want to fly from a city to a gate at a destination airport:
* **Step 1:** City -> Airport (e.g. Bhubaneswar -> BPI)
* **Step 2:** Airport -> Gate (e.g. BPI -> Gate 12)

Composition combines these flights into a single route mapping:
* **Composed:** City -> Gate (Bhubaneswar -> Gate 12)

The middle airport step is hidden. In CuTe, composition is everywhere. We can compose:
`Thread Coordinate -> Tile Coordinate -> Global Coordinate -> Memory Offset`
into:
`Thread Coordinate -> Memory Offset`
which tells a thread exactly where to read or write in memory in a single step.

### CuTeDSL Code
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def composition_example():
    L1 = cute.make_layout((8, 8), stride=(1, 8))
    L2 = cute.make_layout((2, 2), stride=(4, 1))
    
    # Composed layout mapping L2 coordinates directly to L1 offsets
    L_composed = cute.composition(L1, L2)
```

---

## 6. Divide Creates Tiles

### Definition
Logical divide splits a coordinate space into a nested coordinate space containing a Tile coordinate and a Local coordinate:
$$\text{Coordinate} \to (\text{Tile Coordinate}, \text{Local Coordinate})$$

### Explanation
Logical divide partitions a layout using a tiler. This is the math behind dividing a large global matrix into local blocks, or partitioning a block's work across a grid of warps and threads.

### Example
Suppose we have:
* **Matrix:** 128x128
* **Tile:** 8x8
* **Coordinate:** `(20, 29)`

Applying logical divide:
$$\text{tile\_r} = 20 \mathbin{//} 8 = 2$$
$$\text{tile\_c} = 29 \mathbin{//} 8 = 3$$
$$\text{inner\_r} = 20 \mathbin{\%} 8 = 4$$
$$\text{inner\_c} = 29 \mathbin{\%} 8 = 5$$

Result coordinate:
`(2, 3, 4, 5)`

This tells us the element is at Tile `(2, 3)`, and at position `(4, 5)` inside that tile. This division algebra is the math that powers helper functions like `cute.local_tile()`.

### CuTeDSL Code
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def divide_example():
    L = cute.make_layout((16), stride=(1))
    tiler = (4)
    
    # Partition L into tiles of size 4
    L_divided = cute.logical_divide(L, tiler)
```

---

## 7. Coalesce Simplifies Layouts

### Definition
Coalescing simplifies a layout's structure by merging adjacent, contiguous modes and removing redundant modes of size 1.

### Explanation
Coalescing does not change the physical mapping of coordinates to offsets, but it simplifies the layout representation. This is essential for memory performance, as it collapses multi-dimensional loops into flat, contiguous blocks that compile into vectorized memory instructions.

### Example
Consider the layout:
* **Shape:** `((4, 2), 8)`

The first dimension represents a nested coordinate space of size $4 \times 2 = 8$. Since these dimensions are contiguous, we can collapse the hierarchy:
* **Coalesced Shape:** `(8, 8)`

This simplifies the coordinate space while mapping to the exact same memory layout.

### CuTeDSL Code
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def coalesce_example():
    S = (2, 1)
    D = (3, 1)
    L = cute.make_layout(shape=S, stride=D)
    
    # Coalescing removes the redundant size 1 dimension
    cL = cute.coalesce(L) # Output layout will be 2:3
    cute.printf("L = %s, cL = %s\n", L, cL)
```

---

## 8. Complement Finds Remaining Dimensions

### Definition
Given a layout that covers a portion of a tensor, the complement operation calculates the layout representing the remaining uncovered offsets.

### Explanation
When partitioning data (such as assigning elements to threads), we need to know what part of the memory space is left over. The complement layout represents the strides and shapes needed to sweep through the rest of the memory space. Concatenating a layout with its complement yields a full bijection over a target size $K$.

### Example
Suppose we have:
* **Tensor:** 128 elements
* **Thread layout size:** 32 threads

The thread layout covers 32 elements. To partition all 128 elements, what is missing?
$$128 \mathbin{/} 32 = 4$$

The complement layout has a size of 4. When we combine the thread layout and its complement, each thread gets a mapping of shape `(32, 4)`. A thread indexes into this combined layout using `(thread_idx, value_idx)` to cover the full 128 elements.

### CuTeDSL Code
```python
import cutlass
import cutlass.cute as cute

@cute.jit
def complement_example():
    S = (2, 4)
    D = (1, 2)
    L = cute.make_layout(shape=S, stride=D)
    K = 16
    
    # Find complement layout for L under size K
    cL = cute.complement(L, K) # Returns layout 2:8
    cute.printf("L = %s, cL = %s\n", L, cL)
```

---

## The Big Picture

A Matrix Multiplication (GEMM) kernel in CuTe can be understood as a series of coordinate space transformations:

```text
Global Matrix
      ↓ divide
    Tiles
      ↓ composition
Memory Layout
      ↓ complement
Thread Ownership
      ↓ composition
Warp Ownership
      ↓ composition
Tensor Core Fragment
      ↓
    Offset
```

Everything in CuTe is built on top of transforming coordinate spaces. This is why CuTe is called a layout algebra rather than a standard tensor library.
