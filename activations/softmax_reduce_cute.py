import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cupy as cp
import numpy as np
import cuda.bindings.driver as cuda

# softmax(x_i) = exp(x_i - max_j x_j) / sum_k exp(x_k - max_j x_j)
#
# Design: ONE CTA per row. The THREADS threads in the CTA cooperate.
#   - each thread handles a strided set of columns: j = tidx, tidx+THREADS, ...
#   - phase 1: each thread finds its local max, then a tree-reduction in
#              shared memory turns THREADS local maxes into one row max.
#   - phase 2: same shape of code, but summing exp(x - row_max).
#   - phase 3: each thread writes exp(x - row_max) / row_sum for its columns.
#
# N is chosen divisible by THREADS so every thread handles exactly N/THREADS
# columns and you do not need a bounds guard.

M, N = 1024, 1024
THREADS = 256


class SoftmaxReduce:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx

        # shared-memory scratch: one float per thread (reused for max then sum)
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(THREADS),
                                   byte_alignment=4)

        # ---------------------------------------------------------------
        # PHASE 1: row max
        # TODO 1a: compute this thread's local max over its strided columns.
        #          loop j = tidx, tidx+THREADS, ... < N  and track the largest
        #          mX[row, j]. Start local_max from mX[row, tidx].
        #
        # TODO 1b: red[tidx] = local_max ; cute.arch.sync_threads()
        #
        # TODO 1c: tree reduction. with s = THREADS//2 down to 1 (s //= 2):
        #            if tidx < s: red[tidx] = max(red[tidx], red[tidx + s])
        #            cute.arch.sync_threads() after each step
        #
        # TODO 1d: row_max = red[0] ; then cute.arch.sync_threads()
        #          (sync so no thread overwrites red before everyone read it)
        # ---------------------------------------------------------------

        # ---------------------------------------------------------------
        # PHASE 2: row sum of exp(x - row_max)
        # TODO 2a: local_sum = 0.0; loop the same strided columns adding
        #          cute.exp(mX[row, j] - row_max)
        # TODO 2b: red[tidx] = local_sum ; sync
        # TODO 2c: tree reduction again, but with += instead of max
        # TODO 2d: row_sum = red[0] ; sync
        # ---------------------------------------------------------------

        # ---------------------------------------------------------------
        # PHASE 3: write the answer
        # TODO 3: loop the same strided columns and store
        #         mO[row, j] = cute.exp(mX[row, j] - row_max) / row_sum
        # ---------------------------------------------------------------
        pass

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mX, mO).launch(
            grid=[M, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


def main():
    print(f"Row-wise softmax (reduction) over a {M}x{N} matrix...")

    x_np = np.random.randn(M, N).astype(np.float32)
    x = cp.asarray(x_np)
    o = cp.zeros((M, N), dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    SoftmaxReduce()(from_dlpack(x), from_dlpack(o), stream)
    cp.cuda.get_current_stream().synchronize()

    ref = np.exp(x_np - x_np.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(o.get(), ref, atol=1e-5, rtol=1e-4)
    print("SUCCESS! reduction softmax matched NumPy result.")


if __name__ == "__main__":
    main()
