"""Real TileLang 64-lane vadd-one kernel used by H4.

This is the actual TL source compiled with the installed TileLang Ascend backend;
it is not the generic torch add and is not a hand-written adapter payload.
"""
from __future__ import annotations

import numpy as np
import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T


def vadd_one(n: int = 64):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1) as bx:
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n,), "float32")
            T.copy(A[0:n], a)
            with T.SimtVF(threads=64):
                for i in T.Parallel(n):
                    c[i] = a[i] + 1.0
            T.copy(c, C[0:n])
    return main


def compile_vadd_one(n: int = 64):
    return tilelang.compile(vadd_one(n), target="ascend", out_idx=-1)


def readback_vadd_one(kernel, input_bytes: bytes) -> tuple[bytes, tuple, str]:
    x = torch.tensor(np.frombuffer(input_bytes, dtype=np.float32).copy(), device="npu:0")
    y = kernel(x)
    torch.npu.synchronize()
    return y.cpu().numpy().tobytes(), tuple(y.shape), "float32"
