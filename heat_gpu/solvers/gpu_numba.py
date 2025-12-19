from __future__ import annotations
import math
from turtle import right
import numpy as np
from numba import cuda
from .base import SolverBase, StepContext

@cuda.jit
def heat_step_dirichlet_kernel(T,Tnew,r,left_bc,right_bc):
    i = cuda.grid(1)
    n = T.shape[0]

    if i<n:
        if i == 0:
            Tnew[i] = left_bc
        elif i == n-1:
            Tnew[i] = right_bc
        else:
            Tnew[i] = T[i] + r * (T[i-1] - 2.0 * T[i] + T[i+1])

class GPUNumbaSolver(SolverBase):
    def __init__(self,threads_per_block:int=256):
        self.threads_per_block = threads_per_block

    def step_device(self,d_T,d_Tnew,ctx:StepContext,left_bc:float,right_bc:float)->None:
        n = d_T.shape[0]
        blocks = (n+self.threads_per_block -1) // self.threads_per_block
        heat_step_dirichlet_kernel[blocks,self.threads_per_block](d_T,d_Tnew,ctx.r,left_bc,right_bc)

    def step(self,T:np.ndarray,Tnew:np.ndarray,ctx:StepContext)->None:
        raise RuntimeError("Use step_device() with device arrays for GPU solver")


