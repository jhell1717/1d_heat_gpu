from __future__ import annotations
import math
from turtle import right
import numpy as np
from numba import cuda
from .base import SolverBase, StepContext

@cuda.jit
def heat_step_dirichlet_kernel(T,Tnew,rx,ry,Tbc):
    i,j = cuda.grid(2)
    Ny,Nx = T.shape

    if 1 <= i < Ny-1 and 1 <=j < Nx-1:
        Tnew[i,j] = (
            T[i,j]
            + rx * (T[i,j-1] - 2*T[i,j] + T[i,j+1])
            + ry * (T[i-1,j] - 2*T[i,j] + T[i+1,j])
        )
    elif i < Ny and j < Nx:
        Tnew[i,j] = Tbc

class GPUNumbaSolver(SolverBase):
    def __init__(self,threads_per_block:int=(16,16)):
        self.threads_per_block = threads_per_block

    def step_device(self,d_T,d_Tnew,ctx:StepContext,Tbc:float)->None:
        Ny,Nx = d_T.shape
        ty,tx = self.threads_per_block

        # blocks = (n+self.threads_per_block -1) // self.threads_per_block
        blocks = (
            math.ceil(Ny / ty),
            math.ceil(Nx / tx),
            )

        heat_step_dirichlet_kernel[blocks,self.threads_per_block](d_T,d_Tnew,ctx.rx,ctx.ry,Tbc)

    def step(self,T:np.ndarray,Tnew:np.ndarray,ctx:StepContext)->None:
        raise RuntimeError("Use step_device() with device arrays for GPU solver")


