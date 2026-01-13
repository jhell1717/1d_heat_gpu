from __future__ import annotations
import numpy as np
from numba import njit
from .base import SolverBase, StepContext

@njit(parallel=True,cache=True)
def _step_cpu_numba(T:np.ndarray,Tnew:np.ndarray,rx:float,ry:float)->None:
    Nx, Ny = T.shape
    for j in range(1,Ny-1):
        for i in range(1,Nx-1):
            Tnew[j,i] = (
                T[j,i] 
                + rx * (T[j,i-1] - 2*T[j,i] + T[j,i+1])
                + ry * (T[j-1,i] - 2*T[j,i] + T[j+1,i])
            )

class CPUNumbaSolver(SolverBase):
    def step(self,T:np.ndarray,Tnew:np.ndarray,ctx:StepContext) -> None:
        _step_cpu_numba(T,Tnew,ctx.rx,ctx.ry)
