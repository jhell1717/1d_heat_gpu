from __future__ import annotations
import numpy as np
from numba import njit
from .base import SolverBase, StepContext

@njit(cache=True)
def _step_cpu_numba(T:np.ndarray,Tnew:np.ndarray,r:float)->None:
    n = T.shape[0]
    for i in range(1,n-1):
        Tnew[i] = T[i] + r* (T[i-1] - 2.0 * T[i] + T[i+1])

class CPUNumbaSolver(SolverBase):
    def step(self,T:np.ndarray,Tnew:np.ndarray,ctx:StepContext) -> None:
        _step_cpu_numba(T,Tnew,ctx.r)
