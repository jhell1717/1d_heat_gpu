from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Grid1D:
    L : float
    N : int

    @property
    def dx(self) -> float:
        return self.L / (self.N - 1)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0,self.L,self.N,dtype=np.float64)