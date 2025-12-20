from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Grid1D:
    Lx : float
    Ly : float
    Nx : int
    Ny : int

    @property
    def dx(self) -> float:
        return self.Lx / (self.Nx - 1)

    @property
    def dy(self) -> float:
        return self.Ly / (self.Ny - 1)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0,self.Lx,self.Nx,dtype=np.float64)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(0.0,self.Ly,self.Ny,dtype=np.float64)

