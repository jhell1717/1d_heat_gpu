from __future__ import annotations
from dataclasses import dataclass
import numpy as np

class BoundaryCondition:
    def apply(self,T:np.ndarray) -> None:
        raise NotImplementedError

@dataclass(frozen=True)
class DirichletBC(BoundaryCondition):
    left : float
    right : float

    def apply(self,T:np.ndarray) -> None:
        T[0] = self.left
        T[-1] = self.right