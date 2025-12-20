from __future__ import annotations
from dataclasses import dataclass
import numpy as np

class BoundaryCondition:
    def apply(self,T:np.ndarray) -> None:
        raise NotImplementedError

@dataclass(frozen=True)
class DirichletBC(BoundaryCondition):
    Tbc : float
    
    def apply(self,T:np.ndarray) -> None:
        T[0,:] = self.Tbc # Bottom
        T[-1,:] = self.Tbc # Top

        T[:,0] = self.Tbc # Left
        T[:,-1] = self.Tbc # Right


        