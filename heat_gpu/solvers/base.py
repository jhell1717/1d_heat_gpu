from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class StepContext:
    r: float

class SolverBase:
    def step(self,T:np.ndarray,Tnew:np.ndarray,ctx: StepContext) -> None:
        raise NotImplementedError