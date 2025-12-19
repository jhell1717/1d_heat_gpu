from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class HeatEquationParams:
    alpha : float
    r : float = 0.45

    def dt(self,dx:float) -> float:
        return self.r * dx * dx / self.alpha