from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class HeatEquationParams:
    alpha : float
    cfl : float = 0.45

    def dt_2d(self,dx:float,dy:float) -> float:
        return self.cfl / self.alpha / (1/dx**2 + 1.0/dy**2)

    def rx_ry(self,dx:float,dy:float) -> tuple[float,float]:
        dt = self.dt_2d(dx,dy)
        rx = self.alpha * dt / dx**2
        ry = self.alpha * dt / dy**2
        return rx,ry