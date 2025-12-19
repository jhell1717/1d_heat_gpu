from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import cuda

from .grid import Grid1D
from .physics import HeatEquationParams
from .bc import DirichletBC
from .solvers.base import StepContext

from .solvers.cpu_numba import CPUNumbaSolver
from .solvers.gpu_numba import GPUNumbaSolver

@dataclass
class SimulationConfig:
    t_final:float
    snapshot_every:int = 50

class HeatSimulation:
    def __init__(self,grid:Grid1D,params:HeatEquationParams,bc:DirichletBC):
        self.grid = grid
        self.params = params
        self.bc = bc
        self.dx = grid.dx
        self.dt = params.dt(self.dx)
        self.r = params.alpha * self.dt / (self.dx * self.dx)
        self.ctx = StepContext(r=self.r)

    def run_cpu(self,T0:np.ndarray,cfg:SimulationConfig) -> tuple[np.ndarray,np.ndarray]:
        solver = CPUNumbaSolver()
        T = T0.copy()
        Tnew = np.empty_like(T)
        self.bc.apply(T)

        nsteps = int(cfg.t_final/self.dt)
        frames = []
        times = []

        for n in range(nsteps):
            solver.step(T,Tnew,self.ctx)
            self.bc.apply(Tnew)
            T, Tnew = Tnew, T

            if n% cfg.snapshot_every == 0:
                frames.append(T.copy())
                times.append(n*self.dt)

        return np.array(times), np.array(frames)
        
