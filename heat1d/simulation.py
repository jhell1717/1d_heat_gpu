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

    def run_gpu(self, T0: np.ndarray, cfg: SimulationConfig, threads_per_block: int = 256) -> tuple[np.ndarray, np.ndarray]:
        solver = GPUNumbaSolver(threads_per_block=threads_per_block)

        # host -> device
        h_T = T0.copy()
        h_T[0] = self.bc.left
        h_T[-1] = self.bc.right

        d_T = cuda.to_device(h_T)
        d_Tnew = cuda.device_array_like(d_T)

        nsteps = int(cfg.t_final / self.dt)
        frames = []
        times = []

        for n in range(nsteps):
            solver.step_device(d_T, d_Tnew, self.ctx, self.bc.left, self.bc.right)
            d_T, d_Tnew = d_Tnew, d_T  # swap

            if (n % cfg.snapshot_every) == 0:
                frames.append(d_T.copy_to_host())
                times.append(n * self.dt)

        return np.array(times), np.array(frames)
        
