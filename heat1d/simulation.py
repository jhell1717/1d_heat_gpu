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
        self.dy = grid.dy
        self.dt = params.dt_2d(self.dx, self.dy)
        self.rx,self.ry = params.rx_ry(self.dx,self.dy)
        self.ctx = StepContext(rx=self.rx,ry=self.ry)

    def run_cpu(self,T0:np.ndarray,cfg:SimulationConfig) -> tuple[np.ndarray,np.ndarray]:
        
        # Setup CPU instance, using Numba njit compiler.
        solver = CPUNumbaSolver()

        # Create array same shape as initial conditions, T.
        T = T0.copy()

        # Create empty array like T for which new values will go into.
        Tnew = np.empty_like(T)

        # Apply boundary conditions to T[0] & T[-1] (i.e., ends).
        self.bc.apply(T)

        # compute number of time steps in simulation.
        nsteps = int(cfg.t_final/self.dt)

        # initialise empty arrays for times & frames. Frames = snapshots of T.
        frames = []
        times = []

        # Steps through the simulation in time for nsteps.
        for n in range(nsteps):

            # Take single time step with CPU solver.
            solver.step(T,Tnew,self.ctx)

            # Step updates boundaries, so BCs are applied to new spatial solution (along x)
            self.bc.apply(Tnew)

            # The new T along x becomes the current step
            T, Tnew = Tnew, T

            # Append solution time and values at snapshot interval.
            if n% cfg.snapshot_every == 0:
                frames.append(T.copy())
                times.append(n*self.dt)

        return np.array(times), np.array(frames)

    def run_gpu(self, T0: np.ndarray, cfg: SimulationConfig, threads_per_block: int = 256) -> tuple[np.ndarray, np.ndarray]:
        # Setup GPU instance of solver. Uses CUDA.jit - specifying threads per block.
        solver = GPUNumbaSolver(threads_per_block=threads_per_block)

        # host -> device
        # Create array like T0 on the host.
        h_T = T0.copy()

        # Manual specification of left & right boundary condition 
        # TODO: Build boundary conditions into the solver to avoid assignment here.
        h_T[0] = self.bc.left
        h_T[-1] = self.bc.right

        # Move host T array to device
        d_T = cuda.to_device(h_T)

        # Create a new target array on device like initial T.
        d_Tnew = cuda.device_array_like(d_T)

        # Compute number of time steps.
        nsteps = int(cfg.t_final / self.dt)
        n_snapshots = (nsteps + cfg.snapshot_every - 1) // cfg.snapshot_every

        d_frames = cuda.device_array(
            (n_snapshots, d_T.shape[0]),
            dtype=d_T.dtype
        )

        # Initialise empty arrays for T frames & times steps.
        frames = []
        times = []

        # Iterate over number of time steps.
        snap_idx = 0

        for n in range(nsteps):
            # Take step using GPU solver.
            # Computes number of blocks per grid using shape of T array. 
            solver.step_device(d_T, d_Tnew, self.ctx, self.bc.left, self.bc.right)
            d_T, d_Tnew = d_Tnew, d_T  # swap

            # # Save times and T solution if at timestep interval.
            # if (n % cfg.snapshot_every) == 0:
            #     frames.append(d_T.copy_to_host())
            #     times.append(n * self.dt)

            if n % cfg.snapshot_every == 0:
                    d_frames[snap_idx, :] = d_T
                    snap_idx += 1
        frames = d_frames.copy_to_host()
        times = np.arange(n_snapshots) * cfg.snapshot_every * self.dt
        # return times, frames
        return np.array(times), np.array(frames)
        
