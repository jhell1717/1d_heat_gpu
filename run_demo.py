# run_demo.py
import numpy as np
import time

from heat1d.grid import Grid2D
from heat1d.physics import HeatEquationParams
from heat1d.bc import DirichletBC
from heat1d.ic import gaussian_hotspot
from heat1d.simulation import HeatSimulation, SimulationConfig
from heat1d.viz import plot_field_2d, animate_field_2d

from heat1d.utils import cuda_available

def main():
    # Problem setup
    L = 1.0
    N = 1000  # increase to 100k+ for clearer GPU advantage
    alpha = 1e-3

    grid = Grid2D(Lx=L,Ly=L,Nx=N,Ny=N)
    params = HeatEquationParams(alpha=alpha, cfl=0.45,)
    bc = DirichletBC(Tbc=0.0)

    sim = HeatSimulation(grid, params, bc)

    # Initial condition: Gaussian hot spot
    x = grid.x
    y = grid.y
    T0 = gaussian_hotspot(x,y, base=0.0, amp=1.0, x0=0.5 * L,y0=0.5*L, sigma=0.05 * L)
    print(T0.shape)

    cfg = SimulationConfig(t_final=10, snapshot_every=1000)

    print("Starting simulation")
    # CPU run
    start = time.time()
    t_cpu, frames_cpu = sim.run_cpu(T0, cfg)
    end = time.time()

    print(f'CPU took: {end-start} seconds')
    # GPU run
    # t_gpu, frames_gpu = sim.run_gpu(T0, cfg, threads_per_block=256)
    if cuda_available():
        t_gpu, frames_gpu = sim.run_gpu(T0, cfg)
        # Sanity check: compare frames (same snapshot schedule)
        if len(t_cpu) == len(t_gpu):
            err = np.max(np.abs(frames_cpu - frames_gpu))
            print(f"Max |CPU-GPU| over snapshots: {err:.3e}")

    else:
        print("CUDA not available; skipping GPU run.")

    # plot_field_2d(grid.x, grid.y, frames_cpu[0], title="Initial condition")
    # plot_field_2d(grid.x, grid.y, frames_cpu[-1], title="Final state")

    animate_field_2d(grid.x, grid.y, t_cpu, frames_cpu, title="2D Heat Diffusion")


if __name__ == "__main__":
    main()
