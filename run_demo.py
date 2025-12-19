# run_demo.py
import numpy as np

from heat1d.grid import Grid1D
from heat1d.physics import HeatEquationParams
from heat1d.bc import DirichletBC
from heat1d.ic import gaussian_hotspot
from heat1d.simulation import HeatSimulation, SimulationConfig
from heat1d.viz import plot_space_time, animate_line

from heat1d.utils import cuda_available



def main():
    # Problem setup
    L = 1.0
    N = 20000  # increase to 100k+ for clearer GPU advantage
    alpha = 1e-3

    grid = Grid1D(L=L, N=N)
    params = HeatEquationParams(alpha=alpha, r=0.45)
    bc = DirichletBC(left=0.0, right=0.0)

    sim = HeatSimulation(grid, params, bc)

    

    # Initial condition: Gaussian hot spot
    x = grid.x
    T0 = gaussian_hotspot(x, base=0.0, amp=1.0, x0=0.5 * L, sigma=0.05 * L)

    cfg = SimulationConfig(t_final=10, snapshot_every=1000)

    # CPU run
    t_cpu, frames_cpu = sim.run_cpu(T0, cfg)

    # GPU run
    # t_gpu, frames_gpu = sim.run_gpu(T0, cfg, threads_per_block=256)
    if cuda_available():
        t_gpu, frames_gpu = sim.run_gpu(T0, cfg)
        # Sanity check: compare frames (same snapshot schedule)
        if len(t_cpu) == len(t_gpu):
            err = np.max(np.abs(frames_cpu - frames_gpu))
            print(f"Max |CPU-GPU| over snapshots: {err:.3e}")

        plot_space_time(x, t_gpu, frames_gpu, title="GPU: 1D Heat Equation (space-time)")
        # animate_line(x, t_gpu, frames_gpu, title="GPU: 1D Heat Equation")


    else:
        print("CUDA not available; skipping GPU run.")

    # Visualize
    plot_space_time(x, t_cpu, frames_cpu, title="CPU: 1D Heat Equation (space-time)")


    # Optional animation (use fewer frames or smaller N if slow)
    animate_line(x, t_cpu, frames_cpu, title="CPU: 1D Heat Equation")

if __name__ == "__main__":
    main()
