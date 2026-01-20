import time
import argparse
from heat2d.grid import Grid2D
from heat2d.physics import HeatEquationParams
from heat2d.bc import DirichletBC
from heat2d.ic import gaussian_hotspot
from heat2d.simulation import HeatSimulation, SimulationConfig
from heat2d.utils import cuda_available
from heat2d.viz import plot_field_2d, animate_field_2d


def main():
    parser = argparse.ArgumentParser(description="2D Heat Equation Solver")

    # Grid parameters
    parser.add_argument("--L", type=float, default=1.0, help="Domain size (meters)")
    parser.add_argument("--N", type=int, default=100, help="Number of grid points per dimension")

    # Physics parameters
    parser.add_argument("--alpha", type=float, default=0.1, help="Thermal diffusivity")
    parser.add_argument("--cfl", type=float, default=0.45, help="CFL number")

    # Boundary conditions
    parser.add_argument("--Tbc", type=float, default=0.0, help="Dirichlet boundary condition")

    # Simulation
    parser.add_argument("--t_final", type=float, default=0.01, help="Final simulation time")
    parser.add_argument("--snapshot_every", type=int, default=10, help="Snapshot interval (timesteps)")

    # Backend
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu", help="Choose solver backend")

    args = parser.parse_args()

    # Set up grid, params, BC
    grid = Grid2D(Lx=args.L, Ly=args.L, Nx=args.N, Ny=args.N)
    params = HeatEquationParams(alpha=args.alpha, cfl=args.cfl)
    bc = DirichletBC(Tbc=args.Tbc)

    sim = HeatSimulation(grid, params, bc)

    # Initial condition
    x, y = grid.x, grid.y
    T0 = gaussian_hotspot(x, y, base=0.0, amp=1.0, x0=0.5*args.L, y0=0.5*args.L, sigma=0.05*args.L)

    # Simulation config
    cfg = SimulationConfig(t_final=args.t_final, snapshot_every=args.snapshot_every)

    # Run simulation
    start = time.time()
    if args.backend == "cpu":
        t, frames = sim.run_cpu(T0, cfg)
    elif args.backend == "gpu":
        t, frames = sim.run_gpu(T0, cfg)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    end = time.time()

    print(f"{args.backend.upper()} simulation took: {end-start:.4f} seconds")
