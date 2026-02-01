**Run GPU Simulation in Google Colab**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/jhell1717/1d_heat_gpu/blob/dimensionality%2F2D/run_on_colab.ipynb
)


### Observations:
* For 2D, the time difference between GPU and CPU becomes more noticeble due to higher GPU occupancy.

1d_heat_gpu/
├── .cursor/
│   └── debug.log
├── .gitignore
├── Dockerfile
├── examples/
│   └── run_on_colab.ipynb
├── heat2d/
│   ├── __init__.py
│   ├── bc.py
│   ├── cli.py
│   ├── grid.py
│   ├── ic.py
│   ├── physics.py
│   ├── simulation.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── cpu_numba.py
│   │   └── gpu_numba.py
│   ├── utils.py
│   └── viz.py
├── pyproject.toml
├── README.md
└── run_demo.py
