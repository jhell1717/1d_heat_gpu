from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import cuda

from .grid import Grid1D
from .physics import HeatEquationParams
from .bc import DirichletBC
from .solvers.base import StepContext

from .solvers.cpu_numa