**Run GPU Simulation in Google Colab**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/jhell1717/1d_heat_gpu/blob/main/run_on_colab.ipynb
)


### Observations:
* For the 1D explicit heat equation, the CPU implementation outperforms the GPU at moderate grid sizes due to kernel launch overhead and the memory-bandwidth-bound nature of the stencil. GPU acceleration becomes beneficial only at sufficiently large spatial resolutions.