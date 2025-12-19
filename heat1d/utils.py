# heat1d/utils.py
from numba import cuda

def cuda_available() -> bool:
    try:
        return cuda.is_available()
    except Exception:
        return False