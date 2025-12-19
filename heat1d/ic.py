import numpy as np

def gaussian_hotspot(x:np.ndarray,base:float,amp:float,x0:float,sigma:float) -> np.ndarray:
    return base + amp * np.exp(-0.5 * ((x - x0) / sigma) **2)