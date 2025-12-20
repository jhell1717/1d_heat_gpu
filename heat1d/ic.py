import numpy as np

def gaussian_hotspot(x:np.ndarray,y:np.ndarray,base:float,amp:float,x0:float,y0:float,sigma:float) -> np.ndarray:
    X,Y = np.meshgrid(x,y,indexing='xy')

    return base + amp * np.exp(
        -0.5 * ((x - x0)**2 + (y-y0)**2) / sigma**2
        )