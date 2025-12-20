# heat1d/viz.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    title: str = "",
) -> None:
    """
    Plot a single 2D temperature field.
    T shape: (Ny, Nx)
    """
    plt.figure()
    plt.imshow(
        T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.colorbar(label="T")
    plt.show()

def animate_field_2d(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    frames: np.ndarray,
    title: str = "",
    interval: int = 50,
) -> None:
    """
    Animate a 2D temperature field over time.
    frames shape: (nframes, Ny, Nx)
    """
    fig, ax = plt.subplots()

    im = ax.imshow(
        frames[0],
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        vmin=frames.min(),
        vmax=frames.max(),
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("T")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}  t={times[0]:.4g}")

    def update(k):
        im.set_data(frames[k])
        ax.set_title(f"{title}  t={times[k]:.4g}")
        return (im,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval,
        blit=False,
    )

    plt.show()
