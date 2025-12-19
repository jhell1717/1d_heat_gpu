# heat1d/viz.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_space_time(x: np.ndarray, times: np.ndarray, frames: np.ndarray, title: str = "") -> None:
    # frames shape: (nframes, N)
    plt.figure()
    plt.imshow(
        frames,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], times[0], times[-1]],
    )
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.colorbar(label="T")
    plt.show()

def animate_line(x: np.ndarray, times: np.ndarray, frames: np.ndarray, title: str = "") -> None:
    fig, ax = plt.subplots()
    line, = ax.plot(x, frames[0])
    ax.set_xlabel("x")
    ax.set_ylabel("T")
    ax.set_title(title)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(frames.min(), frames.max())

    def update(k):
        line.set_ydata(frames[k])
        ax.set_title(f"{title}  t={times[k]:.4g}")
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(times), interval=30, blit=False)
    plt.show()
