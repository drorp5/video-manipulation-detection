import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('src')

def plot_rgb(rgb_img: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(rgb_img)
    plt.tight_layout()
    plt.show()