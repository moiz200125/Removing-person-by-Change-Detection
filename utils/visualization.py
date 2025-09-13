import os
import matplotlib.pyplot as plt
import numpy as np
from utils.morphology import create_kernel, erode, dilate

def compare_morphological_ops(mask: np.ndarray, kernel_sizes=[5,7,11], save_path="morph_compare.png"):
    """
    Show original mask vs morphological operations with different kernels.

    Args:
        mask (np.ndarray): Binary mask (0/255).
        kernel_sizes (list): List of kernel sizes to test.
        save_path (str): Where to save the side-by-side plot.
    """
    fig, axes = plt.subplots(1, len(kernel_sizes)+1, figsize=(4*(len(kernel_sizes)+1), 4))

    # Original
    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for idx, k in enumerate(kernel_sizes):
        kernel = create_kernel(k)

        # Opening = erosion followed by dilation
        eroded = erode(mask, kernel, iterations=1)
        opened = dilate(eroded, kernel, iterations=1)

        axes[idx+1].imshow(opened, cmap="gray")
        axes[idx+1].set_title(f"Opening {k}x{k}")
        axes[idx+1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Morphological comparison saved to {save_path}")
