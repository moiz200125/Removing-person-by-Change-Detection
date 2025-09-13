import numpy as np

def create_kernel(kernel_size: int = 3) -> np.ndarray:
    """
    Create a square kernel (structuring element) of ones.

    Args:
        kernel_size (int): Size of the kernel (odd number).

    Returns:
        np.ndarray: Kernel of shape (k, k).
    """
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)


def erode(mask: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Perform binary erosion.

    Args:
        mask (np.ndarray): Binary mask (0/255).
        kernel (np.ndarray): Structuring element.
        iterations (int): How many times to apply erosion.

    Returns:
        np.ndarray: Eroded mask.
    """
    k = kernel.shape[0]
    pad = k // 2
    eroded = mask.copy()

    for _ in range(iterations):
        out = np.zeros_like(eroded)
        for i in range(pad, eroded.shape[0] - pad):
            for j in range(pad, eroded.shape[1] - pad):
                region = eroded[i-pad:i+pad+1, j-pad:j+pad+1]
                if np.all(region[kernel == 1] == 255):
                    out[i, j] = 255
        eroded = out
    return eroded


def dilate(mask: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Perform binary dilation.

    Args:
        mask (np.ndarray): Binary mask (0/255).
        kernel (np.ndarray): Structuring element.
        iterations (int): How many times to apply dilation.

    Returns:
        np.ndarray: Dilated mask.
    """
    k = kernel.shape[0]
    pad = k // 2
    dilated = mask.copy()

    for _ in range(iterations):
        out = np.zeros_like(dilated)
        for i in range(pad, dilated.shape[0] - pad):
            for j in range(pad, dilated.shape[1] - pad):
                region = dilated[i-pad:i+pad+1, j-pad:j+pad+1]
                if np.any(region[kernel == 1] == 255):
                    out[i, j] = 255
        dilated = out
    return dilated
