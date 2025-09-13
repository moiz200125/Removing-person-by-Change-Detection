import numpy as np

def compute_mean(frames: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel mean across frames.

    Args:
        frames: np.ndarray of shape (N, H, W) or (N, H, W, C), dtype float32/float64/uint8

    Returns:
        mean_frame: np.ndarray of shape (H, W) or (H, W, C), dtype float32
    """
    # cast to float32 for safe math
    frames_f = frames.astype(np.float32)
    mean_frame = np.mean(frames_f, axis=0)
    return mean_frame.astype(np.float32)


def compute_variance(frames: np.ndarray, mean_frame: np.ndarray = None) -> np.ndarray:
    """
    Compute per-pixel population variance across frames.

    Args:
        frames: np.ndarray of shape (N, H, W) or (N, H, W, C)
        mean_frame: optional precomputed mean of shape (H,W) or (H,W,C)

    Returns:
        var_frame: np.ndarray of same spatial shape as mean_frame, dtype float32
    """
    frames_f = frames.astype(np.float32)
    if mean_frame is None:
        mean_frame = compute_mean(frames_f)
    # broadcasting: frames_f - mean_frame has shape (N,H,W[,C])
    var_frame = np.mean((frames_f - mean_frame) ** 2, axis=0)
    return var_frame.astype(np.float32)


# Welford's algorithm (single-pass, numerically stable) for streaming / memory constrained cases
def compute_mean_variance_welford(frames_iterable):
    """
    Compute mean and population variance using Welford's algorithm in one pass.
    Accepts an iterable that yields frames one by one (numpy arrays of shape HxW or HxWxC).

    Returns:
        mean: np.ndarray same shape as a single frame (float32)
        var: np.ndarray same shape as a single frame (float32)
    """
    count = 0
    mean = None
    M2 = None  # sum of squared differences * count

    for frame in frames_iterable:
        x = frame.astype(np.float64)  # use float64 internally for stability
        if mean is None:
            mean = np.zeros_like(x, dtype=np.float64)
            M2 = np.zeros_like(x, dtype=np.float64)
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2

    if count == 0:
        raise ValueError("No frames provided")

    # population variance = M2 / N
    var = (M2 / count).astype(np.float32)
    mean = mean.astype(np.float32)
    return mean, var
