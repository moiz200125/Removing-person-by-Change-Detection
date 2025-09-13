import numpy as np

def compute_mask(frame: np.ndarray,
                 mean_frame: np.ndarray,
                 variance_frame: np.ndarray,
                 threshold: float = 5.0,
                 squared: bool = False,
                 min_std: float | None = None,
                 eps: float = 1e-6) -> np.ndarray:
    """
    Compute a binary motion mask using a Mahalanobis-style per-pixel distance.

    Args:
        frame: Current frame, shape (H, W) or (H, W, C). dtype uint8 or float32.
        mean_frame: Background mean, same shape as frame (H,W) or (H,W,C).
        variance_frame: Background variance, same spatial shape; if grayscale shape (H,W) and frame RGB,
                        it will be broadcast across channels.
        threshold: Distance threshold in std units (default 5.0).
                   If squared=True, this threshold is compared to squared Mahalanobis values.
        squared: If True, compute (I-μ)^2 / (σ^2 + eps) and compare to threshold.
                 If False (default), compute |I-μ| / (sqrt(σ^2)+eps) and compare to threshold.
        min_std: Optional. If given, std is clamped to at least min_std (useful to avoid explosion).
                 Example: min_std = 1.0 on 0-255 images.
        eps: small constant to prevent division by zero.

    Returns:
        mask: uint8 binary mask, shape (H, W), values 0 or 255.
    """
  

    # Convert inputs to float32
    I = frame.astype(np.float32)
    mu = mean_frame.astype(np.float32)
    var = variance_frame.astype(np.float32)

    # If var is scalar or single-channel and I is multi-channel, broadcast
    if var.ndim == 2 and I.ndim == 3:
        var = np.repeat(var[:, :, np.newaxis], I.shape[2], axis=2)
    if mu.ndim == 2 and I.ndim == 3:
        mu = np.repeat(mu[:, :, np.newaxis], I.shape[2], axis=2)

    # Compute std (sqrt of variance)
    if squared:
        # Use squared Mahalanobis: (I-mu)^2 / (var+eps)
        denom = var + eps
        # Optionally clamp small denominator by using min_std**2
        if min_std is not None:
            denom = np.maximum(denom, (min_std ** 2))
        M = ((I - mu) ** 2) / denom  # shape (H,W[,C])
        # For multi-channel, combine channels: use max to be sensitive if any channel deviates
        if M.ndim == 3:
            M_comb = np.max(M, axis=2)
        else:
            M_comb = M
        mask_bool = M_comb > threshold
    else:
        # Standardized distance: |I-mu| / (sqrt(var)+eps)
        std = np.sqrt(var) + eps
        if min_std is not None:
            std = np.maximum(std, float(min_std))
        M = np.abs(I - mu) / std  # shape (H,W[,C])
        if M.ndim == 3:
            M_comb = np.max(M, axis=2)
        else:
            M_comb = M
        mask_bool = M_comb > threshold

    # Convert boolean mask to uint8 0/255
    mask = (mask_bool.astype(np.uint8)) * 255
    return mask
