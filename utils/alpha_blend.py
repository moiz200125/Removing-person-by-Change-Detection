import numpy as np
import cv2  # only for Gaussian blur (can replace with scipy if needed)

from utils.masking import compute_mask
from utils.morphology import create_kernel, erode, dilate
from utils.connected_components import find_connected_components


def soften_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Convert binary mask (0/255) to float mask [0,1] and soften edges with Gaussian blur.

    Args:
        mask (np.ndarray): binary mask (H, W)
        radius (int): blur kernel size (must be odd)

    Returns:
        np.ndarray: softened mask (H, W) in [0,1]
    """
    mask_f = (mask.astype(np.float32) / 255.0)
    if radius > 0:
        mask_f = cv2.GaussianBlur(mask_f, (radius, radius), 0)
    return np.clip(mask_f, 0.0, 1.0)


def blend_roi(frame: np.ndarray, background: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    Alpha blend the person region in frame with background.

    Args:
        frame (np.ndarray): current frame (H, W, 3) uint8
        background (np.ndarray): background image (H, W, 3) uint8
        mask (np.ndarray): binary mask (H, W), 0/255
        alpha (float): blending factor (0..1)

    Returns:
        np.ndarray: blended frame (H, W, 3) uint8
    """
    # soften mask
    soft_mask = soften_mask(mask, radius=7)  # feather edges
    soft_mask = (alpha * soft_mask).astype(np.float32)  # scale with alpha
    soft_mask3 = np.repeat(soft_mask[:, :, None], 3, axis=2)  # make 3-channel

    frame_f = frame.astype(np.float32)
    bg_f = background.astype(np.float32)

    blended = (1 - soft_mask3) * frame_f + soft_mask3 * bg_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def alpha_blend_sequence(frames: np.ndarray,
                         mean_frame: np.ndarray,
                         t_start: int,
                         alpha_schedule: str = "linear",
                         morph_kernel: int = 5,
                         re_detect: bool = True):
    """
    Apply alpha blending to remove a person across consecutive frames.

    Args:
        frames (np.ndarray): (F, H, W, 3) uint8 video frames
        mean_frame (np.ndarray): background model (H, W, 3) uint8
        t_start (int): index to start processing (after background model frames)
        alpha_schedule (str): "linear" (default) or "ease"
        morph_kernel (int): kernel size for morphology
        re_detect (bool): if True, re-run detection on blended frame

    Returns:
        list of np.ndarray: blended frames
    """
    F = frames.shape[0]
    H, W, _ = frames.shape[1:]
    blended_frames = []

    kernel = create_kernel(morph_kernel)

    for i in range(t_start, F):
        frame = frames[i]

        # detection mask (reuse pipeline: compute mask vs mean+var)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mean_gray = cv2.cvtColor(mean_frame, cv2.COLOR_RGB2GRAY)

        # naive variance (for quick detection): difference
        diff = cv2.absdiff(gray.astype(np.float32), mean_gray.astype(np.float32))
        mask = np.where(diff > 25, 255, 0).astype(np.uint8)

        # morphology
        mask = erode(mask, kernel, iterations=1)
        mask = dilate(mask, kernel, iterations=1)

        # connected components → keep largest
        num_comps, labeled, comp_info = find_connected_components(mask)
        if num_comps > 0:
            largest = max(comp_info, key=lambda c: c["area"])
            mask = np.where(labeled == largest["id"], 255, 0).astype(np.uint8)
        else:
            blended_frames.append(frame)
            continue

        # compute alpha
        if alpha_schedule == "linear":
            alpha = min(1.0, (i - t_start + 1) / (F - t_start))
        else:  # ease-in-out
            x = (i - t_start + 1) / (F - t_start)
            alpha = x * x * (3 - 2 * x)

        # blend
        blended = blend_roi(frame, mean_frame, mask, alpha)
        blended_frames.append(blended)

        # re-detection (optional)
        if re_detect:
            gray_b = cv2.cvtColor(blended, cv2.COLOR_RGB2GRAY)
            diff_b = cv2.absdiff(gray_b.astype(np.float32), mean_gray.astype(np.float32))
            mask_b = np.where(diff_b > 25, 255, 0).astype(np.uint8)
            mask = mask_b  # for next iteration if needed

    return blended_frames
