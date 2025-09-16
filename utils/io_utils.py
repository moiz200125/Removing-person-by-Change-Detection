import os
import re
import cv2
import re
import numpy as np
import struct, zlib
from PIL import Image 
import matplotlib.pyplot as plt

def read_frames(path: str) -> np.ndarray:
    """
    Reads frames from a folder (using PIL) and converts them to grayscale.

    Args:
        path (str): Path to the folder containing frames.

    Returns:
        np.ndarray: Array of shape (F, H, W) where
                    F = number of frames,
                    H = height,
                    W = width.
    """
    # collect image files
    files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # sort numerically based on digits in filename
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

    frames = []
    for fname in files:
        img_path = os.path.join(path, fname)
        # Open with Pillow and convert to grayscale ("L" mode)
        img = Image.open(img_path).convert("L")
        frames.append(np.array(img, dtype=np.float32))

    frames = np.stack(frames)  # shape (F, H, W)
    return frames



def plot_frames(frames: list, num_frames: int, save_name: str) -> None:
    """
    Plots and saves multiple frames in a single image.

    Args:
        frames (list): List/array of frame arrays (grayscale).
        num_frames (int): Number of frames to display.
        save_name (str): File name for the saved plot.
    """
    n = min(num_frames, len(frames))
    cols = 5
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(15, 3*rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(frames[i], cmap='gray')
        plt.title(f"Frame {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()


def write_png(path: str, arr: np.ndarray) -> None:
    """
    Save an image (grayscale or RGB) as PNG using zlib compression.
    
    Args:
        path (str): Output file path
        arr (np.ndarray): Image array
                         - Grayscale: shape (H, W)
                         - RGB: shape (H, W, 3)
                         dtype must be uint8
    """
    if arr.dtype != np.uint8:
        raise ValueError("Input array must be uint8")

    # Handle grayscale vs RGB
    if arr.ndim == 2:   # grayscale
        color_type = 0
        h, w = arr.shape
        stride = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:  # RGB
        color_type = 2
        h, w, _ = arr.shape
        stride = arr.reshape(h, w*3)
    else:
        raise ValueError("Only grayscale (HxW) or RGB (HxWx3) supported")

    # PNG file signature
    png_sig = b'\x89PNG\r\n\x1a\n'

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag+data) & 0xffffffff))

    # IHDR
    bit_depth = 8
    ihdr = struct.pack(">IIBBBBB", w, h, bit_depth, color_type, 0, 0, 0)

    # IDAT: add filter byte 0 at start of each row
    raw = b''.join(b'\x00' + stride[i].tobytes() for i in range(h))
    idat = zlib.compress(raw)

    # IEND
    iend = b''

    # Assemble PNG
    png = png_sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', iend)

    with open(path, "wb") as f:
        f.write(png)



def save_masks_as_video(mask_folder, output_path, fps=25):
    """
    Collects PNG masks from a folder and saves them as an .mp4 video.

    Args:
        mask_folder (str): Folder with mask_XXXX.png files
        output_path (str): Path to save output video (must end with .mp4)
        fps (int): Frames per second
    """
    # collect and sort mask files
    files = [f for f in os.listdir(mask_folder) if f.endswith(".png")]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # numeric sort

    if not files:
        print(f"[WARN] No PNG files in {mask_folder}")
        return

    # open first image to get size
    first_img = np.array(Image.open(os.path.join(mask_folder, files[0])))
    H, W = first_img.shape[:2]

    # video writer (grayscale converted to 3-channel for mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=True)

    for fname in files:
        img = np.array(Image.open(os.path.join(mask_folder, fname)))

        # ensure 3-channel
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        writer.write(img)

    writer.release()
    print(f"[INFO] Saved video: {output_path}")
