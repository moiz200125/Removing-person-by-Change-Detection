import os
import re
import numpy as np
from PIL import Image   # Pillow

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


import matplotlib.pyplot as plt

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


import struct, zlib

def write_png_gray(path: str, arr: np.ndarray) -> None:
    """
    Write a grayscale image to PNG using zlib compression.
    Args:
        path (str): output file path
        arr (np.ndarray): grayscale image (H,W) dtype uint8
    """
    h, w = arr.shape
    arr = arr.astype(np.uint8)

    # PNG file signature
    png_sig = b'\x89PNG\r\n\x1a\n'

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag+data) & 0xffffffff)

    # IHDR
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit depth, color type 0 (grayscale)
    # IDAT (prefix each row with filter byte 0)
    raw = b''.join(b'\x00' + arr[i].tobytes() for i in range(h))
    idat = zlib.compress(raw)
    # IEND
    iend = b''

    # Assemble PNG
    png = png_sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', iend)

    with open(path, "wb") as f:
        f.write(png)
