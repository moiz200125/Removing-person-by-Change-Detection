import os
import argparse
import numpy as np
from utils.io_utils import read_frames, plot_frames,write_png
from utils.backgroundmodel import compute_mean, compute_variance
from utils.masking import compute_mask

def save_mean_and_variance(mean_frame: np.ndarray, var_frame: np.ndarray, out_dir: str):
    """
    Save mean and variance frames as PNG images.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save mean directly (clip to 0–255)
    mean_u8 = np.clip(mean_frame, 0, 255).astype(np.uint8)
    write_png(os.path.join(out_dir, "mean.png"), mean_u8)

    # Variance visualization (linear scaling)
    eps = 1e-6
    var_min, var_max = float(var_frame.min()), float(var_frame.max())
    if var_max - var_min < 1e-9:
        var_vis = np.zeros_like(var_frame, dtype=np.uint8)
    else:
        var_vis = ((var_frame - var_min) / (var_max - var_min + eps) * 255.0).astype(np.uint8)
    write_png(os.path.join(out_dir, "variance.png"), var_vis)

    # Optional: stddev visualization
    std = np.sqrt(var_frame)
    std_vis = ((std / (std.max() + eps)) * 255.0).astype(np.uint8)
    write_png(os.path.join(out_dir, "stddev.png"), std_vis)


def changeDetection(input_folder, output_folder, input_ext, output_ext, video_format):
    # Step 1: Read frames
    print(f"[INFO] Reading frames from {input_folder}")
    frames = read_frames(input_folder)
    print(f"[INFO] Loaded {frames.shape[0]} frames of size {frames.shape[1:]}")

    # Step 2: Visualization (save first 10 frames into a grid for report)
    plot_path = os.path.join(output_folder, "frames_preview.png")
    os.makedirs(output_folder, exist_ok=True)
    plot_frames(frames, num_frames=10, save_name=plot_path)
    print(f"[INFO] Preview frames saved to {plot_path}")

    # Step 3: Background model
    t = 60 if "person3" in input_folder.lower() else 70
    print(f"[INFO] Using first {t} frames for background model")
    mean_frame = compute_mean(frames[:t])
    var_frame = compute_variance(frames[:t], mean_frame)

    # Save mean/variance for report
    save_mean_and_variance(mean_frame, var_frame, output_folder)

    # Step 4: Compute masks for multiple thresholds
    thresholds = [2.0, 5.0, 8.0]
    for thr in thresholds:
        thr_dir = os.path.join(output_folder, f"thr_{thr}")
        os.makedirs(thr_dir, exist_ok=True)
        print(f"[INFO] Generating masks at threshold={thr}")

        for i, frame in enumerate(frames[t:]):  # process frames after background window
            mask = compute_mask(frame, mean_frame, var_frame,
                                threshold=thr, min_std=1.0)
            save_path = os.path.join(thr_dir, f"mask_{i:04d}.{output_ext}")
            write_png(save_path, mask)

    print(f"[INFO] Mean/Variance + Masks saved under {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Detection using Classical CV")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to input folder containing frames")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to save masks and video")
    parser.add_argument("--input_ext", type=str, default="png",
                        help="Extension of input images (png, jpg, jpeg)")
    parser.add_argument("--output_ext", type=str, default="png",
                        help="Extension of output masks (png, jpg)")
    parser.add_argument("--video_format", type=str, default="mp4",
                        help="Video format for saving output (mp4, avi)")
    args = parser.parse_args()

    changeDetection(args.input_folder, args.output_folder,
                    args.input_ext, args.output_ext, args.video_format)
