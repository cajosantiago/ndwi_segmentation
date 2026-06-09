import os
import glob
import cv2 as cv
import numpy as np
import subprocess

def main():
    print("=" * 80)
    print("   Cyanobacteria Distribution Over Time Video Generator (2022 - 2024)")
    print("=" * 80)

    img_dir = "MAGO_Maranhão_imgs"
    if not os.path.exists(img_dir):
        print(f"[ERROR] Directory '{img_dir}' does not exist! Run generate_mago_images.py first.")
        return

    # Find all cyanobacteria concentration map images (excluding _nb.png without text/colorbar)
    pattern = os.path.join(img_dir, "*_cyano_cells_mL.png")
    img_paths = sorted(glob.glob(pattern))

    if not img_paths:
        print("[ERROR] No cyanobacteria map images found in MAGO_Maranhão_imgs!")
        return

    total_images = len(img_paths)
    print(f"Found {total_images} images to compile into the timelapse video.")

    # Read the first image to establish the base resolution
    first_img = cv.imread(img_paths[0])
    if first_img is None:
        print(f"[ERROR] Could not read first image: {img_paths[0]}")
        return
        
    h, w, _ = first_img.shape
    # Ensure dimensions are even (H.264 libx264 codec requirement for universal playback)
    w = w if w % 2 == 0 else w + 1
    h = h if h % 2 == 0 else h + 1

    print(f"Base video resolution: {w} x {h}")

    temp_avi = "temp_lossless.avi"
    # Use MJPG lossless-like format for fast intermediate writing
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    
    # 2.0 frames per second (0.5 seconds per observation, giving a great timelapse flow)
    fps = 2.0
    video = cv.VideoWriter(temp_avi, fourcc, fps, (w, h))

    if not video.isOpened():
        print("[ERROR] Could not open VideoWriter!")
        return

    print("Stitching frames...")
    for idx, path in enumerate(img_paths):
        img = cv.imread(path)
        if img is None:
            print(f"  [WARN] Skipping unreadable image: {os.path.basename(path)}")
            continue
            
        # Resize to absolute target size (avoids mismatches from matplotlib bbox_inches variations)
        img_resized = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
        video.write(img_resized)
        
        # Display clean progress
        print(f"  [{idx+1}/{total_images}] Added: {os.path.basename(path)}")

    video.release()
    print("Intermediate lossless video successfully created.")

    # ── High-Compatibility H.264 MP4 Re-encoding ──
    print("Re-encoding video to high-compatibility MP4 using ffmpeg...")
    final_mp4 = "cyano_evolution_2022_2024.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_avi,
        "-c:v", "libx264",       # H.264 video codec
        "-pix_fmt", "yuv420p",   # YUV 4:2:0 chroma subsampling for browser compatibility
        "-r", str(fps),          # output framerate
        final_mp4
    ]

    try:
        subprocess.run(cmd, check=True)
        print("═" * 80)
        print(f"Timelapse video successfully generated: {final_mp4}")
        print("═" * 80)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg re-encoding failed with exit code {e.returncode}: {e}")
    finally:
        if os.path.exists(temp_avi):
            os.remove(temp_avi)

if __name__ == "__main__":
    main()
