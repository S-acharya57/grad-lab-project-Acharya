import sys, os, glob
import numpy as np
import cv2

frames_dir, video_path = sys.argv[1], sys.argv[2]

# file names are configured in this way in the Angelopoulos eye dataset 
paths = sorted(glob.glob(f"{frames_dir}/*.png"), key=lambda p: int(os.path.basename(p).split("_")[0]))
frames, timestamps_us = [], []
for p in paths:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # cv2 returns None on a corrupted file, so skipping it
        print(f"  WARNING: could not read {p} (corrupted?), skipping this frame")
        continue
    frames.append(img)
    timestamps_us.append(int(os.path.basename(p).split(".")[0].split("_")[-1]))

frames = np.stack(frames)
timestamps_us = np.array(timestamps_us, dtype="int64")
span_s = (timestamps_us[-1] - timestamps_us[0]) / 1e6
fps = len(frames) / span_s
print(f"{len(frames)} frames, {span_s:.1f}s, {fps:.2f}fps")

h, w = frames.shape[1], frames.shape[2]
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for frame in frames:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
writer.release()

with open(os.path.join(os.path.dirname(video_path), "fps.txt"), "w") as f:
    f.write(str(fps))
