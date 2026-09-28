import cv2
import numpy as np
import os
from pathlib import Path

VIDEO_DIR  = "E:/datasets for local training/Celeb-DF-v2"
OUTPUT_DIR = "C:/deepfake_project/data/fake"
MAX_FRAMES = 10000
FRAMES_PER_VIDEO = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
count = 0

for vpath in Path(VIDEO_DIR).rglob("*.mp4"):
    if count >= MAX_FRAMES:
        break
    cap = cv2.VideoCapture(str(vpath))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, FRAMES_PER_VIDEO, dtype=int)
    for i, fi in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            fname = f"{vpath.stem}_f{i}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, fname), frame)
            count += 1
    cap.release()
    print(f"  Processed {vpath.name} — total frames: {count}")

print(f"\nDone! Extracted {count} fake frames → {OUTPUT_DIR}")