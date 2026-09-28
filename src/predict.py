import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import cv2
from PIL import Image

from config import CFG
from dataset import get_transforms
from discriminator import Discriminator, VideoDiscriminator


def predict_image(image_path, ckpt_path, device=CFG.DEVICE):
    transform = get_transforms(CFG.IMG_SIZE, mode="val")
    img = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    D = Discriminator(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    D.load_state_dict(ckpt["D"])
    D.eval()

    with torch.no_grad():
        score = torch.sigmoid(D(tensor)).item()

    label = "REAL" if score >= 0.5 else "FAKE"
    print(f"  Prediction : {label}")
    print(f"  Confidence : {score:.4f}")
    return score, label


def predict_video(video_path, ckpt_path, device=CFG.DEVICE, n_frames=CFG.FRAMES_PER_CLIP):
    transform = get_transforms(CFG.IMG_SIZE, mode="val")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(image=frame)["image"])
    cap.release()
    while len(frames) < n_frames:
        frames.append(frames[-1])

    clip = torch.stack(frames).unsqueeze(0).to(device)  # (1, N, C, H, W)

    D = VideoDiscriminator(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    D.load_state_dict(ckpt["D"])
    D.eval()

    with torch.no_grad():
        score = torch.sigmoid(D(clip)).item()

    label = "REAL" if score >= 0.5 else "FAKE"
    print(f"  Prediction : {label}")
    print(f"  Confidence : {score:.4f}")
    return score, label


if __name__ == "__main__":
    # Example usage
    # predict_image("path/to/image.jpg", "checkpoints/latest_image.pth")
    # predict_video("path/to/video.mp4", "checkpoints/latest_video.pth")
    pass 
