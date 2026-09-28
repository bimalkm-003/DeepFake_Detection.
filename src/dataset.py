import cv2
import numpy as np
import random
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size, mode="train"):
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
            A.GaussNoise(noise_scale_factor=0.05, p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])


class DeepfakeImageDataset(Dataset):
    """Label: 1 = Real, 0 = Fake. Random samples every instantiation."""
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=10000):
        real_all = (
            [(p, 1) for p in Path(real_dir).glob("*.jpg")] +
            [(p, 1) for p in Path(real_dir).glob("*.png")]
        )
        fake_all = (
            [(p, 0) for p in Path(fake_dir).glob("*.jpg")] +
            [(p, 0) for p in Path(fake_dir).glob("*.png")]
        )
        real = random.sample(real_all, min(max_samples, len(real_all)))
        fake = random.sample(fake_all, min(max_samples, len(fake_all)))
        self.samples = real + fake
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.float32)


class DeepfakeVideoDataset(Dataset):
    """Returns clip tensor (N, C, H, W). Label: 1 = Real, 0 = Fake"""
    def __init__(self, real_dir, fake_dir, n_frames=8, transform=None, max_samples=10000):
        real_all = [(p, 1) for p in Path(real_dir).glob("*.mp4")]
        fake_all = [(p, 0) for p in Path(fake_dir).glob("*.mp4")]
        real = random.sample(real_all, min(max_samples, len(real_all)))
        fake = random.sample(fake_all, min(max_samples, len(fake_all)))
        self.samples = real + fake
        self.n_frames = n_frames
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        while len(frames) < self.n_frames:
            frames.append(frames[-1])
        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._extract_frames(path)
        tensors = []
        for f in frames:
            if self.transform:
                f = self.transform(image=f)["image"]
            tensors.append(f)
        clip = torch.stack(tensors, dim=0)
        return clip, torch.tensor(label, dtype=torch.float32)