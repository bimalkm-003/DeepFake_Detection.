import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from config import CFG
from dataset import DeepfakeImageDataset, DeepfakeVideoDataset, get_transforms
from discriminator import Discriminator, VideoDiscriminator


def evaluate_model(ckpt_path, mode="image"):
    device = CFG.DEVICE
    transform = get_transforms(CFG.IMG_SIZE, mode="val")

    # Dataset
    if mode == "image":
        dataset = DeepfakeImageDataset(CFG.REAL_DIR, CFG.FAKE_DIR, transform)
        D = Discriminator(pretrained=False).to(device)
    else:
        dataset = DeepfakeVideoDataset(
            CFG.REAL_DIR, CFG.FAKE_DIR,
            n_frames=CFG.FRAMES_PER_CLIP,
            transform=transform
        )
        D = VideoDiscriminator(pretrained=False).to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    D.load_state_dict(ckpt["D"])
    D.eval()

    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            scores = torch.sigmoid(D(data)).squeeze(1).cpu().numpy()
            preds  = (scores >= 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().astype(int))

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*40}")
    print(f"  Mode      : {mode.upper()}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"{'='*40}\n")

    return {"accuracy": acc, "auc": auc, "f1": f1, "cm": cm}


if __name__ == "__main__":
    evaluate_model(ckpt_path="checkpoints/latest_image.pth", mode="image") 
