import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from config import CFG
from dataset import DeepfakeImageDataset, DeepfakeVideoDataset, get_transforms
from generator import Generator
from discriminator import Discriminator, VideoDiscriminator
from losses import generator_loss, bce_loss

def save_checkpoint(epoch, G, D, opt_G, opt_D, path):
    torch.save({
        "epoch": epoch,
        "G":     G.state_dict(),
        "D":     D.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
    }, path)
    print(f"  ✔ Checkpoint saved → {path}")

def load_checkpoint(path, G, D, opt_G, opt_D):
    ckpt = torch.load(path, map_location=CFG.DEVICE)
    G.load_state_dict(ckpt["G"])
    D.load_state_dict(ckpt["D"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    print(f"  ✔ Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]

def train(mode="image"):
    device = CFG.DEVICE
    transform = get_transforms(CFG.IMG_SIZE, mode="train")
    
    # 1. Instantiate Models (Once)
    if mode == "image":
        G = Generator().to(device)
        D = Discriminator().to(device)
    else:
        G = Generator().to(device)
        D = VideoDiscriminator().to(device)
        
    # 2. Instantiate Optimizers (Once)
    opt_G = optim.Adam(G.parameters(), lr=CFG.LR_G, betas=(CFG.BETA1, CFG.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=CFG.LR_D, betas=(CFG.BETA1, CFG.BETA2))
    
    # 3. Resume if checkpoint exists
    start_epoch = 0
    ckpt_path = Path(CFG.CKPT_DIR) / f"latest_{mode}.pth"
    if ckpt_path.exists():
        start_epoch = load_checkpoint(str(ckpt_path), G, D, opt_G, opt_D)
        
    print(f"\n{'='*50}")
    print(f"  Device: {device.upper()}  |  Mode: {mode.upper()}")
    print(f"  Epochs: {start_epoch} to {CFG.NUM_EPOCHS}")
    print(f"{'='*50}\n")
    
    # 4. Training Loop
    for epoch in range(start_epoch, CFG.NUM_EPOCHS):
        
        # ── FRESH RANDOM SAMPLING EVERY EPOCH ──
        if mode == "image":
            dataset = DeepfakeImageDataset(CFG.REAL_DIR, CFG.FAKE_DIR, transform, CFG.MAX_SAMPLES)
        else:
            dataset = DeepfakeVideoDataset(
                CFG.REAL_DIR, CFG.FAKE_DIR, 
                n_frames=CFG.FRAMES_PER_CLIP, 
                transform=transform, 
                max_samples=CFG.MAX_SAMPLES
            )
            
        loader = DataLoader(
            dataset, batch_size=CFG.BATCH_SIZE,
            shuffle=True, num_workers=2, pin_memory=True
        )
        
        G.train()
        D.train()
        total_G, total_D = 0.0, 0.0
        
        for i, (data, labels) in enumerate(loader):
            if mode == "image":
                real_imgs = data.to(device)
            else:
                clips = data.to(device)
                real_imgs = clips[:, 0]
                
            # ── Train Discriminator ──
            opt_D.zero_grad()
            fake_imgs  = G(real_imgs).detach()
            real_preds = D(real_imgs) if mode == "image" else D(clips)
            fake_preds = D(fake_imgs) if mode == "image" else D(
                fake_imgs.unsqueeze(1).expand_as(clips)
            )
            
            real_labels = torch.ones_like(real_preds) * 0.9
            fake_labels = torch.zeros_like(fake_preds)
            loss_D = (bce_loss(real_preds, real_labels) + bce_loss(fake_preds, fake_labels)) / 2
            loss_D.backward()
            opt_D.step()
            
            # ── Train Generator (2x per D update) ──
            for _ in range(2):
                opt_G.zero_grad()
                fake_imgs  = G(real_imgs)
                fake_preds = D(fake_imgs) if mode == "image" else D(
                    fake_imgs.unsqueeze(1).expand_as(clips)
                )
                loss_G = generator_loss(fake_preds, fake_imgs, real_imgs, CFG.LAMBDA_L1)
                loss_G.backward()
                opt_G.step()
                
            total_G += loss_G.item()
            total_D += loss_D.item()
            
            if i % 50 == 0:
                print(
                    f"  Epoch [{epoch+1}/{CFG.NUM_EPOCHS}] "
                    f"Batch [{i}/{len(loader)}]  "
                    f"Loss_G: {loss_G.item():.4f}  "
                    f"Loss_D: {loss_D.item():.4f}"
                )
                
        avg_G = total_G / len(loader)
        avg_D = total_D / len(loader)
        print(f"\n  ✦ Epoch {epoch+1} — Avg Loss_G: {avg_G:.4f}  Avg Loss_D: {avg_D:.4f}\n")
        
        if (epoch + 1) % CFG.SAVE_EVERY == 0:
            save_checkpoint(
                epoch + 1, G, D, opt_G, opt_D,
                str(Path(CFG.CKPT_DIR) / f"ckpt_{mode}_epoch{epoch+1}.pth")
            )
        save_checkpoint(epoch + 1, G, D, opt_G, opt_D, str(ckpt_path))
        
    print("Training complete!")
    return G, D

if __name__ == "__main__":
    train(mode="image")