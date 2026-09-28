import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from generator import Generator
from discriminator import Discriminator, VideoDiscriminator
from losses import generator_loss, bce_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device.upper()}\n")

# ── Test Generator ──────────────────────────────────────
G = Generator().to(device)
x = torch.randn(2, 3, 256, 256).to(device)
out = G(x)
assert out.shape == (2, 3, 256, 256), f"Generator shape wrong: {out.shape}"
print(f"✔ Generator OK: {out.shape}")

# ── Test Image Discriminator ────────────────────────────
D = Discriminator(pretrained=False).to(device)
score = D(x)
assert score.shape == (2, 1), f"Discriminator shape wrong: {score.shape}"
print(f"✔ Discriminator OK: {score.shape}")

# ── Test Video Discriminator ────────────────────────────
VD = VideoDiscriminator(pretrained=False).to(device)
clip = torch.randn(2, 8, 3, 256, 256).to(device)
score = VD(clip)
assert score.shape == (2, 1), f"VideoDiscriminator shape wrong: {score.shape}"
print(f"✔ VideoDiscriminator OK: {score.shape}")

# ── Test Losses ─────────────────────────────────────────
fake = G(x)
rp = D(x)
fp = D(fake.detach())
real_labels = torch.ones_like(rp)
fake_labels = torch.zeros_like(fp)
ld = (bce_loss(rp, real_labels) + bce_loss(fp, fake_labels)) / 2
lg = generator_loss(D(fake), fake, x)
print(f"✔ Losses OK — Loss_D: {ld.item():.4f}  Loss_G: {lg.item():.4f}")
# ── Test Backward Pass ──────────────────────────────────
ld.backward()
lg.backward()
print(f"✔ Backward pass OK")

print("\n All smoke tests passed!\n")