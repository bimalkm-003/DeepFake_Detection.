# Deepfake Detection (Group 31)

Deepfake detection system using supervised GAN and confidence scoring on Celeb-DF-v2.
Final-year B.Tech capstone project. Full design: see PLAN.md.

## Status
- **Current step:** implement `src/preprocess_faces.py`
- **Next step:** `src/make_splits.py`
*(Rule: Update this status block after every completed step.)*

## Environment
- Stack: PyTorch, timm, facenet-pytorch (MTCNN), albumentations, scikit-learn, ONNX Runtime.
- Training: Google Colab (T4 GPU) with data and checkpoints stored on Google Drive.
- Local machine: Editing, small smoke tests, and git operations only. Never run heavy jobs locally.

## Global Conventions
- Labels: `0 = Real`, `1 = Fake`. Score = `P(fake)` in `[0, 1]`.
- Input: 224×224 MTCNN face crops with ImageNet normalization (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`).
- Splits: Identity-disjoint. Fixed seeds for all runs.
- Thresholds: Frame threshold ($\tau_{\text{frame}}$) and video threshold ($\tau_{\text{video}}$) chosen independently on val, then frozen for test.

## Integrity Rules
- Train-only fitting: Normalization stats, PCA, and GMM/EM are fitted on TRAIN only.
- Generator / StyleGAN outputs exist ONLY in train — never in val or test.
- Test sets (`test.csv`, `test_official.csv`) are evaluated ONCE per final model.
- A fake video enters a split only if BOTH of its constituent identities belong to that split.
- Every run logs git commit hash, full config, and library versions.

## Stage Plan
1. **Stage 1**: Supervised EfficientNet-B0 baseline with class-weighted BCE → establish Tier 1 baseline.
2. **Stage 2**: Adversarial GAN fine-tuning with U-Net generator fakes (TTUR, frozen-D handling).
3. **Stage 3**: StyleGAN2-ADA diverse fakes (offline generation, nuisance-feature shortcut probe).
4. **EM Head**: Gaussian Mixture on frozen embeddings (ships only if it beats sigmoid on val ECE without AUC loss).
5. **Export & App**: ONNX FP32 (default) export, static INT8 QDQ, benchmarking, Flask app integration.

## Rules for the coding assistant
- One step per turn. Plan first and write code only after user approval.
- Never run training or heavy computation locally.
- Never commit data files, `.pth`, `.onnx`, or Google Drive paths. Never overwrite checkpoints.
- Show diff summary and stop after every step.
- If PLAN.md and user message disagree, ask the user.
