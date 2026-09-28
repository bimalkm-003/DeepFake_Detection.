# Deepfake Detection System — Design & Implementation Plan

**Project:** Celeb-DF-v2 Deepfake Detector (Group 31)  
**Date:** 2026-09-24  
**Status:** Planning phase — no code written yet

---

## Current Status / Next Step

**Last updated:** 2026-09-24

**Status:** Plan under revision — applying BATCH 1 amendments.

**Next steps:** 
1. Finish all amendments to PLAN.md (BATCH 1 in progress)
2. Create CLAUDE.md (project context for Claude Code)
3. Implement `src/preprocess_faces.py`

**Blockers:** None.

---

## 1. Audit Findings

### 1.1 Critical Issues in Current Codebase

| Issue | Location | Impact | Severity |
|-------|----------|--------|----------|
| **Discriminator trained on wrong task** | `src/train.py:93-104` | D learns "dataset frame vs U-Net output," not "real vs fake." Dataset labels never used in training. | 🔴 Critical |
| **Data leakage** | `src/dataset.py`, `src/train.py`, `src/evaluate.py` | Train and eval read from same dirs. Frame-level random sampling allows same-video frames in both splits. No identity-level separation. | 🔴 Critical |
| **Buggy AUC metric** | `src/evaluate.py:49` | `roc_auc_score` receives binary predictions, not continuous scores → computes balanced accuracy, not AUC. | 🔴 Critical |
| **Wrong normalization** | `src/dataset.py:20` | Uses `mean=0.5, std=0.5` (range [-1,1]), but EfficientNet-B4 is pretrained on ImageNet (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`). | 🟠 High |
| **No class imbalance handling** | `src/train.py`, `src/losses.py` | Test set is 704 real : 6768 fake (~1:9.6 ratio). No `pos_weight`, no weighted sampler, no stratification. | 🟠 High |
| **No validation set** | `src/train.py` | Training runs for fixed epochs with no early stopping. Best model = last model. | 🟠 High |
| **Full-frame input** | `src/dataset.py`, `extract_frames.py` | No face detection/cropping. Model sees backgrounds, shoulders, text overlays — irrelevant to face manipulation. | 🟡 Medium |
| **No model checkpoints committed** | `checkpoints/` | Only `placeholder.txt` exists locally. `.pth` files live on Google Drive and aren't in version control. | 🟡 Medium |
| **Empty notebooks** | `notebooks/01_*.ipynb`, `02_*.ipynb`, `03_*.ipynb` | Only `deepfake_gan.ipynb` has content. Others are placeholders. | 🟡 Low |

### 1.2 What Works

- **Model architecture**: EfficientNet-B4 backbone + MLP head is a solid choice (though B0 is faster with similar accuracy).
- **Augmentation**: Horizontal flip, ColorJitter, GaussNoise are appropriate for face forgery detection.
- **Framework hygiene**: Uses `timm` for backbones, Albumentations for transforms, proper train/eval mode switching.
- **Modular code structure**: Separate files for config, dataset, models, training, evaluation.

---

## 2. Design Decisions

### 2.1 Label Convention (Global Standard)

**0 = Real, 1 = Fake** everywhere (CSVs, dataset, loss, eval, predict).

**Rationale:**
- Intuitive: 1 = positive detection = deepfake detected
- Avoids semantic confusion from old GAN setup where "high D score" meant "dataset frame"
- Standard in binary anomaly detection (1 = anomaly)

### 2.2 Model Architecture: Supervised GAN Hybrid

**Goal:** Train a discriminator that generalizes beyond Celeb-DF-v2 synthesis artifacts by seeing multiple fake sources.

#### Components

| Component | Architecture | Purpose | Stage |
|-----------|-------------|---------|-------|
| **Discriminator (D)** | Configurable backbone (default: `efficientnet_b0`, 224×224) + 2-layer MLP → logit | Binary classifier: real (0) vs fake (1) | Train + Inference |
| | **Embeddings hook**: extract 1280-d (B0) or 1792-d (B4) features before final head | For EM-based confidence scoring | Inference optional |
| **Generator 1 (G_unet)** | U-Net (6 encoder/decoder stages, skip connections, 256×256) | Produce low-quality fakes for D to learn reconstruction artifacts | Training only (Stage 2) |
| **Generator 2 (G_stylegan)** | StyleGAN2-ADA pretrained on FFHQ-256 | Produce high-quality fakes for D to learn GAN fingerprints | Training only (Stage 3) |
| **EM Mixture Model** | EM mixture head on D's embeddings (spec in section 4.3) | Alternative confidence score via log-likelihood ratio | Inference optional |

#### Why Two Generators?

- **U-Net (Stage 2)**: Cheap to train, produces visible artifacts (blur, color shifts). Teaches D to catch low-hanging fruit.
- **StyleGAN2-ADA (Stage 3)**: State-of-the-art quality, produces imperceptible fakes. Teaches D to learn spectral/statistical fingerprints that generalize to unseen GAN architectures.
- Both run only during training — neither ships with the detector.

#### Discriminator Backbone Options

| Backbone | Input Size | Params | ImageNet Top-1 | Speed | Use Case |
|----------|-----------|--------|----------------|-------|----------|
| `efficientnet_b0` | 224×224 | 5.3M | 77.7% | Fast | **Default** — best speed/accuracy tradeoff for real-time inference |
| `efficientnet_b4` | 380×380 (or 256×256) | 19.3M | 83.0% | Medium | Higher accuracy, more compute |
| `efficientnet_b7` | 600×600 (or 256×256) | 66.3M | 84.3% | Slow | Research/benchmark ceiling |
| `resnet50` | 224×224 | 25.6M | 80.4% | Medium | Ablation baseline |

**Decision**: Default to `efficientnet_b0` for production. Train B4 and B7 variants for ablation studies.

---

## 3. Non-negotiable Integrity Rules

These rules ensure reproducible, leak-free experiments:

1. **Identity-disjoint splits**: No identity appears in more than one of train/val/test. Celeb-real videos are grouped by identity. A Celeb-synthesis video belongs to a split only if **BOTH** of its identities belong to that split (see section 4.1 Step 3). YouTube-real videos are split at the video level.

2. **Train-only statistics**: Normalization statistics (mean, std), PCA transformations, and EM/GMM mixture parameters are fitted on the **TRAIN split only**. Validation is used for early stopping, threshold selection (EER or Youden's J, then frozen for test), and calibration. Test is evaluated **once per final model**.

3. **Generator outputs in train only**: U-Net and StyleGAN2 generated frames exist only in the training split. They **never** appear in validation or test sets.

4. **Full reproducibility**: All experiments are seeded (`random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`, `torch.backends.cudnn.deterministic=True` where feasible). Every training run logs:
   - Git commit hash
   - Full config (hyperparameters, data paths, model architecture)
   - Library versions (PyTorch, timm, albumentations, sklearn, Python)

5. **Label and score convention**: **0 = Real, 1 = Fake** everywhere (CSVs, dataset, loss targets, predictions). Model output score = **P(fake)**. High score means high confidence of manipulation.

---

## 4. Training Pipeline

### 4.1 Preprocessing (Run Once)

#### Step 1: Frame Extraction
**Script:** `extract_frames.py` (already exists, modify to handle all sources)

**Input:**
- `Celeb-DF-v2/Celeb-real/*.mp4`
- `Celeb-DF-v2/YouTube-real/*.mp4`
- `Celeb-DF-v2/Celeb-synthesis/*.mp4`

**Output:** `data/frames/` (flat directory, ~50k JPGs)

**Method:**
- Extract 8 uniformly-spaced frames per video
- Filename: `{video_stem}_f{frame_idx}.jpg`

#### Step 2: Face Detection & Cropping
**Script:** `src/preprocess_faces.py` (new)

**Input:** `data/frames/`  
**Output:** `data/frames_cropped/` + `data/face_crop_manifest.json`

**Method:**
- MTCNN (`facenet-pytorch`) detects faces
- If 1 face: crop with 1.3× margin, resize to 224×224, save with same filename
- If 0 or >1 faces: log failure, skip frame
- Manifest logs: total, success, failures (no_face, multi_face, error)

**Expected drop rate:** ~5-10% (videos with occlusions, profile shots, multiple faces)

#### Step 3: Identity-Disjoint Split
**Script:** `src/make_splits.py` (new)

**Input:** `data/frames_cropped/`, `face_crop_manifest.json`  
**Output:** `data/train.csv`, `data/val.csv`, `data/test.csv`, `data/test_official.csv`

**Method:**
1. Parse video stems to extract identities:
   - `id<N>_<clip>` (Celeb-real) → identity = `N`, label = 0
   - `id<A>_id<B>_<clip>` (Celeb-synthesis) → **UNVERIFIED**: check Celeb-DF-v2 README / list files to confirm whether `id<A>` or `id<B>` is the source vs target face. Until verified, treat both `A` and `B` as belonging to this video for split assignment.
   - `<number>` (YouTube-real) → identity = `yt_<stem>`, label = 0
2. Collect unique Celeb identities, shuffle with `CFG.SEED`, split 70/15/15 by identity count.
3. **Exclusion rule for fakes**: A Celeb-synthesis video enters a split only if **BOTH** of its identities belong to that split's identity set. Otherwise, exclude the video and log the count of excluded videos.
4. Assign all frames from an identity's videos to one split (no identity spans multiple splits).
5. Assign YouTube-real videos randomly at the video level (no identity grouping), proportionally distributed across splits.
6. **Minimum counts**: Require at least 30 real AND 30 fake videos in each of val and test. If not met, retry with a different seed and log the attempt.
7. Filter to only frames in `face_crop_manifest` (successful crops).
8. Write CSVs with columns: `path` (relative), `label`, `video_id`, `source`.
9. **Official test set**: Also generate `data/test_official.csv` from the official Celeb-DF-v2 testing list (`List_of_testing_videos.txt`). Generate an identity-overlap report comparing `test_official` identities against `train` identities. Report metrics on both test sets (`test.csv` for leak-free eval, `test_official.csv` for literature comparison). Literature comparisons are only allowed on the official list and must explicitly state whether the comparison is identity-disjoint.

**CSV columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `path` | str | Relative path from `data/frames_cropped/` | `id5_id1_0003_f2.jpg` |
| `label` | int | 0 = real, 1 = fake | `1` |
| `video_id` | str | Video stem (for video-level aggregation) | `id5_id1_0003` |
| `source` | str | `celeb_real`, `yt_real`, `celeb_synth` | `celeb_synth` |

**Dataset size estimate:**
- Celeb-DF-v2 has ~5,639 fake and ~890 real source videos × 8 frames ≈ 86% fake before filtering.
- Actual frame and video counts per split will be recorded in the split summary log after face detection filtering.

---

### 4.2 Training: Supervised GAN (3 Stages)

#### Stage 1: Supervised Baseline (Celeb-DF only)
**Script:** `src/train_classifier.py` (new)

**Architecture:** `Discriminator(backbone="efficientnet_b0", pretrained=True)`

**Data:**
- Positive (fake) = Celeb-synthesis frames from `train.csv`
- Negative (real) = Celeb-real + YouTube-real frames from `train.csv`
- Class weight: `pos_weight = count(real) / count(fake)` in `BCEWithLogitsLoss`

**Hyperparameters:**
```python
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER = "AdamW"
AMP = True  # Automatic Mixed Precision (fp16)
PATIENCE = 5  # Early stopping
METRIC = "auc"  # Validation metric to maximize
```

**Training loop:**
1. Each epoch: train on `train.csv`, validate on `val.csv`
2. Validation: compute frame-level AUC (continuous scores → `roc_auc_score`)
3. Save `checkpoints/latest_classifier.pth` every epoch (full state: model, optimizer, epoch, best_val_auc)
4. Save `checkpoints/best_classifier.pth` when val AUC improves (model state dict only)
5. Early stop if no improvement for 5 epochs
6. Log to console + `logs/training.csv`: epoch, train_loss, val_loss, val_acc, val_auc, val_f1

**Expected result:** To be measured — this establishes Tier 1 baseline.

#### Stage 2: Adversarial GAN Fine-Tuning (U-Net Fakes)
**Script:** `src/train_gan_hybrid.py` (new)

**Architecture:** `Discriminator` (initialized from Stage 1 best checkpoint) + `Generator` (U-Net, 224×224 input/output).

**Initial Hyperparameters (tune on val):**
- **Stage 2 adversarial training**: 20 epochs, $\text{LR}_D = 5 \times 10^{-5}$, $\text{LR}_G = 2 \times 10^{-4}$
- **Generator warm-up**: 5 epochs (L1 reconstruction only)
- Batch size, optimizer (AdamW), AMP, and weight decay inherit from Stage 1

**True Adversarial Game Design:**
1. **Warm-up phase**: Pretrain U-Net Generator $G$ independently for $N=5$ epochs using $L_1$ reconstruction loss (optionally + perceptual loss via VGG/LPIPS) on real training frames, so initial generator outputs are recognizable face reconstructions rather than pure noise.
2. **Alternating optimization loop (each iteration)**:
   - **(i) Discriminator step**:
     - Forward pass on batch: $x_{\text{real}}$ (label 0, weight 1.0) + $x_{\text{celeb\_synth}}$ (label 1, weight 1.0) + $G(x_{\text{real}}).\text{detach}()$ (label 1, per-source weight $w_G \in [0.5, 1.0]$).
     - Compute $\mathcal{L}_D = \text{BCEWithLogitsLoss}$ with `pos_weight` and per-source loss weighting.
     - Update $D$ weights via `opt_D.step()`.
   - **(ii) Generator step (with $D$ frozen)**:
     - Set $D$ to eval mode for Dropout (`D.eval()`), but retain gradient computation for $G$'s forward path. Handle BatchNorm: keep running stats frozen during the $G$ step to prevent generator fakes from corrupting $D$'s batch statistics.
     - Forward pass: $x_{\text{fake}} = G(x_{\text{real}})$.
     - Compute $\mathcal{L}_G = \text{BCEWithLogitsLoss}(D(x_{\text{fake}}), \mathbf{0}) + \lambda_{L1} \cdot \|x_{\text{fake}} - x_{\text{real}}\|_1$ (+ optional frequency-domain spectral loss). Target is 0 because $G$ aims to fool $D$ into predicting Real (0).
     - Update $G$ weights via `opt_G.step()`.
   - **Two-Time-Scale Learning Rates (TTUR)**: Use $\text{LR}_G > \text{LR}_D$ (e.g., $\text{LR}_G = 2\times 10^{-4}$, $\text{LR}_D = 5\times 10^{-5}$) to ensure stable adversarial dynamics without mode collapse.

**Monitoring & Abort Criteria:**
- **Per-epoch metrics to track**:
  - $D$ detection rate on $G$-generated fakes ($\%$ of $G(x)$ classified as Fake with score $\ge 0.5$).
  - Generator loss $\mathcal{L}_G$ trajectory.
  - Validation AUC on `val.csv` (Celeb-DF data only — **no $G$ fakes in validation**).
  - Per-source validation metrics (Celeb-real, YouTube-real, Celeb-synthesis).
- **Tuning and abort conditions**:
  - **(a)** Detection near 100% early on is **expected and normal** (not an abort condition). $D$ initially dominates the untrained $G$.
  - **(b)** If detection stays at 100% AND $\mathcal{L}_G$ is not decreasing for $\ge 3$ consecutive epochs, $G$ is not learning. **Tuning action** (not abort): raise $\text{LR}_G$ by 2× or lower $\lambda_{L1}$ by 0.5×.
  - **(c) Abort conditions**:
    - Detection collapses to near 0% for $\ge 3$ consecutive epochs (generator overpower, $D$ can't learn).
    - Validation AUC falls $>0.02$ below the Stage 1 baseline ($D$ is regressing).
    - `yt_real` validation specificity drops $>3.0$ percentage points below Stage 1 ($D$ learned a false shortcut).

**Risk Note**: Blurry or low-fidelity $G$ outputs can teach $D$ that "low resolution / blur = fake", severely damaging specificity on compressed real videos (especially YouTube-real). The warm-up phase, $L_1$ penalty, and $w_G$ weighting mitigate this.

#### Stage 3: StyleGAN2-ADA Diverse Fakes (Optional Extension)
**Script:** `src/train_gan_hybrid.py` with `--generator stylegan2`

**Architecture:** `Discriminator` (from Stage 2) + StyleGAN2-ADA (pretrained FFHQ checkpoint, frozen).

**Initial Hyperparameters (tune on val):**
- **Stage 3 fine-tuning**: 10 epochs, $\text{LR}_D = 5 \times 10^{-5}$
- Batch size, optimizer, AMP inherit from Stage 2

**Pretrained Weights & Colab Compatibility:**
- The standard NVIDIA release provides FFHQ at 1024×1024 (`ffhq.pkl`). Downsample generated samples to 224×224 via bicubic interpolation with antialiasing.
- Custom CUDA plugins (`upfirdn2d`, `bias_act`) require Ninja compilation. If compilation fails in the Colab environment, use the pure-PyTorch fallback implementation (`torch.nn.functional` equivalents).

**Offline Sample Generation & Leak-Free Pipeline:**
1. Pre-generate 10,000 StyleGAN2 face samples offline with mixed truncation $\psi \in [0.7, 1.0]$ to balance visual diversity and quality.
2. Pass all generated samples through the **EXACT same pipeline** as real training frames: MTCNN face detection $\to$ 1.3× bounding-box crop $\to$ 224×224 resize $\to$ JPEG re-encoding (quality 85–95).
3. **Shortcut Prevention**: To prevent $D$ from learning an "FFHQ image style / alignment = fake" shortcut:
   - Add a matched set of real FFHQ crops (label 0) processed through the identical pipeline to the training pool. Cap real FFHQ crops at **no more than 25%** of the real training pool and log the ratio.
   - **(a) Nuisance-feature probe**: Train a small classifier **from scratch** on low-level statistics only (face bounding-box relative size, mean and std of RGB channels, Laplacian variance as a blur measure, JPEG file size in bytes) to separate StyleGAN crops from real FFHQ crops. Accuracy near **chance (~50%)** means the pipelines are matched. High accuracy (>70%) means a preprocessing mismatch exists — fix the pipeline before training Stage 3.
   - **(b) Held-out FFHQ validation**: Hold out 1,000 real FFHQ crops that are never trained on. After Stage 3 training, $D$'s specificity on these held-out FFHQ crops must be **at least equal to** the Stage 2 `yt_real` validation specificity (no regression on out-of-distribution reals).
4. **Data Isolation**: StyleGAN2 and FFHQ samples are included in the **training split only**. They never appear in `val.csv` or `test.csv`.

---

### 4.3 EM-Based Confidence Scoring (Post-Training)

**Goal:** Replace the sigmoid head with a probabilistic score based on embedding distributions.

**Script:** `src/train_em_mixture.py` (new)

**Method:**
1. Load best discriminator checkpoint from Stage 2 or 3 and freeze it.
2. Extract embeddings (penultimate layer, pre-logit features: 1280-d for B0, 1792-d for B4) from **entire training set** (`train.csv` only).
3. **Standardize** embeddings: subtract mean and divide by std (fitted on train embeddings).
4. **Optional PCA**: Apply PCA to retain 95% variance, capped at 64–128 dimensions for computational efficiency. This reduces noise and improves GMM fitting.
5. **Per-class Gaussian Mixture fitting**:
   - Fit separate `sklearn.mixture.GaussianMixture` for real (label 0) and fake (label 1) embeddings.
   - Number of components $k \in \{1, 2, 3, 4, 6, 8\}$, selected by **BIC** (Bayesian Information Criterion) on train embeddings.
   - Covariance type $\in \{\text{diag}, \text{tied}\}$. Use `full` only if dimensionality $\le 64$ (otherwise too many parameters).
   - Initialization: `k-means++`, `n_init \ge 5` random restarts.
   - Tune `reg_covar` (covariance regularization) to prevent singular matrices.
   - **Note**: $k=1$ per class is equivalent to closed-form class-conditional Gaussians (no EM needed). Report both $k=1$ baseline and BIC-selected $k$ to quantify the EM contribution.
6. **Calibration**: Compute log-likelihood ratio $\text{LLR} = \log p(e | \text{fake}) - \log p(e | \text{real})$ on train embeddings. Fit Platt scaling or temperature scaling on **validation set** to map LLR → calibrated $P(\text{fake})$.
7. Save: standardization params, PCA transform (if used), GMM parameters (μ, Σ, weights per component per class), calibration params → `checkpoints/em_mixture.pkl`.

**Inference:**
1. Extract embedding $e$ from input frame (frozen discriminator).
2. Standardize $e$, apply PCA transform (if trained with PCA).
3. Compute per-class log-likelihoods: $\ell_{\text{real}} = \log p(e | \text{real})$, $\ell_{\text{fake}} = \log p(e | \text{fake})$.
4. Compute $\text{LLR} = \ell_{\text{fake}} - \ell_{\text{real}}$.
5. Apply calibration: $P(\text{fake}) = \sigma(\alpha \cdot \text{LLR} + \beta)$ where $\alpha, \beta$ are from calibration fit.
6. **Unfamiliarity score** (optional): $\text{max}(\ell_{\text{real}}, \ell_{\text{fake}})$ — low values indicate out-of-distribution inputs.

**Comparison:**
- **Sigmoid head**: $P(\text{fake}) = \sigma(D.\text{head}(e))$ — single linear layer.
- **EM mixture**: $P(\text{fake}) = \text{calibrated}(\log p(e|\text{fake}) - \log p(e|\text{real}))$ — captures intra-class multimodality.

**Evaluation on test set:**
- Report: AUC, EER, accuracy, F1, **ECE** (Expected Calibration Error), reliability diagram.
- Compare sigmoid vs EM mixture ($k=1$ vs BIC-selected $k$).
- **K-fold ablation**: Fit GMM on out-of-fold embeddings (5-fold split of training set) to check for overfitting to train distribution.
- **Out-of-distribution check**: Evaluate on an unseen manipulation type (e.g., FaceSwap, Deepfakes, Face2Face from FaceForensics++) and report AUC + unfamiliarity score distribution.

**Hypotheses to Test:**
- **H1**: EM mixture with $k>1$ may capture intra-class multimodality in embeddings better than a single linear head, particularly for out-of-distribution inputs.
- **H2**: The benefit of EM over sigmoid may be larger when train and test distributions differ (e.g., cross-dataset evaluation).

**Shipping Decision**: The EM mixture head ships as the default inference method **only if** it beats the sigmoid head on validation ECE (calibration) without losing AUC ($\Delta \text{AUC} \le 0.005$). Otherwise, ship the sigmoid head and keep EM as an optional research variant.

---

## 5. Evaluation & Metrics

### 5.1 Evaluation Script
**Script:** `src/evaluate.py` (refactored)

**Function signature:**
```python
evaluate_model(ckpt_path, split="test", method="sigmoid")
# method: "sigmoid" (default) or "em" (EM mixture)
```

**Metrics computed:**

#### Frame-Level (Primary)
- **Accuracy**: `accuracy_score(labels, preds)`
- **AUC-ROC**: `roc_auc_score(labels, scores)` — uses continuous scores (not binary predictions).
- **Equal Error Rate (EER)**: Threshold where False Positive Rate = False Negative Rate (computed from ROC curve).
- **F1 Score**: `f1_score(labels, preds, pos_label=1)` — fake is positive class.
- **Precision**: `precision_score(labels, preds, pos_label=1)`.
- **Recall (TPR)**: `recall_score(labels, preds, pos_label=1)`.
- **Specificity (TNR)**: `confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1])`.
- **Confusion Matrix**: 2×2 table (rows=actual, cols=predicted).
- **Calibration Error (ECE)**: Expected Calibration Error in 10 bins.

#### Threshold Selection
- The decision threshold $\tau$ is **NOT fixed at 0.5**. Frame-level and video-level thresholds are **selected independently** on the **validation set**:
  - **Frame-level threshold** $\tau_{\text{frame}}$: Selected using Youden's J statistic ($\arg\max_\tau (\text{TPR}(\tau) - \text{FPR}(\tau))$) or EER point ($\text{FPR}(\tau) \approx \text{FNR}(\tau)$) on frame-level validation scores.
  - **Video-level threshold** $\tau_{\text{video}}$: Selected using the same method on video-aggregated validation scores (after selecting the best aggregation method).
- Once selected on validation, both thresholds are **frozen** and applied directly to their respective test set scores.

#### Video-Level (Secondary)
- Group frame predictions by `video_id`.
- **Video score aggregation methods** (compare on validation set, select best for test):
  - **Mean probability**: Average predicted $P(\text{fake})$ across all frames of the video.
  - **Mean logit**: Average pre-sigmoid logits, then apply sigmoid.
  - **Top-$k$ mean**: Average the top $k$ highest $P(\text{fake})$ scores (e.g., $k=3$ out of 8 frames — focuses on the most suspicious frames).
- Apply the frozen video-level threshold $\tau_{\text{video}}$ to the aggregated video score $\to$ binary video prediction.
- Report: video-level Accuracy, AUC-ROC, EER, F1.
- **Note**: Video-level AUC is typically $\ge$ frame-level AUC in literature, as multi-frame aggregation reduces variance and single-frame false positives.

#### Per-Source (Diagnostic)
- Report frame-level accuracy, precision, and recall separately for:
  - Celeb-real (label 0)
  - YouTube-real (label 0)
  - Celeb-synthesis (label 1)
- Identifies if the model overfits to one data source or develops class-specific shortcuts.

### 5.2 Visualization Script
**Script:** `src/plot.py` (refactored)

**Generates:**
1. **Confusion Matrix** (frame-level)
2. **ROC Curve** (frame + video overlaid)
3. **Score Distribution** (histogram: real vs fake, overlapping)
4. **Per-Video Score Distribution** (histogram of video-aggregated scores)
5. **Calibration Plot** (predicted confidence vs empirical accuracy)

**Output:** `outputs/plots_test/` or `outputs/plots_val/`

---

### 5.3 Benchmark Script (New)
**Script:** `src/benchmark_model.py` (new)

**Purpose:** Measure inference speed and memory usage across model variants and export formats.

**Formats & Hardware Targets:**
- **CPU Primary**: ONNX FP32 (default production deployment for CPU environments)
- **GPU Option**: ONNX FP16 (CUDA execution provider)
- **INT8 Static Quantization**: ONNX INT8 (QDQ format, per-channel weights). Quantized using 500–1000 calibration crops from the training set. *Note*: Depthwise separable convolutions and SiLU activations in EfficientNet are known to be sensitive to INT8 quantization; keep FP32 as the reliable fallback.

**Benchmark Logging:**
- System details: CPU model, physical/logical thread count, GPU model (if applicable), CUDA version, ONNX Runtime version.
- Test parameters: Warmup iterations (100 runs), benchmark iterations (1000 runs), batch sizes (1 and 32).
- Metrics: Latency mean/p50/p95/p99 (ms), throughput (images/sec), peak memory (MB), model file size (MB).

**Output:** `benchmarks/results.csv`

---

### 5.4 ONNX Export Script (New)
**Script:** `src/export_onnx.py` (new)

**Method:**
```python
# 1. FP32 Export (Opset 17)
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "checkpoints/best_classifier.onnx",
    input_names=["image"], output_names=["logit"],
    dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
    opset_version=17
)

# 2. FP16 Conversion (GPU target only)
from onnxmltools.utils import float16_converter
model_fp16 = float16_converter.convert_float_to_float16(onnx.load("checkpoints/best_classifier.onnx"))
onnx.save(model_fp16, "checkpoints/best_classifier_fp16.onnx")

# 3. INT8 Static Quantization (QDQ format with calibration)
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
# Use CalibrationDataReader with 500-1000 training face crops
quantize_static(
    "checkpoints/best_classifier.onnx",
    "checkpoints/best_classifier_int8.onnx",
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8
)
```

**Parity Checks & Acceptance Criteria:**
- **FP32 Parity**: Max absolute difference between PyTorch and ONNX FP32 logits across 100 validation samples must be $\le 1 \times 10^{-3}$.
- **FP16 & INT8 Parity**: Judged by validation performance — validation AUC drop must be $\le 0.005$ compared to the PyTorch FP32 model. If INT8 AUC drops $>0.005$, fall back to FP32.

---

## 6. Ablation Studies

**Goal:** Quantify the contribution of each design decision against the honest Tier 1 baseline.

### 6.1 Core Experiments (Priority)

These four experiments form the primary research progression:

| Code | Model Description | Fake Sources | Head | Params | CPU ms/frame | Val Frame AUC | Val Video AUC | Test Video AUC | Status |
|------|-------------------|--------------|------|--------|--------------|---------------|---------------|----------------|--------|
| **E1** | Supervised EfficientNet-B0 (Stage 1 baseline) | Celeb-DF fakes only | Sigmoid | 5.3M | *Measure* | *Measure* | *Measure* | *Measure* | Primary Baseline |
| **E2** | + U-Net GAN Adversarial Fakes (Stage 2) | Celeb-DF + U-Net | Sigmoid | 5.3M (D) | *Measure* | *Measure* | *Measure* | *Measure* | Core Study |
| **E3** | + StyleGAN2-ADA Diverse Fakes (Stage 3) | Celeb-DF + U-Net + StyleGAN | Sigmoid | 5.3M (D) | *Measure* | *Measure* | *Measure* | *Measure* | Core Study |
| **E4** | + EM Gaussian Mixture Head (Stage 4) | Celeb-DF + U-Net + StyleGAN | EM Mixture | 5.3M | *Measure* | *Measure* | *Measure* | *Measure* | Core Study |

### 6.2 Extended Experiments (Optional, Gated on Core Results)

These ablations explore secondary architecture and optimization choices, run only if compute and time permit after Core results are established:

| Code | Experiment Description | Purpose | Gating Condition | Status |
|------|------------------------|---------|------------------|--------|
| **X1** | EfficientNet-B4 Backbone (Supervised) | Compare capacity vs B0; potential teacher model | Run after E1 complete | Optional |
| **X2** | Video Score Aggregation: Mean Prob vs Mean Logit vs Top-k | Determine optimal temporal aggregation on val | Run on E1 predictions | Optional |
| **X3** | EM Mixture: $k=1$ (closed-form) vs BIC-selected $k$ | Quantify true EM benefit over Gaussian baseline | Run after E4 complete | Optional |
| **X4** | EM Mixture: PCA (64-d) vs Full 1280-d | Evaluate dimensionality reduction impact | Run after E4 complete | Optional |
| **X5** | Out-of-Fold GMM Fitting (5-Fold CV) | Check for training set overfitting in EM head | Run after E4 complete | Optional |
| **X6** | EfficientNet-B7 Ceiling | Explore maximum capacity ceiling | Run only if B4 shows significant gain over B0 | Optional |

*Note: Legacy full-frame B4 experiments without face cropping (A1–A5) are dropped because no valid legacy checkpoint exists locally and full-frame training is not an intended architecture.*

---

## 7. Inference Pipeline (What Ships)

### 7.1 Shipped Components (Minimal)

| Component | File | Precision | Target Environment | Size |
|-----------|------|-----------|-------------------|------|
| **Discriminator (Default)** | `best_classifier.onnx` | FP32 | CPU production deployment | *To be measured* |
| Discriminator (Optional GPU) | `best_classifier_fp16.onnx` | FP16 | CUDA GPU environments | *To be measured* |
| Discriminator (Optional Quantized) | `best_classifier_int8.onnx` | INT8 (QDQ) | Low-memory edge (if parity passes) | *To be measured* |
| Face Detector Weights | `mtcnn_weights.pt` | FP32 | Face localization | ~2 MB |
| Inference Config | `inference_config.json` | N/A | Normalization params, frozen threshold $\tau$ | <1 KB |
| EM Mixture (Conditional) | `em_mixture.pkl` | N/A | Only if shipped per section 4.3 rule | *To be measured* |

**NOT shipped:**
- Generator weights (U-Net, StyleGAN2) — training augmentation only.
- Raw training/val/test CSVs and frame datasets.
- Training and ablation scripts.

### 7.2 Inference API

**Script:** `src/predict.py` (refactored)

**Function:**
```python
def predict_image(image_path, model_path, method="sigmoid"):
    """
    Args:
        image_path: path to image file (JPG/PNG)
        model_path: path to .onnx checkpoint
        method: "sigmoid" (default) or "em"
    
    Returns:
        {
            "status": "success" | "no_face",
            "p_fake": float (probability of fake in [0,1]) or None,
            "label": "REAL" | "FAKE" or None,
            "confidence": float (max(p_fake, 1 - p_fake)) or None,
            "threshold_used": float (frozen threshold from val),
            "face_detected": bool,
            "face_box": [x1, y1, x2, y2] or None,
            "inference_ms": float
        }
    """
    # 1. Load image via PIL
    # 2. Detect face with MTCNN:
    #    - If NO face detected: return {"status": "no_face", "face_detected": False, ...}
    #      Do NOT attempt classification. Do NOT use center-crop fallback.
    #    - If face detected: crop with 1.3x margin, record face_box coordinates
    # 3. Resize face crop to 224x224 (bicubic)
    # 4. Normalize with ImageNet mean/std
    # 5. Run ONNX inference (FP32 default)
    # 6. Compute p_fake: sigmoid(logit) or calibrated EM LLR
    # 7. Apply frozen threshold tau_frame from val:
    #    label = "FAKE" if p_fake >= tau_frame else "REAL"
    # 8. Compute confidence = max(p_fake, 1 - p_fake)
    # 9. Return structured dictionary
```

**Video inference:**
```python
def predict_video(video_path, model_path, method="sigmoid", n_frames=8):
    """
    Extract n_frames uniformly, run MTCNN + model on each detected face.
    Aggregate scores using the best method from validation (mean prob / mean logit / top-k).
    Apply frozen video threshold tau_video.
    Returns overall video prediction + per-frame breakdown.
    """
```

### 7.3 Flask Integration (app.py)

**Modify route:**
```python
@app.route("/detect", methods=["POST"])
def detect():
    file = request.files['media']
    filepath = save_to_uploads(file)
    
    # Call predict API (default: FP32 ONNX)
    result = predict_image(filepath, model_path="checkpoints/best_classifier.onnx")
    
    if result["status"] == "no_face":
        flash("No face detected in the uploaded media. Please upload a clear face image or video.", "warning")
        return render_template("detect.html")
    
    # Render results.html with:
    # - Label (REAL / FAKE)
    # - Confidence percentage
    # - Face crop preview + bounding box
    # - Threshold used and raw score
    # - Inference latency (ms)
    return render_template("results.html", **result)
```

---

## 8. File Changes Summary

### New Files (13)

| File | Purpose | Lines | Stage |
|------|---------|-------|-------|
| `src/preprocess_faces.py` | MTCNN face detection + cropping | ~150 | Preprocessing |
| `src/make_splits.py` | Identity-disjoint CSV split generation | ~200 | Preprocessing |
| `src/train_classifier.py` | Supervised baseline (Stage 1) | ~300 | Training |
| `src/train_gan_hybrid.py` | GAN-augmented training (Stage 2-3) | ~400 | Training |
| `src/train_em_mixture.py` | Fit GMM on embeddings | ~100 | Post-training |
| `src/export_onnx.py` | ONNX export (FP32/FP16/INT8) | ~100 | Export |
| `src/benchmark_model.py` | Speed/memory benchmarks | ~200 | Evaluation |
| `notebooks/04_supervised_training.ipynb` | Orchestrate training pipeline | ~50 cells | Training |
| `notebooks/05_ablation_studies.ipynb` | Run ablation experiments | ~30 cells | Evaluation |
| `data/train.csv` | Training split metadata (identity-disjoint) | N/A | Data |
| `data/val.csv` | Validation split metadata (identity-disjoint) | N/A | Data |
| `data/test.csv` | Test split metadata (identity-disjoint) | N/A | Data |
| `data/test_official.csv` | Official Celeb-DF-v2 test split metadata | N/A | Data |

### Modified Files (8)

| File | Changes | Priority |
|------|---------|----------|
| `src/config.py` | Replace `REAL_DIR`/`FAKE_DIR` with CSV paths. Add `FRAMES_DIR`, `IMG_SIZE=224`, `pos_weight`, `BACKBONE="efficientnet_b0"`. Remove GAN params or move to `CFG_GAN`. | High |
| `src/dataset.py` | Replace glob logic with CSV loading. Add `video_id` to return. Fix normalization to ImageNet params. | High |
| `src/discriminator.py` | Add `backbone` argument to constructor. Add `get_embeddings()` method (return pre-logit features). Update docstring (0=real, 1=fake). | High |
| `src/evaluate.py` | Add `split` and `method` args. Load CSV. Fix AUC bug. Add video-level and per-source metrics. | High |
| `src/plot.py` | Load CSV. Add calibration plot. Add video-score histogram. | Medium |
| `src/predict.py` | Add MTCNN face crop. Fix normalization. Flip label semantics (>=0.5 → FAKE). Add EM method. Return dict. | High |
| `extract_frames.py` | Generalize to handle all three Celeb-DF folders (real, yt, synth). | Medium |
| `app.py` | Integrate `predict_image` in `/detect` route. Render results. | Low (post-training) |

### Unchanged Files (4)

| File | Reason |
|------|--------|
| `src/generator.py` | Used in training only, no API changes needed |
| `src/losses.py` | GAN losses kept for Stage 2-3, no changes |
| `src/train.py` | Old GAN training script — keep for reference, not used in new pipeline |
| `tests/smoke_test.py` | Update imports later, not blocking |

---

## 9. Execution Checklist & Stage Gating

### Phase 1: Preprocessing & Data Splits
- [ ] Run `extract_frames.py` on all Celeb-DF-v2 folders → `data/frames/`
- [ ] Implement and run `src/preprocess_faces.py` → `data/frames_cropped/` + `face_crop_manifest.json`
- [ ] Implement and run `src/make_splits.py` → `data/train.csv`, `val.csv`, `test.csv`, `test_official.csv`
- [ ] Generate identity-overlap report for `test_official.csv` vs `train.csv`
- [ ] **Gate 1 Check**: Verify no Celeb identity appears in >1 split; verify $\ge 30$ real and $\ge 30$ fake videos in both val and test; verify all CSV frame paths exist in `data/frames_cropped/`.

### Phase 2: Supervised Baseline (Stage 1)
- [ ] Update `src/config.py` (paths, ImageNet norm, `BACKBONE="efficientnet_b0"`, `IMG_SIZE=224`)
- [ ] Update `src/dataset.py` (CSV loading, ImageNet normalization, return `video_id`)
- [ ] Update `src/discriminator.py` (configurable backbone, pre-logit embeddings hook)
- [ ] Implement `src/train_classifier.py` (supervised BCE with `pos_weight`, val AUC early stopping, AMP)
- [ ] Train Stage 1 model on Colab (checkpoint to Google Drive)
- [ ] Update `src/evaluate.py` (continuous-score AUC, EER, threshold selection on val, video aggregation)
- [ ] Evaluate Stage 1 on `val.csv`, `test.csv`, and `test_official.csv` $\to$ **establish Tier 1 baseline**
- [ ] **Gate 2 (Go / No-Go for Stage 2)**: Proceed to Stage 2 **ONLY IF**:
  - Stage 1 training converged stably (val AUC $\ge 0.85$, training loss decreased monotonically).
  - Validation metrics show no obvious pathology (both real specificity and fake recall $>75\%$).
  - Tier 1 baseline is fully documented in `PLAN.md`.

### Phase 3: Adversarial GAN Augmentation (Stage 2 & 3)
- [ ] Implement `src/train_gan_hybrid.py` (alternating D/G steps, TTUR learning rates, frozen-D handling)
- [ ] Warm up U-Net Generator on real training frames ($N=5$ epochs L1)
- [ ] Train Stage 2 (D + U-Net fakes) with monitoring of D detection rate and per-source val metrics
- [ ] Evaluate Stage 2 on `val.csv` and `test.csv`
- [ ] **Gate 3 (Go / No-Go for Stage 3)**: Proceed to Stage 3 **ONLY IF**:
  - Stage 2 did not regress on `yt_real` validation specificity by $>3.0$ points vs Stage 1.
  - Stage 2 improved or maintained video-level validation AUC vs Stage 1.
- [ ] (Optional) Offline StyleGAN2-ADA sample generation (10k fakes + matched real FFHQ crops)
- [ ] Run nuisance-feature probe (verify accuracy $\approx 50\%$ before training)
- [ ] Train Stage 3 (D + U-Net + StyleGAN2 fakes)
- [ ] Verify held-out FFHQ validation specificity $\ge$ Stage 2 `yt_real` specificity

### Phase 4: EM Mixture Head (Stage 4)
- [ ] Implement `src/train_em_mixture.py` (extract train embeddings, standardize, PCA, fit GMM by BIC)
- [ ] Fit per-class GMM ($k=1$ baseline and BIC-selected $k \in \{1,2,3,4,6,8\}$)
- [ ] Fit calibration (Platt / temperature scaling) on validation set
- [ ] Evaluate EM head vs sigmoid head on `val.csv` and `test.csv` (AUC, EER, ECE, reliability diagrams)
- [ ] Run out-of-fold GMM ablation and OOD evaluation
- [ ] **Gate 4 (Shipping Decision)**: EM head is set as default in inference config **ONLY IF** validation ECE improves without AUC drop ($\Delta \text{AUC} \le 0.005$).

### Phase 5: Export & Benchmarking
- [ ] Implement `src/export_onnx.py` (FP32 opset 17, optional FP16 GPU, static INT8 QDQ with calibration)
- [ ] Validate FP32 parity (max abs diff $\le 1 \times 10^{-3}$ vs PyTorch)
- [ ] Validate INT8 parity (val AUC drop $\le 0.005$; if fails, keep FP32 default)
- [ ] Implement `src/benchmark_model.py` (measure latency, throughput, memory, size on target CPU)
- [ ] Update `src/plot.py` (generate all diagnostic and publication plots)
- [ ] Populate Ablation Table (Core E1–E4 + completed extended runs)

### Phase 6: Flask Application Integration
- [ ] Update `src/predict.py` (structured dict return, MTCNN crop, no-face guard, threshold application)
- [ ] Integrate into `app.py` (`/detect` route with face detection validation and rich results view)
- [ ] End-to-end testing: upload real image, fake image, non-face image, short video clip
- [ ] Final documentation and repository cleanup

---

## 10. Expected Outcomes & Performance Targets

### 10.1 Two-Tier Performance Target Framework

Rather than prescribing arbitrary numeric targets before running experiments on the leak-free split, we use a **two-tier evaluation rule**:

- **Tier 1: Measured Stage 1 Baseline**:
  - Train the Stage 1 supervised EfficientNet-B0 model on `train.csv`.
  - Evaluate on `test.csv` (identity-disjoint) and `test_official.csv` (official list).
  - Record the actual measured metrics: Frame-level AUC, Video-level AUC, EER, F1, and per-source accuracies.
  - This establishes the empirical baseline against which all subsequent stages and ablations are measured.

- **Tier 2: Target Criteria (Set After Stage 1 Baseline Exists)**:
  - Once Tier 1 baseline is measured, targets for Stages 2, 3, and EM head will be defined relative to the baseline:
    - **Stage 2 (+ U-Net GAN)**: Must improve video-level AUC over Stage 1 without dropping `yt_real` specificity by $>3.0$ points.
    - **Stage 3 (+ StyleGAN2)**: Must improve out-of-distribution detection without regressing on held-out FFHQ reals.
    - **EM Mixture Head**: Must improve ECE (calibration) and match or exceed sigmoid head AUC.
  - **Cross-dataset evaluation**: If tested on external datasets (e.g., FaceForensics++, DFDC), cross-dataset AUC will be reported separately. It is expected to be substantially lower than in-dataset AUC due to domain shift in compression and synthesis methods.

### 10.2 Literature Comparison (TODO)

> **TODO**: Fill this table ONLY with verified numbers cited directly from original source papers evaluated under the same test protocol. Every entry must state whether the test split is identity-disjoint or uses the official Celeb-DF-v2 test list.

| Paper / Model | Test Protocol | Identity-Disjoint? | Frame AUC | Video AUC | EER | Reference / Source |
|---------------|---------------|-------------------|-----------|-----------|-----|-------------------|
| *To be populated from verified sources* | | | | | | |

---

## 11. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Face detection fails on >10% of frames** | Log failures, manual review. Consider RetinaFace as fallback. Document drop rate in split manifest. |
| **Fake-video identity mapping unverified** | Mark as UNVERIFIED in PLAN.md until Celeb-DF-v2 README/List files confirm whether `id<A>_id<B>_<clip>` means source=A, target=B or vice versa. Treat both IDs as belonging to the video (conservative exclusion rule) until verified. |
| **Official test set overlaps with train** | Generate identity-overlap report in `make_splits.py`. If overlap $>10\%$ of test identities, document it and report metrics on both test sets. Literature comparison must explicitly state the split method used. |
| **Class imbalance degrades precision** | Use `pos_weight` in BCE loss. Monitor per-class precision/recall on validation. If fake recall $<70\%$, consider focal loss or per-source reweighting. |
| **U-Net fakes too easy for D** | Warm-up G for 5 epochs before adversarial training. Monitor D detection rate; if $>95\%$ after 10 epochs, lower $w_G$ weight or add JPEG/blur augmentation to G outputs. |
| **StyleGAN2-ADA CUDA compilation fails on Colab** | Use pure-PyTorch fallback (`torch.nn.functional` equivalents for `upfirdn2d`, `bias_act`). If still fails, pregenerate all 10k samples offline on a local machine with working CUDA and upload to Drive. |
| **StyleGAN alignment shortcut** | Run nuisance-feature probe before training; accuracy $>70\%$ means pipeline mismatch (fix preprocessing). Add matched real FFHQ crops (label 0) and cap at $\le 25\%$ of real pool. Validate held-out FFHQ specificity post-training. |
| **INT8 quantization accuracy loss** | Use static QDQ quantization with 500–1000 calibration samples. If val AUC drops $>0.005$, depthwise convs and SiLU are the likely culprits — fall back to FP32 default and mark INT8 as experimental. |
| **EM mixture doesn't improve over sigmoid** | Ship sigmoid head as default. Keep EM as an optional research variant in the checkpoint. Document ECE and AUC comparison in ablation table. |
| **Colab session timeout during long training** | Checkpoint to Google Drive every epoch (`latest_classifier.pth` with full state). Resume from `latest_classifier.pth` on reconnect. Enable Colab Pro if free tier timeouts become frequent. |
| **FaceForensics++ access delayed (OOD evaluation)** | FF++ requires a request form and may take 1–2 weeks for approval. Apply early. If access is delayed, use a small public subset (e.g., FaceForensics-lite on Kaggle) for preliminary OOD check and document the limitation. |
| **Model overfits to Celeb-DF synthesis method** | Evaluate on external datasets (FF++, DFDC) and report cross-dataset AUC separately with the expectation of substantial drop. Add per-source metrics to identify if specificity collapses on YouTube-real. |

---

## 12. Future Work (Out of Scope)

- **Temporal model** (VideoDiscriminator with LSTM) — requires video-level training
- **Attention visualization** (GradCAM) — show which regions trigger detection
- **Cross-dataset evaluation** — FaceForensics++, DFDC, DeeperForensics
- **Adversarial robustness** — test against JPEG compression, blur, noise
- **Explainability** — generate textual explanations ("detected GAN fingerprints in frequency domain")
- **Multi-modal** — audio + video fusion for deepfake detection

---

## 13. Success Criteria

### Tier 1: Minimum Viable Product (MVP)
The baseline must establish a reproducible, honest experimental foundation:
- [ ] Identity-disjoint splits implemented (no Celeb identity appears in >1 split)
- [ ] MTCNN face cropping applied to all training/val/test data
- [ ] Stage 1 supervised baseline trained with proper ImageNet normalization and class weighting
- [ ] **Tier 1 Baseline Metrics Measured**: Frame-level AUC, Video-level AUC, EER, and per-source metrics recorded on both `test.csv` (identity-disjoint) and `test_official.csv` (official Celeb-DF-v2 list)
- [ ] Flask `/detect` route functional with ONNX inference and no-face validation

### Tier 2: Performance Targets (Set After Tier 1 Exists)
Quantitative goals will be defined relative to the measured Tier 1 baseline:
- Stage 2 (U-Net GAN) must improve video-level validation AUC without dropping `yt_real` specificity by $>3$ pts
- Stage 3 (StyleGAN2) must improve OOD detection without regressing on held-out FFHQ reals
- EM mixture head ships only if it improves validation ECE without AUC loss ($\Delta \text{AUC} \le 0.005$)
- Cross-dataset evaluation (if conducted) will be reported separately with the expectation that cross-dataset AUC is substantially lower than in-dataset AUC due to domain shift

### Non-Numeric Quality Gates (All Phases)
- All training runs log git commit hash, config, and library versions
- All splits pass integrity checks (no identity leakage, minimum video counts met)
- All ablation results populate the Core Experiments table (E1–E4)
- All ONNX exports pass parity checks before deployment

---

**End of Plan**  
**Next step:** Begin Phase 1 (Preprocessing) when authorized to write code.
