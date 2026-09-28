import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader

from config import CFG
from dataset import DeepfakeImageDataset, get_transforms
from discriminator import Discriminator

def generate_plots(ckpt_path, num_samples=1000):
    os.makedirs("outputs", exist_ok=True)
    device = CFG.DEVICE
    transform = get_transforms(CFG.IMG_SIZE, mode="val")
    
    # 1. Load Data (use a subset for quick evaluation)
    print(f"Loading {num_samples} samples per class for evaluation...")
    dataset = DeepfakeImageDataset(CFG.REAL_DIR, CFG.FAKE_DIR, transform, max_samples=num_samples)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    print("Loading Discriminator weights...")
    D = Discriminator(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    D.load_state_dict(ckpt["D"])
    D.eval()
    
    all_labels = []
    all_scores = []
    
    # 3. Run Inference
    print("Running inference...")
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            # Apply sigmoid because Discriminator outputs raw logits
            scores = torch.sigmoid(D(data)).squeeze(1).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            
    # 4. Plot 1: Confusion Matrix
    preds = [1 if s >= 0.5 else 0 for s in all_scores]
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 5. Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/roc_curve.png', dpi=300)
    plt.close()

    # 6. Plot 3: Score Distribution Histogram
    plt.figure(figsize=(7,5))
    sns.histplot(x=all_scores, hue=all_labels, bins=50, kde=True, 
                 palette=['red', 'blue'], element="step")
    plt.title('Discriminator Confidence Scores')
    plt.xlabel('Predicted Probability (0 = Fake, 1 = Real)')
    plt.legend(title='Actual Class', labels=['Real (1)', 'Fake (0)'])
    plt.tight_layout()
    plt.savefig('outputs/score_distribution.png', dpi=300)
    plt.close()
    
    print("\n✔ Success! Plots saved to the 'outputs' folder:")
    print("  - outputs/confusion_matrix.png")
    print("  - outputs/roc_curve.png")
    print("  - outputs/score_distribution.png")

if __name__ == "__main__":
    generate_plots(ckpt_path=CFG.CKPT_DIR + '/latest_image.pth', num_samples=1000)