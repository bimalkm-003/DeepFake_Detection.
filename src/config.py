import torch

class CFG:
    REAL_DIR          = "/content/frames/real"
    FAKE_DIR          = "/content/frames/fake"
    CKPT_DIR          = "/content/drive/MyDrive/deepfake_checkpoints"
    IMG_SIZE          = 256
    BATCH_SIZE        = 8
    NUM_EPOCHS        = 15
    LR_G              = 5e-4
    LR_D              = 2e-4
    BETA1             = 0.5
    BETA2             = 0.999
    LAMBDA_L1         = 10.0
    SAVE_EVERY        = 5
    MAX_SAMPLES       = 3000
    FRAMES_PER_CLIP   = 8
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
    SEED              = 42
