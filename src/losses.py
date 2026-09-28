import torch
import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss()
l1_loss  = nn.L1Loss()


def discriminator_loss(real_preds, fake_preds):
    real_labels = torch.ones_like(real_preds)
    fake_labels = torch.zeros_like(fake_preds)
    loss = (bce_loss(real_preds, real_labels) + bce_loss(fake_preds, fake_labels)) / 2
    return loss


def generator_loss(fake_preds, fake_imgs, real_imgs, lambda_l1=10.0):
    adv_loss = bce_loss(fake_preds, torch.ones_like(fake_preds))
    rec_loss = l1_loss(fake_imgs, real_imgs)
    return adv_loss + lambda_l1 * rec_loss