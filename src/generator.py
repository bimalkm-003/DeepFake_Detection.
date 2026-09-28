import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_bn=True, dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False) if down
            else nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c) if use_bn else nn.Identity(),
            nn.Dropout2d(0.5) if dropout else nn.Identity(),
            nn.LeakyReLU(0.2) if down else nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    """
    U-Net Generator
    Input:  (B, 3, 256, 256)
    Output: (B, 3, 256, 256)
    """
    def __init__(self, in_c=3, features=64):
        super().__init__()
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(in_c, features, 4, 2, 1), nn.LeakyReLU(0.2))
        self.e2 = ConvBlock(features,     features * 2)
        self.e3 = ConvBlock(features * 2, features * 4)
        self.e4 = ConvBlock(features * 4, features * 8)
        self.e5 = ConvBlock(features * 8, features * 8)
        self.e6 = ConvBlock(features * 8, features * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )

        # Decoder with skip connections
        self.d1 = ConvBlock(features * 8,     features * 8, down=False, dropout=True)
        self.d2 = ConvBlock(features * 8 * 2, features * 8, down=False, dropout=True)
        self.d3 = ConvBlock(features * 8 * 2, features * 8, down=False, dropout=True)
        self.d4 = ConvBlock(features * 8 * 2, features * 4, down=False)
        self.d5 = ConvBlock(features * 4 * 2, features * 2, down=False)
        self.d6 = ConvBlock(features * 2 * 2, features,     down=False)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_c, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        s5 = self.e5(s4)
        s6 = self.e6(s5)
        bn = self.bottleneck(s6)
        d = self.d1(bn)
        d = self.d2(torch.cat([F.interpolate(d, s6.shape[2:]), s6], dim=1))
        d = self.d3(torch.cat([F.interpolate(d, s5.shape[2:]), s5], dim=1))
        d = self.d4(torch.cat([F.interpolate(d, s4.shape[2:]), s4], dim=1))
        d = self.d5(torch.cat([F.interpolate(d, s3.shape[2:]), s3], dim=1))
        d = self.d6(torch.cat([F.interpolate(d, s2.shape[2:]), s2], dim=1))
        return self.out(torch.cat([F.interpolate(d, s1.shape[2:]), s1], dim=1))