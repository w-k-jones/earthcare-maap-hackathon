# models/unetskip.py
# ---------------------------------------------------------------------
# File overview
# UNet with skip connections used for downscaling experiments.
# ---------------------------------------------------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    """
    Build a 2-layer convolutional block.

    Inputs:
        in_ch: input channels.
        out_ch: output channels.

    Outputs:
        nn.Sequential block with Conv-ReLU-Conv-ReLU.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNetSkip(nn.Module):
    def __init__(self, in_channels, out_channels=1, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Encoder
        self.enc1 = conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_channels * 2, base_channels)

        # Final 1x1 conv
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _match_spatial_size(x, reference):
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # [B,32,H,W]
        p1 = self.pool1(e1)        # [B,32,H/2,W/2]
        e2 = self.enc2(p1)         # [B,64,H/2,W/2]
        p2 = self.pool2(e2)        # [B,64,H/4,W/4]
        e3 = self.enc3(p2)         # [B,128,H/4,W/4]
        p3 = self.pool3(e3)        # [B,128,H/8,W/8]

        # Bottleneck
        b = self.bottleneck(p3)    # [B,256,H/8,W/8]

        # Decoder with skip connections
        u3 = self.up3(b)           # [B,128,H/4,W/4]
        u3 = self._match_spatial_size(u3, e3)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)         # [B,128,H/4,W/4]

        u2 = self.up2(d3)          # [B,64,H/2,W/2]
        u2 = self._match_spatial_size(u2, e2)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)         # [B,64,H/2,W/2]

        u1 = self.up1(d2)          # [B,32,H,W]
        u1 = self._match_spatial_size(u1, e1)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)         # [B,32,H,W]

        # Output
        return self.out_conv(d1)   # [B,out_channels,H,W]  
