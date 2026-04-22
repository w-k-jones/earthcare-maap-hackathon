# models/unet.py
# ---------------------------------------------------------------------
# File overview
# Standard UNet implementation for downscaling experiments.
# ---------------------------------------------------------------------
import torch.nn as nn
import torch

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


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # Encoder
        self.c1 = conv_block(in_channels, 32)
        self.p1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.c2 = conv_block(32, 64)
        self.p2 = nn.MaxPool2d(kernel_size=(2, 1))
        # Bottleneck
        self.b = conv_block(64, 128)
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1))
        self.d2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32,kernel_size=(2, 1), stride=(2, 1))
        self.d1 = conv_block(64, 32)
        # Final
        self.head = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        b = self.b(p2)
        u2 = self.up2(b)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.d2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.d1(u1)
        
        z = d1.mean(dim=2)        
        out = self.head(z)        

        return out

