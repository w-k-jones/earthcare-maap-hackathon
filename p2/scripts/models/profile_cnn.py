import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_block(in_channels, out_channels, stride=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(5, 5),
            stride=stride,
            padding=(2, 2),
            bias=False,
        ),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.GELU(),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False,
        ),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.GELU(),
    )


class ResidualConv1d(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ProfileCNN(nn.Module):
    """
    Model for EarthCARE profile-to-track regression.

    Input:
        x: [B, C_in, height, along_track]

    Output:
        y: [B, C_out, along_track]

    The 2D encoder downsamples only the vertical height dimension. The track
    dimension is preserved, then modeled with dilated 1D convolutions.
    """

    def __init__(self, in_channels, out_channels=1, base_channels=32, nonnegative_output=False):
        super().__init__()
        self.nonnegative_output = nonnegative_output

        self.encoder = nn.Sequential(
            conv2d_block(in_channels, base_channels, stride=(1, 1)),
            conv2d_block(base_channels, base_channels * 2, stride=(2, 1)),
            conv2d_block(base_channels * 2, base_channels * 4, stride=(2, 1)),
            conv2d_block(base_channels * 4, base_channels * 4, stride=(2, 1)),
        )

        channels = base_channels * 4
        self.track_model = nn.Sequential(
            ResidualConv1d(channels, dilation=1),
            ResidualConv1d(channels, dilation=2),
            ResidualConv1d(channels, dilation=4),
            ResidualConv1d(channels, dilation=8),
        )
        self.head = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, x.shape[-1])).squeeze(2)
        x = self.track_model(x)
        x = self.head(x)
        if self.nonnegative_output:
            x = F.softplus(x)
        return x
