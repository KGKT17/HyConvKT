import torch
import torch.nn as nn


def manual_circular_pad_3d(x, pad):
    """3D circular padding for convolution."""
    pad_w, _, pad_h, _, pad_d, _ = pad

    if pad_d > 0:
        front = x[:, :, -pad_d:, :, :]
        back = x[:, :, :pad_d, :, :]
        x = torch.cat([front, x, back], dim=2)

    if pad_h > 0:
        top = x[:, :, :, -pad_h:, :]
        bottom = x[:, :, :, :pad_h, :]
        x = torch.cat([top, x, bottom], dim=3)

    if pad_w > 0:
        left = x[:, :, :, :, -pad_w:]
        right = x[:, :, :, :, :pad_w]
        x = torch.cat([left, x, right], dim=4)

    return x


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation Block for 3D feature maps."""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class CircularConv3d(nn.Module):
    """3D convolution with circular padding and batch normalization."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
        k_d, k_h, k_w = kernel_size
        self.pad_d = (k_d - 1) // 2
        self.pad_h = (k_h - 1) // 2
        self.pad_w = (k_w - 1) // 2
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        pad = (self.pad_w, self.pad_w, self.pad_h, self.pad_h, self.pad_d, self.pad_d)
        x_padded = manual_circular_pad_3d(x, pad)
        return self.bn(self.conv(x_padded))
