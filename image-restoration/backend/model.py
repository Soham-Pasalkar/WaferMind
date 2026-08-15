import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        y = self.pool(x)
        y = self.conv1(y)
        y = self.conv2(y)
        return x * y


class NAFBlock(nn.Module):
    def __init__(self, channels, expansion=2, dropout=0.0):
        super().__init__()
        hidden = channels * expansion

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1)
        self.conv2 = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.sg = SimpleGate()
        self.sca = SimpleChannelAttention(hidden)
        self.conv3 = nn.Conv2d(hidden, channels, 1)
        self.dropout1 = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, hidden * 2, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(hidden, channels, 1)
        self.dropout2 = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        x = x + y * self.gamma

        return x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Conv2d(channels, channels * 2, 2, stride=2)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class NAFNetSR(nn.Module):
    def __init__(
        self,
        img_channels=1,
        width=32,
        enc_blocks=(2, 2, 4),
        dec_blocks=(2, 2, 2)
    ):
        super().__init__()

        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)

        self.encoder1 = nn.Sequential(
            *[NAFBlock(width) for _ in range(enc_blocks[0])]
        )
        self.down1 = Downsample(width)

        self.encoder2 = nn.Sequential(
            *[NAFBlock(width * 2) for _ in range(enc_blocks[1])]
        )
        self.down2 = Downsample(width * 2)

        self.encoder3 = nn.Sequential(
            *[NAFBlock(width * 4) for _ in range(enc_blocks[2])]
        )
        self.down3 = Downsample(width * 4)

        self.middle = nn.Sequential(
            NAFBlock(width * 8),
            NAFBlock(width * 8)
        )

        self.up3 = Upsample(width * 8)
        self.decoder3 = nn.Sequential(
            *[NAFBlock(width * 4) for _ in range(dec_blocks[0])]
        )

        self.up2 = Upsample(width * 4)
        self.decoder2 = nn.Sequential(
            *[NAFBlock(width * 2) for _ in range(dec_blocks[1])]
        )

        self.up1 = Upsample(width * 2)
        self.decoder1 = nn.Sequential(
            *[NAFBlock(width) for _ in range(dec_blocks[2])]
        )

        self.sr = nn.Sequential(
            nn.Conv2d(width, width * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(width, img_channels, 3, padding=1)
        )

    def forward(self, x):
        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False
        )

        x = self.intro(x)

        e1 = self.encoder1(x)
        x = self.down1(e1)

        e2 = self.encoder2(x)
        x = self.down2(e2)

        e3 = self.encoder3(x)
        x = self.down3(e3)

        x = self.middle(x)

        x = self.up3(x)
        x = x + e3
        x = self.decoder3(x)

        x = self.up2(x)
        x = x + e2
        x = self.decoder2(x)

        x = self.up1(x)
        x = x + e1
        x = self.decoder1(x)

        residual = self.sr(x)
        return base + residual
