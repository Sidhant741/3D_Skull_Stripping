import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """
    Double 3x3x3 conv block for the encoder path.
    Returns (pooled_output, pre_pool_residual).
    """
    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super().__init__()
        self.conv1   = nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm3d(out_channels // 2)
        self.conv2   = nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2     = nn.BatchNorm3d(out_channels)
        self.relu    = nn.ReLU(inplace=True)
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = self.pooling(res) if not self.bottleneck else res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    Upsample + double 3x3x3 conv block for the decoder path.
    """
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super().__init__()
        assert (not last_layer and num_classes is None) or (last_layer and num_classes is not None), \
            'last_layer=True requires num_classes to be set'

        self.upconv1 = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.relu    = nn.ReLU(inplace=True)
        self.bn1     = nn.BatchNorm3d(in_channels // 2)
        self.bn2     = nn.BatchNorm3d(in_channels // 2)
        self.conv1   = nn.Conv3d(in_channels + res_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2   = nn.Conv3d(in_channels // 2,           in_channels // 2, kernel_size=3, padding=1)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, x, residual=None):
        out = self.upconv1(x)
        if residual is not None:
            out = F.interpolate(out, size=residual.shape[2:], mode='trilinear', align_corners=False)
            out = torch.cat((out, residual), dim=1)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        if self.last_layer:
            out = self.conv3(out)
        return out


class UNet3D(nn.Module):
    """Standard 3-level 3D U-Net."""
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super().__init__()
        l1, l2, l3 = level_channels
        self.a_block1   = Conv3DBlock(in_channels, l1)
        self.a_block2   = Conv3DBlock(l1, l2)
        self.a_block3   = Conv3DBlock(l2, l3)
        self.bottleNeck = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)
        self.s_block3   = UpConv3DBlock(bottleneck_channel, res_channels=l3)
        self.s_block2   = UpConv3DBlock(l3, res_channels=l2)
        self.s_block1   = UpConv3DBlock(l2, res_channels=l1, num_classes=num_classes, last_layer=True)

    def forward(self, x):
        x, r1 = self.a_block1(x)
        x, r2 = self.a_block2(x)
        x, r3 = self.a_block3(x)
        x, _  = self.bottleNeck(x)
        x = self.s_block3(x, r3)
        x = self.s_block2(x, r2)
        x = self.s_block1(x, r1)
        return x


class UNet3DDeep(nn.Module):
    """
    4-level 3D U-Net with deep supervision.

    During training  → returns (main_logits, aux_logits)
        main_logits : full-resolution output  (B, num_classes, D, H, W)
        aux_logits  : half-resolution output  (B, num_classes, D/2, H/2, W/2)
    During eval/inference → returns main_logits only.
    """
    def __init__(self, in_channels, num_classes,
                 level_channels=[32, 64, 128, 256],
                 bottleneck_channel=512) -> None:
        super().__init__()
        assert len(level_channels) == 4, "level_channels must have 4 entries for UNet3DDeep"
        l1, l2, l3, l4 = level_channels

        # Encoder
        self.a_block1   = Conv3DBlock(in_channels, l1)
        self.a_block2   = Conv3DBlock(l1, l2)
        self.a_block3   = Conv3DBlock(l2, l3)
        self.a_block4   = Conv3DBlock(l3, l4)
        self.bottleNeck = Conv3DBlock(l4, bottleneck_channel, bottleneck=True)

        # Decoder
        self.s_block4 = UpConv3DBlock(bottleneck_channel, res_channels=l4)
        self.s_block3 = UpConv3DBlock(l4, res_channels=l3)
        self.s_block2 = UpConv3DBlock(l3, res_channels=l2)
        self.s_block1 = UpConv3DBlock(l2, res_channels=l1, num_classes=num_classes, last_layer=True)

        # Auxiliary head for deep supervision (attached after s_block2)
        # s_block2 outputs l3//2 = l3 channels halved by UpConv — but
        # UpConv3DBlock halves in_channels, so output channels = l3 // 2
        self.deep_head = nn.Conv3d(l3 // 2, num_classes, kernel_size=1)

    def forward(self, x):
        x, r1 = self.a_block1(x)
        x, r2 = self.a_block2(x)
        x, r3 = self.a_block3(x)
        x, r4 = self.a_block4(x)
        x, _  = self.bottleNeck(x)

        x  = self.s_block4(x, r4)
        x  = self.s_block3(x, r3)
        d2 = self.s_block2(x, r2)          # intermediate feature map
        d1 = self.s_block1(d2, r1)         # final full-res logits

        if self.training:
            # Apply the auxiliary 1x1 head here — keeps forward() self-contained
            aux = self.deep_head(d2)        # half-res logits
            return d1, aux

        return d1


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3DDeep(in_channels=1, num_classes=1,
                        level_channels=[32, 64, 128, 256],
                        bottleneck_channel=512).to(device)

    x = torch.randn(1, 1, 128, 128, 128).to(device)

    model.train()
    d1, aux = model(x)
    print(f"Train — main: {d1.shape}, aux: {aux.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Eval  — output: {out.shape}")