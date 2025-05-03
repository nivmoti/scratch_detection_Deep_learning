import torch
import torch.nn.functional as F
import torch.nn as nn
from Model.residual import *
from Model.cbam import *


class ContractingPath(nn.Module):
    def __init__(self):
        super(ContractingPath, self).__init__()
        self.enc1 = ResidualBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        x4 = self.enc4(p3)
        p4 = self.pool4(x4)

        return x1, x2, x3, x4, p4
class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.residual = ResidualBlock(512, 512)

    def forward(self, x):
        return self.residual(x)
class ExpandingPath(nn.Module):
    def __init__(self):
        super(ExpandingPath, self).__init__()

        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.att4 = ResidualAttentionBlock(512)
        self.dec4 = ResidualBlock(1024, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.att3 = ResidualAttentionBlock(256)
        self.dec3 = ResidualBlock(512, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.att2 = ResidualAttentionBlock(128)
        self.dec2 = ResidualBlock(256, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.att1 = ResidualAttentionBlock(64)
        self.dec1 = ResidualBlock(128, 64)

    def forward(self, x, skips):
        x4, x3, x2, x1 = skips

        u4 = self.upconv4(x)
        x4 = self.att4(x4)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.dec4(u4)

        u3 = self.upconv3(u4)
        x3 = self.att3(x3)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.upconv2(u3)
        x2 = self.att2(x2)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.upconv1(u2)
        x1 = self.att1(x1)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec1(u1)

        return u1
class ImprovedUNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ImprovedUNet, self).__init__()
        self.contracting = ContractingPath()
        self.bottleneck = Bottleneck()
        self.expanding = ExpandingPath()
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4, p4 = self.contracting(x)
        b = self.bottleneck(p4)
        u = self.expanding(b, [x4, x3, x2, x1])
        out = self.final_conv(u)
        return out


