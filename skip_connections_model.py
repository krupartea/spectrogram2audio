import torch
import torch.nn as nn
from hparams import *

def make_conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(7, 3), stride=1, padding=(3, 1)),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(7, 3), stride=2, padding=(3, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )


def make_conv_transpose_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(8, 4), stride=2, padding=(3, 1)),
        nn. ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=(7, 3), stride=1, padding=(3, 1)),
        nn. ReLU(),
    )

class Autoencoder(nn.Module):
    def __init__(self):
        super (Autoencoder, self).__init__()
        # dim reduction
        self.conv1 = make_conv_layer(1, 16)
        self.conv2 = make_conv_layer(16, 32)
        self.conv3 = make_conv_layer(32, 64)
        self.conv4 = make_conv_layer(64, 128)
        # upscaling
        self.conv5 = make_conv_transpose_layer(128, 32)
        self.conv6 = make_conv_transpose_layer(96, 16)
        self.conv7 = make_conv_transpose_layer(48, 8)
        self.conv8 = make_conv_transpose_layer(24, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(torch.cat((x5, x3), dim=1))
        x7 = self.conv7(torch.cat((x6, x2), dim=1))
        x8 = self.conv8(torch.cat((x7, x1), dim=1))

        return x8


model = Autoencoder().to(DEVICE)
# print(model(torch.rand(1, 1, 128, 128, device=DEVICE)).shape)