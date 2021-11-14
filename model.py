import torch.nn as nn
import torch
from hparams import DEVICE, SPECTROGRAM_N_FREQS
from math import sqrt

def make_conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(9, 3), stride=1, padding=(4, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=(9, 3), stride=2, padding=(4, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )

def make_conv_transpose_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(8, 4), stride=2, padding=(3, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=(7, 3), stride=1, padding=(3, 1)),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )

def make_flat_conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels)
    )

def make_flat_conv_transpose_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels),
        nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels)
    )

depth1 = SPECTROGRAM_N_FREQS // 16
depth2 = SPECTROGRAM_N_FREQS // 8
depth3 = SPECTROGRAM_N_FREQS // 4
depth4 = SPECTROGRAM_N_FREQS // 2
depth5 = SPECTROGRAM_N_FREQS

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # 2d conv
        self.conv1 = make_conv_layer(1, depth1)
        self.conv2 = make_conv_layer(depth1, depth2)
        self.conv3 = make_conv_layer(depth2, depth3)
        self.conv4 = make_conv_layer(depth3, depth4)

        # 1d conv
        self.flat_conv1 = make_flat_conv_layer(depth4, depth4)
        self.flat_conv2 = make_flat_conv_layer(depth4, depth4)
        self.flat_conv3 = make_flat_conv_layer(depth4, depth4)
        self.flat_conv4 = make_flat_conv_layer(depth4, depth4)

        # 1d deconv
        self.flat_deconv1 = make_flat_conv_transpose_layer(depth4, depth4)
        self.flat_deconv2 = make_flat_conv_transpose_layer(depth4, depth4)
        self.flat_deconv3 = make_flat_conv_transpose_layer(depth4, depth4)
        self.flat_deconv4 = make_flat_conv_transpose_layer(depth4, depth4)

        # 2d deconv
        self.deconv1 = make_conv_transpose_layer(depth5, depth3)
        self.deconv2 = make_conv_transpose_layer(depth4, depth2)
        self.deconv3 = make_conv_transpose_layer(depth3, depth1)
        self.deconv4 = make_conv_transpose_layer(depth2, 1)


    def forward(self, x):
        # conv
        x = self.conv1(x)
        x1 = x
        x = self.conv2(x)
        x2 = x
        x = self.conv3(x)
        x3 = x
        x = self.conv4(x)
        x4 = x

        # flatten
        x = x.flatten(start_dim=2)

        # flat conv
        x = self.flat_conv1(x)
        x = self.flat_conv2(x)
        x = self.flat_conv3(x)
        x = self.flat_conv4(x)

        # flat deconv
        x = self.flat_deconv1(x)
        x = self.flat_deconv2(x)
        x = self.flat_deconv3(x)
        x = self.flat_deconv4(x)

        # unflatten
        side = int(sqrt(x.shape[-1]))
        x = x.reshape(-1, depth4, side, side)

        # deconv
        x = torch.cat((x, x4), dim=1)
        x = self.deconv1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.deconv2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.deconv3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.deconv4(x)

        return x

model = Autoencoder().to(DEVICE)
#print(model(torch.rand(1, 1, SPECTROGRAM_N_FREQS, SPECTROGRAM_N_FREQS, device=DEVICE)).shape)
