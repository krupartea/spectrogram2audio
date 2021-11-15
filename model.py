import torch.nn as nn
import torch
from hparams import DEVICE, SPECTROGRAM_N_FREQS
from math import sqrt

def make_conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(9, 3), stride=1, padding=(4, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(9, 3), stride=2, padding=(4, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def make_conv_transpose_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(8, 4), stride=2, padding=(3, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(7, 3), stride=1, padding=(3, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def make_flat_conv_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )

def make_flat_conv_transpose_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # 2d conv
        self.conv1 = make_conv_layer(1, SPECTROGRAM_N_FREQS // 32)
        self.conv2 = make_conv_layer(SPECTROGRAM_N_FREQS // 32, SPECTROGRAM_N_FREQS // 16)
        self.conv3 = make_conv_layer(SPECTROGRAM_N_FREQS // 16, SPECTROGRAM_N_FREQS // 8)
        self.conv4 = make_conv_layer(SPECTROGRAM_N_FREQS // 8, SPECTROGRAM_N_FREQS // 4)

        # 1d conv
        self.flat_conv1 = make_flat_conv_layer(SPECTROGRAM_N_FREQS // 4, SPECTROGRAM_N_FREQS // 8)
        self.flat_conv2 = make_flat_conv_layer(SPECTROGRAM_N_FREQS // 8, SPECTROGRAM_N_FREQS // 16)
        self.flat_conv3 = make_flat_conv_layer(SPECTROGRAM_N_FREQS // 16, SPECTROGRAM_N_FREQS // 16)
        self.flat_conv4 = make_flat_conv_layer(SPECTROGRAM_N_FREQS // 16, SPECTROGRAM_N_FREQS // 16)

        # 1d deconv
        self.flat_deconv1 = make_flat_conv_transpose_layer(SPECTROGRAM_N_FREQS // 16, SPECTROGRAM_N_FREQS // 16)
        self.flat_deconv2 = make_flat_conv_transpose_layer(SPECTROGRAM_N_FREQS // 16, SPECTROGRAM_N_FREQS // 8)
        self.flat_deconv3 = make_flat_conv_transpose_layer(SPECTROGRAM_N_FREQS // 8, SPECTROGRAM_N_FREQS // 8)
        self.flat_deconv4 = make_flat_conv_transpose_layer(SPECTROGRAM_N_FREQS // 8, SPECTROGRAM_N_FREQS // 4)

        # 2d deconv
        self.deconv1 = make_conv_transpose_layer(SPECTROGRAM_N_FREQS // 2, SPECTROGRAM_N_FREQS // 8)
        self.deconv2 = make_conv_transpose_layer(SPECTROGRAM_N_FREQS // 4, SPECTROGRAM_N_FREQS // 16)
        self.deconv3 = make_conv_transpose_layer(SPECTROGRAM_N_FREQS // 8, SPECTROGRAM_N_FREQS // 32)
        self.deconv4 = make_conv_transpose_layer(SPECTROGRAM_N_FREQS // 16, 1)


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
        x = x.reshape(-1, SPECTROGRAM_N_FREQS // 4, side, side)

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
