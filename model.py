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

        # linear torch.Size([1, 8, 4])
        
        self.fc1 = nn.Linear(4 * SPECTROGRAM_N_FREQS // 16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)
        self.fc7 = nn.Linear(2, 4)
        self.fc8 = nn.Linear(4, 8)
        self.fc9 = nn.Linear(8, 16)
        self.fc10 = nn.Linear(16, 16)
        self.fc11 = nn.Linear(16, 32)
        self.fc12 = nn.Linear(32, 4 * SPECTROGRAM_N_FREQS // 16)

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

        #linear
        
        channels = x.shape[1]
        size = x.shape[2]
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        x = x.reshape(-1, channels, size)

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
#print(model(torch.rand(10, 1, SPECTROGRAM_N_FREQS, SPECTROGRAM_N_FREQS, device=DEVICE)).shape)
