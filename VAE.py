import torch
import torch.nn as nn
from hparams import *



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
        )

        self.latent_space_dim = 2
        self.mean = nn.Linear(8192, self.latent_space_dim)
        self.std = nn.Linear(8192, self.latent_space_dim)
        self.fc1 = nn.Linear(2, 8192)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )




    def forward(self, x):
        x = self.encoder(x)
        # x = (torch.randn(self.latent_space_dim, device=DEVICE) - self.mean(x)) * self.std(x)
        # x = self.fc1(x)
        # x = x.reshape(-1, 8, 32, 32)
        x = self.decoder(x)
        return x


model = Autoencoder().to(DEVICE)
# print(model(torch.rand(1, 1, 128, 128, device=DEVICE)).shape)