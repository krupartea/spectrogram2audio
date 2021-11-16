import torch
import torch.nn as nn
from hparams import *



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
        )

        self.latent_space_dim = 4
        self.linear_for_mu = nn.Linear(2048*8, self.latent_space_dim)
        self.linear_for_log_variance = nn.Linear(2048*8, self.latent_space_dim)
        self.fc1 = nn.Linear(self.latent_space_dim, 2048*8)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
        )




    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        mu = self.linear_for_mu(x)
        log_variance = self.linear_for_log_variance(x)
        std = torch.exp(log_variance)
        x = torch.randn(self.latent_space_dim, device=DEVICE)*std - mu
        x = self.fc1(x)
        x = x.reshape(-1, 16, 32, 32)
        x = self.decoder(x)
        return x, mu, log_variance


model = Autoencoder().to(DEVICE)
# print(model(torch.rand(1, 1, 128, 128, device=DEVICE))[0].shape)