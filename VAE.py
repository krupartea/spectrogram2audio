import torch
import torch.nn as nn
from torchaudio.transforms import Resample
from hparams import *
import torch.nn.functional as F



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
        )

        self.latent_space_dim = 6
        self.mean = nn.Linear(8192, self.latent_space_dim)
        self.std = nn.Linear(8192, self.latent_space_dim)
        self.fc_mu = nn.Linear(64*32*32, self.latent_space_dim)
        self.fc_log_var = nn.Linear(64*32*32, self.latent_space_dim)
        self.fc_to_shape = nn.Linear(self.latent_space_dim, 64*32*32)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    def _reparameterize(self, x):
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        x = self.fc_to_shape(z)
        x = F.relu(x)
        x = x.reshape(-1, 64, 32, 32)

        return x, mu, log_var


    def forward(self, x):
        x = self.encoder(x)
        x, mu, log_var = self._reparameterize(x)
        x = self.decoder(x)
        return x, mu, log_var


model = Autoencoder().to(DEVICE)
# print(model(torch.rand(1, 1, 128, 128, device=DEVICE))[0].shape)