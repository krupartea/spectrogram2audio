import torch
import torch.nn as nn
from hparams import *



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.fc1 = nn.Linear(128, 64)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            
        )




    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder().to(DEVICE)
# print(model(torch.rand(1, 1, 128, 128, device=DEVICE)).shape)