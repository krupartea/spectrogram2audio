import torch
import torch.nn as nn
from hparams import DEVICE

model = nn.Sequential(
# encoder
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn. ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),

    # decoder

    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(256, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn. ReLU(),
    nn.BatchNorm2d(32),
    nn.Conv2d(32, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.Conv2d(16, 1, kernel_size=3, padding=1),
)


model = model.to(DEVICE)
#print(model(torch.rand(1, 1, 128, 128, device=DEVICE)).shape)
