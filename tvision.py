from torchvision.models.segmentation import deeplabv3_resnet101
import torch

DEVICE = 'cuda'

model = deeplabv3_resnet101().to(DEVICE)
model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(DEVICE)
model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)).to(DEVICE)

# with torch.no_grad():
#     model.eval()
#     print(model(torch.rand(1, 1, 128, 128, device=DEVICE)))