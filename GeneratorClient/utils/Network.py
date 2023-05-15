import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.transform(x)
        x = self.resnet(x)
        return x

