from utils.Network import ResNet
import torch

def getModel():
    return ResNet()

def loadModel(PATH):
    model = ResNet()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    return model