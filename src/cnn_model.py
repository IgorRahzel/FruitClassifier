import torch.nn as nn
from torchvision import models

# Criando Rede Neural
class ResNet18(nn.Module):
  def __init__(self, num_classes):
    super(ResNet18, self).__init__()
    self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Utilizando ResNet-34 pre-treinada
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, num_classes)


  def forward(self, x):
    x = self.model(x)
    return x