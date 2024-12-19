import torch.nn as nn
from torchvision import models

# Criando Rede Neural
class ResNet18(nn.Module):
  def __init__(self, num_classes):
    super(ResNet18, self).__init__()
    # Carregando ResNet-18 pré-treinada
    self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Substituindo a camada de classificação
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, num_classes)

  # Definindo o fluxo de forward
  def forward(self, x):
    x = self.model(x)
    return x