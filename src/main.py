import torch
import numpy as np
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt
import matplotlib
from src.load_data import Data
from src.cnn_model import ResNet18
from torch.utils.data import DataLoader
from src.training_utils import train_and_evaluate
from src.visualization_utils import visualize_fruits, plot_loss_acc, show_results

matplotlib.use('Agg')  # Para salvar o gráfico em um arquivo

# Selecionando  3 Tipos de frutas
fruit_list = ['Apple Red 1','Banana 1', 'Maracuja 1']

# Configurações
TRAINING_PATH = "archive/fruits-360_dataset_100x100/fruits-360/Training"
TEST_PATH = "archive/fruits-360_dataset_100x100/fruits-360/Test"
NUM_CLASSES = len(fruit_list)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42 # semente para reprodução de resultados

# Fixando a semente
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():  # Verifica se uma GPU está disponível no sistema.
    torch.cuda.manual_seed(SEED)  # Define a semente para os geradores de números aleatórios usados pela GPU.
    torch.cuda.manual_seed_all(SEED)  # Define a mesma semente para todas as GPUs, caso o sistema tenha múltiplas GPUs.
    torch.backends.cudnn.deterministic = True  # Força o uso de algoritmos determinísticos no backend cuDNN (para reprodutibilidade).
    torch.backends.cudnn.benchmark = False  # Desabilita a busca por algoritmos mais rápidos no cuDNN (para garantir consistência).


# Chamando a função para visualizar frutas
visualize_fruits(TRAINING_PATH,fruit_list)

# Carregando Dados
train_set = Data(TRAINING_PATH,training = True,fruit_list = fruit_list)
test_set = Data(TEST_PATH,training = False,fruit_list = fruit_list)

print(f"Total de imagens de treinamento: {len(train_set)}")
print(f"Total de imagens de teste: {len(test_set)}")

train_loader = DataLoader(train_set,batch_size = 32,shuffle = True)
test_loader = DataLoader(test_set,batch_size = 32,shuffle = False)

# Definindo Hiperparâmetros
model = ResNet18(num_classes = NUM_CLASSES)
model.to(DEVICE)
epochs = 3
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
criterion = nn.CrossEntropyLoss()

# Carregando Modelo
model = model.to(DEVICE)

# Treinando e Avaliando o Modelo
train_losses,train_accuracies,test_losses,test_accuracies = train_and_evaluate(model,train_loader,test_loader,optimizer,criterion,epochs,DEVICE)

# Salvando o modelo
torch.save(model.state_dict(),"model.pth")

# Chamando a função com dados de treinamento e teste
plot_loss_acc(train_accuracies, train_losses, test_accuracies, test_losses)


# Chamando a função para salvar os resultados
show_results(model, test_set, DEVICE)

