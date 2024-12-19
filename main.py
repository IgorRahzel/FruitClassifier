import torch
import numpy as np
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt
import matplotlib
from load_data import Data
from cnn_model import ResNet18
from torch.utils.data import DataLoader

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



# Visualizando Frutas
def visualize_fruits(dir_path, fruit_list, output_path="visualize_fruits.png"):
  _, axes = plt.subplots(len(fruit_list), 5, figsize=(15, 15))
  for i, fruit in enumerate(fruit_list):
    fruit_folder_path = os.path.join(dir_path, fruit)
    for j, image_file in enumerate(os.listdir(fruit_folder_path)[:5]):
      image_path = os.path.join(fruit_folder_path, image_file)
      image = plt.imread(image_path)
      axes[i, j].imshow(image)
      axes[i, j].axis('off')
      axes[i, j].set_title(fruit)
  plt.tight_layout()
  plt.savefig(output_path)  # Salva o gráfico como arquivo
  plt.close()
  print(f"Gráfico de visualização salvo em: {output_path}")

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
epochs = 5
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
criterion = nn.CrossEntropyLoss()

# Carregando Modelo
model = model.to(DEVICE)

# Treinamento e Validação
def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs, device):

  train_losses,train_accuracies = [],[]
  test_losses,test_accuracies = [],[]

  for epoch in range(epochs):
    # Definir o modo do modelo para treinamento
    model.train()
    # Inicializando a loss, acurácia e número de classificações corretas na epoch
    epoch_train_loss,epoch_train_accuracy,epoch_train_correct = 0.0,0.0,0

    # Iterando sobre o batch
    for images,labels in train_loader:
      images,labels = images.to(device),labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      # Extraindo imagens classificadas corretamente
      epoch_train_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      epoch_train_loss += loss.item()

    # Calculando Loss da epoch
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)
    # Calculando Acurácia da epoch
    epoch_train_accuracy = epoch_train_correct / len(train_loader.dataset)
    train_accuracies.append(epoch_train_accuracy)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f} - Train Accuracy: {epoch_train_accuracy:.4f}")

    # Definir o modo do modelo para avaliação
    model.eval()
    epoch_test_loss,epoch_test_accuracy,epoch_test_correct = 0.0,0.0,0
    # Desabilitar o cálculo do gradiente
    with torch.no_grad():
      for images,labels in test_loader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        # Extraindo imagens classificadas corretamente
        epoch_test_correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        loss = criterion(outputs,labels)
        epoch_test_loss += loss.item()

      # Calculando Loss da epoch
      epoch_test_loss /= len(test_loader)
      test_losses.append(epoch_test_loss)
      # Calculando Acurácia da epoch
      epoch_test_accuracy = epoch_test_correct / len(test_loader.dataset)
      test_accuracies.append(epoch_test_accuracy)
      print(f"Epoch {epoch+1}/{epochs} - Test Loss: {epoch_test_loss:.4f} - Test Accuracy: {epoch_test_accuracy:.4f}")

  return train_losses,train_accuracies,test_losses,test_accuracies

train_losses,train_accuracies,test_losses,test_accuracies = train_and_evaluate(model,train_loader,test_loader,optimizer,criterion,epochs,DEVICE)

# Salvando o modelo
torch.save(model.state_dict(),"model.pth")

# Plotando Gráficos de Acuraçia e Loss
def plot_loss_acc(train_accuracies, train_losses, test_accuracies, test_losses, output_path="lossAndAccuracy.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de Acurácia
    axes[0].plot(train_accuracies, label='Train Accuracy')
    axes[0].plot(test_accuracies, label='Test Accuracy')
    axes[0].set_title("Accuracy")
    axes[0].legend()

    # Gráfico de Loss
    axes[1].plot(train_losses, label='Train Loss')
    axes[1].plot(test_losses, label='Test Loss')
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.savefig(output_path)  # Salva o gráfico como arquivo
    plt.close()
    print(f"Gráfico de resultados salvo em: {output_path}")


# Chamando a função para plotar os gráficos de acurácia e loss
plot_loss_acc(train_accuracies, train_losses, test_accuracies, test_losses)


# Visualizando Resultados
def show_results(model, test_set, device, output_path="test_results.png"):
    model.eval()  # Set the model to evaluation mode
    images = []
    labels = []

    n = len(test_set)
    index = np.random.choice(n, 25, replace=False)  # Seleciona 25 índices aleatórios sem reposição

    for i in index:
        img, lbl = test_set[i]
        images.append(img)
        labels.append(lbl)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)

    _, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(25):
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        axes[i].imshow(image)
        predicted_class = outputs[i].argmax().item()
        actual_class = labels[i].item()
        axes[i].set_title(f"Predicted: {predicted_class}\nActual: {actual_class}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)  # Salva o gráfico como arquivo
    plt.close()
    print(f"Resultados salvos em: {output_path}")

# Chamando a função para salvar os resultados
show_results(model, test_set, DEVICE)

