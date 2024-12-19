import os
import random
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn

from load_data import Data
from cnn_model import ResNet18
from training_utils import train_and_evaluate
from visualization_utils import visualize_fruits, plot_loss_acc, show_results

# Para salvar gráficos em arquivos
matplotlib.use('Agg')


# Função para configurar o ambiente e garantir reprodutibilidade
def setup_environment(seed=42):
    """Configura a semente para reprodutibilidade e define o dispositivo."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Função para configurar diretórios do projeto
def setup_directories():
    """Configura os diretórios necessários no projeto."""
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Diretório raiz
    model_dir = os.path.join(root_path, 'model')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


# Função principal de execução
def main():
    # Configurações iniciais
    SEED = 42
    DEVICE = setup_environment(SEED)
    TRAINING_PATH = "archive/fruits-360_dataset_100x100/fruits-360/Training"
    TEST_PATH = "archive/fruits-360_dataset_100x100/fruits-360/Test"
    FRUIT_LIST = ['Apple Red 1', 'Banana 1', 'Maracuja 1']
    NUM_CLASSES = len(FRUIT_LIST)
    BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 0.001

    # Diretório para salvar o modelo
    model_dir = setup_directories()

    # Visualizar exemplos de frutas
    visualize_fruits(TRAINING_PATH, FRUIT_LIST)

    # Carregar datasets
    train_set = Data(TRAINING_PATH, training=True, fruit_list=FRUIT_LIST)
    test_set = Data(TEST_PATH, training=False, fruit_list=FRUIT_LIST)
    print(f"Total de imagens de treinamento: {len(train_set)}")
    print(f"Total de imagens de teste: {len(test_set)}")

    # Carregar dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Inicializar o modelo
    model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Treinamento e avaliação
    train_losses, train_accuracies, test_losses, test_accuracies = train_and_evaluate(
        model, train_loader, test_loader, optimizer, criterion, EPOCHS, DEVICE
    )

    # Imprime a acurácia no conjunto de teste
    print(f"Acurácia no conjunto de teste: {test_accuracies[-1]:.4f}")

    # Salvar o modelo treinado
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    print(f"Modelo salvo em: {os.path.join(model_dir, 'model.pth')}")

    # Gerar gráficos de desempenho
    plot_loss_acc(train_accuracies, train_losses, test_accuracies, test_losses)

    # Salvar resultados de predições
    show_results(model, test_set, DEVICE)


# Execução do script
if __name__ == "__main__":
    main()