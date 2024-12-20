import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_output_path(output_dir="imgs", output_file="output.png"):
    #Retorna o diretório raiz do projeto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_path = os.path.join(project_root, output_dir, output_file)
    # Cria o diretório de saída se não existir
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)
    return output_path


# Visualizando Frutas
def visualize_fruits(dir_path, fruit_list, output_dir="imgs"):
    # Determina o diretório absoluto do destino
    output_path = get_output_path(output_dir, 'fruits.png')
    
    # Cria o gráfico de visualização
    _, axes = plt.subplots(len(fruit_list), 5, figsize=(10, 10))
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


def annotate_plot(axis,train_data,test_data,title):
    # Adiciona anotações ao gráfico
    for i, (train, test) in enumerate(zip(train_data, test_data)):
        axis.annotate(f'{train:.2f}', (i, train), textcoords="offset points", xytext=(0, 5), ha='center')
        axis.annotate(f'{test:.2f}', (i, test), textcoords="offset points", xytext=(0, -15), ha='center')
    axis.set_title(title)
    axis.legend()


# Plotando Gráficos de Acuracia e Loss
def plot_loss_acc(train_accuracies, train_losses, test_accuracies, test_losses, output_dir="imgs"):
    # Determina o diretório absoluto do destino
    output_path = get_output_path(output_dir, 'loss_acc.png')

    # Cria o gráfico de Loss e Accuracy
    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de Acurácia
    axes[0].plot(train_accuracies, label='Train Accuracy', marker='o')
    axes[0].plot(test_accuracies, label='Test Accuracy', marker='o')
    annotate_plot(axes[0],train_accuracies,test_accuracies,"Accuracy")

    # Gráfico de Loss
    axes[1].plot(train_losses, label='Train Loss', marker='o')
    axes[1].plot(test_losses, label='Test Loss', marker='o')
    annotate_plot(axes[1],train_losses,test_losses,"Loss")

    plt.tight_layout()
    plt.savefig(output_path)  # Salva o gráfico como arquivo
    plt.close()
    print(f"Gráfico de Loss e Accuracy salvo em: {output_path}")


# Visualizando Resultados no conjunto de teste
def show_results(model, test_set, device, output_dir="imgs"):
    # Determina o diretório absoluto do destino onde o gráfico será salvo
    output_path = get_output_path(output_dir, 'results.png')
    
    model.eval()  # Coloca o modelo em modo de avaliação (desabilita dropout, batchnorm, etc.)
    
    images = []  # Lista para armazenar as imagens selecionadas
    labels = []  # Lista para armazenar os rótulos das imagens selecionadas

    # Obtém o número total de imagens no conjunto de teste
    n = len(test_set)
    
    # Seleciona aleatoriamente 25 imagens do conjunto de teste (sem reposição)
    index = np.random.choice(n, 25, replace=False)  # 25 imagens aleatórias

    # Carrega as imagens e seus rótulos correspondentes
    for i in index:
        img, lbl = test_set[i]
        images.append(img)
        labels.append(lbl)

    # Converte as listas de imagens e rótulos para tensores
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Envia as imagens e rótulos para o dispositivo correto (GPU ou CPU)
    images = images.to(device)
    labels = labels.to(device)

    # Desabilita o cálculo de gradientes para evitar uso desnecessário de memória
    with torch.no_grad():
        outputs = model(images)  # Faz a previsão do modelo para as imagens selecionadas

    # Cria um gráfico de 5x5 para visualizar as 25 imagens com suas previsões
    _, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.ravel()  # Flatten o array de eixos para facilitar a iteração

    # Exibe cada uma das 25 imagens junto com as previsões do modelo
    for i in range(25):
        image = images[i].cpu().numpy().transpose((1, 2, 0))  # Converte a imagem de tensor para numpy e rearranja os canais
        axes[i].imshow(image)  # Exibe a imagem
        predicted_class = outputs[i].argmax().item()  # Obtém a classe prevista (índice da maior saída)
        actual_class = labels[i].item()  # Obtém a classe real (rótulo)
        
        # Define o título com a classe prevista e a real
        axes[i].set_title(f"Predicted: {predicted_class}\nActual: {actual_class}")
        axes[i].axis('off')  # Desativa os eixos (para uma visualização mais limpa)

    # Ajusta o layout para que as imagens não se sobreponham
    plt.tight_layout()

    # Salva o gráfico como um arquivo de imagem
    plt.savefig(output_path)  # Salva o gráfico no caminho especificado
    plt.close()  # Fecha a figura para liberar memória

    # Exibe mensagem de confirmação sobre o local onde os resultados foram salvos
    print(f"Resultados salvos em: {output_path}")
