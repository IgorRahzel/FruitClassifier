import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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



# Plotando Gráficos de Acuracia e Loss
def plot_loss_acc(train_accuracies, train_losses, test_accuracies, test_losses, output_path="loss_acc.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de Acurácia
    axes[0].plot(train_accuracies, label='Train Accuracy', marker='o')
    axes[0].plot(test_accuracies, label='Test Accuracy', marker='o')
    for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
        axes[0].annotate(f'{train_acc:.2f}', (i, train_acc), textcoords="offset points", xytext=(0, 5), ha='center')
        axes[0].annotate(f'{test_acc:.2f}', (i, test_acc), textcoords="offset points", xytext=(0, -15), ha='center')
    axes[0].set_title("Accuracy")
    axes[0].legend()

    # Gráfico de Loss
    axes[1].plot(train_losses, label='Train Loss', marker='o')
    axes[1].plot(test_losses, label='Test Loss', marker='o')
    for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
        axes[1].annotate(f'{train_loss:.2f}', (i, train_loss), textcoords="offset points", xytext=(0, 5), ha='center')
        axes[1].annotate(f'{test_loss:.2f}', (i, test_loss), textcoords="offset points", xytext=(0, -15), ha='center')
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)  # Salva o gráfico como arquivo
    plt.close()
    print(f"Gráfico de Loss e Accuracy salvo em: {output_path}")


# Visualizando Resultados no conjunto de teste
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