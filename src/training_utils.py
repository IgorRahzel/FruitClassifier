import torch
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