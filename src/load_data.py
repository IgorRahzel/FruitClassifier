import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Data(Dataset):
  def __init__(self,dir_path,training = True,fruit_list = None):
    super().__init__()
    self.dir_path = dir_path
    self.training = training

    # Extraindo o nome e o número de classes
    self.fruit_list = sorted(os.listdir(dir_path)) if fruit_list is None else fruit_list
    self.num_classes = len(self.fruit_list)
    # Criando mapa de classes para índices
    self.class_to_idx = {class_name: idx for idx,class_name in enumerate(self.fruit_list)}

    # Realizando Data Augmentation na fase de treinamento
    self.transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
    ])

    # Transformando imagens de teste para tensor e convertendo para o tamanho correto
    self.test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    ])

    # Listas para armazenar path e label correspondente a cada imagem
    self.image_paths = []
    self.labels = []

    # Extraindo o path e label das imagens e adicionando nas listas
    for fruit in self.fruit_list:
      fruit_folder_path = os.path.join(dir_path,fruit)
      for image_file in os.listdir(fruit_folder_path):
        image_path = os.path.join(fruit_folder_path,image_file)
        self.image_paths.append(image_path)
        self.labels.append(torch.tensor(self.class_to_idx[fruit], dtype = torch.long))

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self,idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path).convert("RGB")
    # Realiza Data Augmentation em caso de treino
    if self.training:
      image = self.transform(image)
    # Apenas realiza resize e converte imagem para tensor em caso de teste
    else:
      image = self.test_transform(image)
    label = self.labels[idx]
    return image,label