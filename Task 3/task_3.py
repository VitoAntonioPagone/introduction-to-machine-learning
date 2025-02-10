from google.colab import drive
drive.mount('/content/drive')

import os
from random import shuffle
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import torch
import torch.utils.data
from torch.nn.functional import normalize
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim
from torch.autograd import Variable

def image_loader(path):
    return Image.open(path.rstrip('\n')).convert('RGB')

class TripletImageDataset(Dataset):

    def __init__(self, path, triplets_file, train):
        self.path = path
        self.train = train
        positive_triplets = []
        negative_triplets = []
        test_triplets = []
        if self.train:
          for line in open(triplets_file):
              line_array = line.split(" ")
              positive_triplets.append((line_array[0], line_array[1], line_array[2], 1))
              negative_triplets.append((line_array[0], line_array[2], line_array[1], 0))
          self.triplets = positive_triplets+negative_triplets
          shuffle(self.triplets)
        else:
          for line in open(triplets_file):
              line_array = line.split(" ")
              test_triplets.append((line_array[0], line_array[1], line_array[2]))
          self.triplets = test_triplets
        self.emb_imgs = normalize(torch.Tensor(np.loadtxt(path)), dim=1)

    def __getitem__(self, index):
        if self.train:
          path1, path2, path3, label = self.triplets[index]
        else:
          path1, path2, path3 = self.triplets[index]
        path1 = int(path1.rstrip('\n').lstrip('0') if path1.rstrip('\n')!="00000" else "0")
        path2 = int(path2.rstrip('\n').lstrip('0') if path2.rstrip('\n')!="00000" else "0")
        path3 = int(path3.rstrip('\n').lstrip('0') if path3.rstrip('\n')!="00000" else "0")
        anchor = self.emb_imgs[path1]
        positive = self.emb_imgs[path2]
        negative = self.emb_imgs[path3]
        if self.train:
          return anchor, positive, negative, label
        else:
          return anchor, positive, negative

    def __len__(self):
        return len(self.triplets)

class EmbeddingNet(nn.Module):

    def __init__(self, pretrained_net=models.vgg19(pretrained=True)):
        super(EmbeddingNet, self).__init__()
        self.features = list(pretrained_net.features)
        self.features = torch.nn.Sequential(*self.features)
        self.pooling = pretrained_net.avgpool
        self.flatten = torch.nn.Flatten()
        self.fc = pretrained_net.classifier[0]

    def forward(self, x):
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

class NeuralNetwork(nn.Module):
  def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(4096*3, 4096)
        self.hidden2 = nn.Linear(4096, 2048)
        self.hidden3 = nn.Linear(2048, 1024)
        self.hidden4 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        
  def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

#Code for building of the image embeddings, this has to be runned only once
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img_base_path = "drive/MyDrive/food/"
emb_img_base_path = "drive/MyDrive/food_embeddings/"
embeddingnet = EmbeddingNet().cuda()
emb_imgs = np.zeros((0, 4096))
for i in range(10000):
  img_file = str(i).zfill(5)+".jpg"
  img_path = img_base_path+img_file
  img = transform(image_loader(img_path))
  img = torch.unsqueeze(img, 0)
  with torch.no_grad():
    img = Variable(img.cuda())
  emb_img = np.squeeze(embeddingnet(img).cpu().detach().numpy())
  emb_imgs = np.vstack((emb_imgs, emb_img))
  print(str(i)+"° image embedded")
  print(np.expand_dims(emb_img, 0))
  print(emb_imgs.shape)
np.savetxt("drive/MyDrive/food_embeddings.txt", emb_imgs)

train_dataset = TripletImageDataset("drive/MyDrive/food_embeddings.txt", "drive/MyDrive/train_triplets.txt", True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256)

#TRAINING PER BINARY CROSS ENTROPY LOSS
model = NeuralNetwork().cuda()
model_name = "vg_19_withbce"
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay= 0.0001)
num_epoch = 8

val_loss_list = []
model.train()
counter = 0
for epoch in range(num_epoch):
  running_loss = 0.0
  loss_train = 0.0
  for batch_idx, (emb_anchor, emb_positive, emb_negative, label) in enumerate(train_dataloader):
    emb_triplet = torch.hstack((emb_anchor, emb_positive, emb_negative))
    emb_triplet = emb_triplet.cuda()
    emb_triplet = Variable(emb_triplet)
    label = torch.unsqueeze(label, 1).float().cuda()
    probs = model(emb_triplet)
    loss = criterion(probs, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss += loss.data
    loss_train_cls = torch.sum(1 * (criterion(probs, label) > 0)) / len(train_dataloader)
    loss_train += loss_train_cls.data
  running_loss /= len(train_dataloader)
  loss_train /= len(train_dataloader)
  print("Training Epoch: {0} | Loss: {1}".format(epoch + 1, running_loss))
  print("Training Epoch: {0} | Classification Loss: {1}".format(epoch + 1, loss_train))
save_path = f'drive/MyDrive/_{model_name}_epoch_{epoch + 1}.pt'
torch.save({'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            save_path)
print(f'Saved model checkpoint to {save_path}')

test_dataset = TripletImageDataset("drive/MyDrive/food_embeddings.txt", "drive/MyDrive/test_triplets.txt", False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

#TESTING PER BINARY CROSS ENTROPY LOSS
model = NeuralNetwork().cuda()
checkpoint = torch.load("drive/MyDrive/_vg_19_withbce_epoch_8.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
predictions=np.zeros((0,1))
for batch_idx, (emb_anchor, emb_positive, emb_negative) in enumerate(test_dataloader):
    emb_triplet = torch.hstack((emb_anchor, emb_positive, emb_negative))
    emb_triplet = emb_triplet.cuda()
    emb_triplet = Variable(emb_triplet)
    probs = model(emb_triplet).cpu().detach().numpy()
    predictions = np.vstack((predictions,(1*(0.5 <= probs))))

df = pd.DataFrame(predictions)
df.to_csv('drive/MyDrive/submission.csv', index=False, header=None)