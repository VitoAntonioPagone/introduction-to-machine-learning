from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

class MLP(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(1000, 500),
      #nn.Dropout(),
      nn.ReLU(),
    )
    self.regressor = nn.Linear(500, 1)

  def forward(self, x):
    x = self.layers(x)
    return self.regressor(x)

pretrain_X = pd.read_csv("drive/MyDrive/pretrain_features.csv")
pretrain_y = pd.read_csv("drive/MyDrive/pretrain_labels.csv")

pretrain_X = torch.Tensor(pretrain_X.iloc[:, 2:].values)
pretrain_y = torch.Tensor(pretrain_y.iloc[:, 1])

pretrain_dataset = torch.hstack((pretrain_X, torch.unsqueeze(pretrain_y, 1)))

pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=256)
model = MLP().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 57

model.train()
for epoch in range(epochs):
    current_loss = 0.0
    for i, data in enumerate(pretrain_dataloader):
      inputs = data[:, :-1].cuda()
      labels = data[:, -1].cuda()
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(torch.squeeze(outputs), labels)
      loss.backward()
      optimizer.step()
      current_loss += loss.item()
    print("Pre-training Epoch: {0} | Loss: {1}".format(epoch + 1, current_loss))

train_X = pd.read_csv("drive/MyDrive/train_features.csv")
train_y = pd.read_csv("drive/MyDrive/train_labels.csv")

train_X = torch.Tensor(train_X.iloc[:, 2:].values)
train_y = torch.Tensor(train_y.iloc[:, 1])

train_dataset = torch.hstack((train_X, torch.unsqueeze(train_y, 1)))

train_dataloader = DataLoader(train_dataset, batch_size=20)
for param in model.layers.parameters():
    param.requires_grad = False
model.regressor = nn.Linear(500, 1).cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.05)
epochs = 40

model.train()
for epoch in range(epochs):
    current_loss = 0.0
    for i, data in enumerate(train_dataloader):
      inputs = data[:, :-1].cuda()
      labels = data[:, -1].cuda()
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(torch.squeeze(outputs), labels)
      loss.backward()
      optimizer.step()
      current_loss += loss.item()
    print("Training Epoch: {0} | Loss: {1}".format(epoch + 1, current_loss))

test_X = pd.read_csv("drive/MyDrive/test_features.csv")
test_dataset = torch.Tensor(test_X.iloc[:, 2:].values)

test_dataloader = DataLoader(test_dataset, batch_size=512)
indices = test_X.iloc[:, 0]

model.eval()
predictions=np.zeros((0,1))
for i, data in enumerate(test_dataloader):
    with torch.no_grad():
      batch_predictions = model(data.cuda()).cpu().detach().numpy()
      predictions = np.vstack((predictions, batch_predictions))

df = pd.DataFrame(predictions)
df = pd.concat([indices, df], axis=1)
df.columns = ['Id', 'y']
df.to_csv('drive/MyDrive/submission.csv', index=False)
