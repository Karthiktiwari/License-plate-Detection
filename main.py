import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import generate_dataframe
from dataset import LicensePlateDataset
from model import Regressor
from tqdm import tqdm
import time
import os
# import pandas as pd


data_dir = r"C:\Users\win10\Documents\images"

df = generate_dataframe(root=data_dir)
pct = 0.9
train_df = df[:int(pct * len(df))]
validation_df = df[int(pct * len(df)):]
# df.to_csv('dataset.csv')
train_df.index = [i for i in range(len(train_df))]
validation_df.index = [i for i in range(len(validation_df))]
print(len(train_df), len(validation_df))
train_data = LicensePlateDataset(root=data_dir, df=train_df)
validation_data = LicensePlateDataset(root=data_dir, df=validation_df)

# train_data.show_examples()
# os._exit(0)
trainset = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
valset = DataLoader(validation_data, batch_size=8, shuffle=True, num_workers=0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = torchvision.models.mobilenet_v3_small(pretrained=True)
for param in base_model.parameters():
    param.required_grad = True

# base_model = nn.Sequential(*list(base_model.children())[:-1])
# print(base_model(torch.rand(1, 3, 512, 512)).shape)

model = Regressor(base_model=base_model)

model.to(device)
print(model.fc)

optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)
loss_fn = nn.MSELoss()
epochs = 50
for epoch in range(epochs):
    time.sleep(1)
    total_train_loss = 0.0
    total_val_loss = 0.0

    model.train()
    for i, batch in enumerate(tqdm(trainset)):
        feature, label = batch['image'].to(device), batch['bbox'].to(device)
        optimizer.zero_grad()
        output = model(feature).squeeze()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    for i, batch in enumerate(tqdm(valset)):
        feature, label = batch['image'].to(device), batch['bbox'].to(device)
        output = model(feature).squeeze()
        loss = loss_fn(output, label.squeeze())
        total_val_loss += loss.item()

    scheduler.step()
    print(f"Train loss on epoch {epoch + 1}={total_train_loss/len(trainset)}")
    print(f"Val loss on epoch {epoch + 1}={total_val_loss / len(valset)}")
    torch.save(model, 'regressor.pt')

print("Training complete")
torch.save(model, 'regressor.pt')



