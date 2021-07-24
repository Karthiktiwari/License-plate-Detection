import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import generate_dataframe
from dataset import LicensePlateDataset, LicensePlateOCRDataset
from model import Regressor
from tqdm import tqdm
import time
import os
import pandas as pd


data_dir = r"C:\Users\win10\Documents\images" 

# df = generate_dataframe(root = data_dir)
# df.to_csv('dataset.csv')

df = pd.read_csv('dataset.csv')
print(df.head())
data = LicensePlateDataset(root = data_dir, df = df)
data.show_examples()
# print(data[0]['text'])
# trainset = DataLoader(data, batch_size = 8, shuffle = True, num_workers = 0)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# base_model = torchvision.models.mobilenet_v3_large(pretrained=True)
# for param in base_model.parameters():
#     param.required_grad = True


# model = Regressor(base_model = base_model)
# model.to(device)
# print(model.fc)

# optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma = 0.7)
# loss_fn = nn.MSELoss()
# epochs=15
# for epoch in range(epochs):
#     time.sleep(1)
#     total_loss = 0.0
#     total_acc=0.0
#     for i, batch in enumerate(tqdm(trainset)):
#         feature, label = batch['image'].to(device), batch['bbox'].to(device)
#         optimizer.zero_grad()
#         output =  model(feature).squeeze()
#         loss = loss_fn(output, label)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item() 
        
#     scheduler.step()
#     print(f"loss on epoch {epoch+1} = {total_loss/len(trainset)}")

# print("Training complete")
# torch.save(model, 'regressor.pt')



