import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import generate_dataframe
from dataset import ALPRDataset
from model import Regressor
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter


data_dir = r"C:\Users\win10\Documents\UFPR-ALPR"
# writer = SummaryWriter()
df = generate_dataframe(root=data_dir)
pct = 0.8
train_df = df[:int(pct * len(df))]
validation_df = df[int(pct * len(df)):]
NUM_OF_CLASSES = 2
GRID_SIZE = 3
IMAGE_SIZE = 768


train_df.reset_index(inplace = True, drop = True)
validation_df.reset_index(inplace = True, drop = True)
train_data = ALPRDataset(root=data_dir, df=train_df, S = GRID_SIZE, C = NUM_OF_CLASSES)
validation_data = ALPRDataset(root=data_dir, df=validation_df, S = GRID_SIZE, C = NUM_OF_CLASSES)

# train_data.show_examples()
# plt.show()

# Dataloaders
trainset = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
valset = DataLoader(validation_data, batch_size=4, shuffle=True, num_workers=0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = torchvision.models.resnet18(pretrained=True)
for param in base_model.parameters():
    param.required_grad = True

# base_model = nn.Sequential(*list(base_model.children())[:-1])
# print(base_model(torch.rand(1, 3, 512, 512)).shape[1])
# os._exit(0)

model = Regressor(base_model=base_model, S = GRID_SIZE, C = NUM_OF_CLASSES, img_size = IMAGE_SIZE)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
loss_fn = nn.MSELoss()
epochs = 50
for epoch in range(epochs):
    time.sleep(1)
    total_train_loss = 0.0
    total_val_loss = 0.0

    model.train()
    for i, batch in enumerate(tqdm(trainset)):
        feature, box = batch['image'].to(device), batch['bbox'].to(device)
        optimizer.zero_grad()
        output = model(feature)
        loss = loss_fn(output.squeeze(), box)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    for i, batch in enumerate(tqdm(valset)):
        feature, box = batch['image'].to(device), batch['bbox'].to(device)
        optimizer.zero_grad()
        output = model(feature)
        loss = loss_fn(output.squeeze(), box)
        total_val_loss += loss.item()

    scheduler.step()
    print(f"Train loss on epoch {epoch + 1}={total_train_loss/len(trainset)}")
    print(f"Val loss on epoch {epoch + 1}={total_val_loss / len(valset)}")
    # writer.add_scalar('Loss/train', total_train_loss/len(trainset), epoch)
    # writer.add_scalar('Loss/test', total_val_loss / len(valset), epoch)
# writer.flush()
print("Training complete")
torch.save(model, 'YOLO.pt')



