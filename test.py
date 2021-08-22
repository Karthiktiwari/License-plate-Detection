import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from utils import generate_dataframe
from dataset import LicensePlateDataset
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

model = torch.load('regressor.pt')
data_dir = r"C:\Users\win10\Documents\images"
df = generate_dataframe(root=data_dir)
pct = 0.9
train_df = df[:int(pct * len(df))]
validation_df = df[int(pct * len(df)):]
train_df.index = [i for i in range(len(train_df))]
validation_df.index = [i for i in range(len(validation_df))]
validation_data = LicensePlateDataset(root=data_dir, df=validation_df)
model.eval()

test_indices = [random.randint(0, len(validation_df)) for i in range(6)]
fig = plt.figure(figsize=(20, 20))
columns = 2
rows = 3
for idx,i in enumerate(test_indices):
	box = model(validation_data[i]['image'].unsqueeze(dim = 0).cuda())
	# os._exit(0)
	vbox = box[0].cpu().detach().numpy()[0]
	box = box[1].cpu().detach().numpy()[0]
	image = np.array((Image.open(os.path.join(data_dir,validation_df['paths'][i]+'.jpg'))))
	fig.add_subplot(rows, columns,idx+1)
	x1 = int(box[0])
	y1 = int(box[1])
	x2 = x1 + int(box[2])
	y2 = y1 + int(box[3])
	p1,p2 = (x1,y1), (x2,y2)
	vp1, vp2 = (int(vbox[0]), int(vbox[1])) , (int(vbox[2]+vbox[0]), int(vbox[3]+vbox[1]))
	out = cv2.rectangle(image, p1, p2, color = (255,255,0), thickness = 2)
	fout = cv2.rectangle(out, vp1, vp2, color = (255,255,0), thickness = 2)
	plt.axis('off')
	plt.imshow(fout)

plt.show()
