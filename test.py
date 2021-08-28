import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from utils import generate_dataframe
from dataset import ALPRDataset
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

model = torch.load('regressor.pt')
data_dir = r"C:\Users\win10\Documents\UFPR-ALPR"
df = generate_dataframe(root=data_dir)
pct = 0.8
validation_df = df[int(pct * len(df)):]
validation_df.index = [i for i in range(len(validation_df))]
validation_data = ALPRDataset(root=data_dir, df=validation_df, S = 3, C = 2)
model.eval()
image_dir = os.path.join(data_dir, 'images')
test_indices = [random.randint(0, len(validation_df)) for i in range(6)]
fig = plt.figure(figsize=(20, 20))
columns = 2
rows = 3
for idx,i in enumerate(test_indices):
	box = model(validation_data[i]['image'].unsqueeze(dim = 0).cuda()).cpu().detach().numpy().reshape(-1,7)
	print(box)
	vehicle_idx = np.argmax(box[:, 0])
	plate_idx = np.argmin(box[:, 1])
	# os._exit(0)
	vbox = box[vehicle_idx][3:]
	box = box[plate_idx][3:]
	image = np.array((Image.open(os.path.join(image_dir,validation_df['paths'][i]+'.jpg'))))
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
	break

plt.show()
