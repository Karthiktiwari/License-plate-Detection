import torch
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from dataset import read_dir, generate_data
import os

root = r"C:\Users\win10\Downloads\UFPR-ALPR\UFPR-ALPR dataset\Train data"
savedir = r"C:\Users\win10\Documents\images" 

paths, indices, ann = read_dir(root = root)			#Read raw data
# generate_data(save_dir = savedir, paths = paths, indices = indices, ann = ann)			#Generate cropped images, transformed annotations and vectorized plate text

print(paths) 


