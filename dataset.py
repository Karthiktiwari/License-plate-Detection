from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import math
import cv2

class ALPRDataset(Dataset):
    """Cropped Images with Transformed annotations and Plate text"""

    def __init__(self, root, df, S, C, transform = None):
        """
        Args:
            root (str): Path of directory
            df (DataFrame): Pandas DataFrame
        """
        self.df = df 
        self.root = root
        self.image_dir = os.path.join(root, 'images')
        self.S = S
        self.C = C
        self.paths = df['paths']
        self.bboxes = df['bbox']
        self.vboxes = df['vbox']


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of item to be retrieved
        Returns:
            dict: Dictionary of image, bbox, plate text
        """       
        image = np.array(Image.open(os.path.join(self.image_dir,str(self.paths[idx])+'.jpg')))
        image = torch.as_tensor(image, dtype = torch.float32)
        lbox = self.bboxes[idx]
        vbox = self.vboxes[idx]
        lbox = [lbox[0], lbox[1], lbox[2], lbox[3]]
        vbox = [vbox[0], vbox[1], vbox[2], vbox[3]]
        bbox = np.zeros((self.S, self.S, self.C + 5))

        d = image.shape[1] / self.S
        bbox[math.floor(lbox[0]/d)][math.floor(lbox[1]/d)] = [0, 1, 1, lbox[0], lbox[1], lbox[2], lbox[3]]
        bbox[math.floor(vbox[0]/d)][math.floor(vbox[1]/d)] = [1, 0, 1, vbox[0], vbox[1], vbox[2], vbox[3]]

        return {'image': image.reshape(3, 768, 768), 'bbox': torch.as_tensor(bbox, dtype = torch.float32)}

    def __len__(self,):
        return len(self.paths)

    def show_examples(self,):

        rand_list = [random.randint(0,len(self.df)) for i in range(9)]
        fig = plt.figure(figsize=(10, 10))
        columns = 3
        rows = 3

        for idx, i in enumerate(rand_list):
            image = np.array((Image.open(os.path.join(self.root,self.paths[i]+'.jpg'))))
            fig.add_subplot(rows, columns,idx+1)
            x1 = int(self.bboxes[i][0])
            y1 = int(self.bboxes[i][1])
            x2 = int(self.bboxes[i][0] + self.bboxes[i][2])
            y2 = int(self.bboxes[i][1] + self.bboxes[i][3])
            p1,p2 = (x1,y1), (x2,y2)
            vx1, vy1, vx2, vy2 = self.vboxes[i][0], self.vboxes[i][1], self.vboxes[i][0]+self.vboxes[i][2], self.vboxes[i][1]+self.vboxes[i][3]
            vp1, vp2 = (vx1, vy1), (vx2, vy2)
            out = cv2.rectangle(image, p1, p2, color = (255,255,0), thickness = 2)
            fout = cv2.rectangle(out, vp1, vp2, color = (255,255,0), thickness = 2)
            plt.axis('off')
            plt.imshow(fout)

        plt.show()

