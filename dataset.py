from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import cv2
import matplotlib.pyplot as plt

class LicensePlateDataset(Dataset):
    """Cropped Images with Transformed annotations and Plate text"""

    def __init__(self, root, df, transform = None):
        """
        Args:
            root (str): Path of directory
            df (DataFrame): Pandas DataFrame
        """
        self.df = df 
        self.root = root
        self.paths = df['paths']
        self.bboxes = df['bbox']


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of item to be retrieved
        Returns:
            dict: Dictionary of image, bbox, plate text
        """       
        image = np.array(Image.open(os.path.join(self.root,str(self.paths[idx])+'.jpg'))) / 255
        image = torch.as_tensor(image.reshape(3,256,448), dtype = torch.float32)

        bbox = torch.tensor(self.bboxes[idx], dtype = torch.float32)

        return {'image': image, 'bbox': bbox}

    def __len__(self,):
        return len(self.paths)

    def show_examples(self,):

        rand_list = [random.randint(0,4400) for i in range(9)]
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
            out = cv2.rectangle(image, p1, p2, color = (255,0,0), thickness = 2)
            plt.axis('off')
            plt.imshow(out)

        plt.show()


class LicensePlateOCRDataset(Dataset):
    """Cropped Images with Transformed annotations and Plate text"""

    def __init__(self, root, df, transform = None):
        """
        Args:
            root (str): Path of directory
            df (DataFrame): Pandas DataFrame
        """
        self.df = df 
        self.root = root
        self.paths = df['paths']
        self.texts = df['texts']
        self.num_classes = 36

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of item to be retrieved
        Returns:
            dict: Dictionary of image, plate text
        """       
        image = np.array(Image.open(os.path.join(self.root,self.paths[idx]+'.jpg')))
        image = torch.as_tensor(image.reshape(3,448,448), dtype = torch.float32)

        text = self.one_hot(np.array(self.texts[idx]), num_classes = self.num_classes)
        text = torch.as_tensor(text, dtype = torch.float32)
        
        return {'image': image, 'text': text}

    def one_hot(self, a, num_classes):
            return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def __len__(self,):
        return len(self.paths)
