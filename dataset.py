from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os

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
        image = np.array(Image.open(os.path.join(self.root,self.paths[idx]+'.jpg')))
        image = torch.as_tensor(image.reshape(3,448,448), dtype = torch.float32)

        bbox = torch.tensor(self.bboxes[idx], dtype = torch.float32)

        return {'image': image, 'bbox': bbox}

    def __len__(self,):
        return len(self.paths)