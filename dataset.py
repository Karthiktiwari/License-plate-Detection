import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


class LicensePlateDataset(Dataset):
    """Cropped Images with Transformed annotations and Plate text"""

    def __init__(self, root, df, transform = None):
        """
        Args:
            root (str): Path of directory
            df (DataFrame): Pandas DataFrame
        """
        self.root = root
        self.df = df 

        self.paths = df['paths']
        self.bboxes = df['bbox']


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of item to be retrieved
        Returns:
            dict: Dictionary of image, bbox, plate text
        """       
        image = Image.open(paths[idx])
        image = torch.as_tensor(image.reshape(3,448,448), dtype = torch.float32)

        bbox = self.bboxes[idx]

        return {'image': image, 'bbox': bbox}

    def __len__(self,):
        return len(self.paths)

