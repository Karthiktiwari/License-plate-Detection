import pandas as pd
import os
from tqdm import tqdm
import torch.nn as nn

def generate_dataframe(root):
        image_paths = []
        boxes = []
        paths =[]
        vboxes = []
        # texts = []
        image_dir = os.path.join(root, 'images')
        labels_dir = os.path.join(root, 'annotations')
        for file in os.listdir(image_dir):
            if(file[-3:]=='jpg'):
                paths.append(file[:-4])

        for path in tqdm(paths):
            with open(os.path.join(labels_dir, path+'.txt'), "r") as txt:
                lst = txt.readlines()
                image_paths.append(path.rsplit('\\')[-1])
                boxes.append(list(int(x) for x in lst[0].rsplit(" ")))
                vboxes.append(list(int(x) for x in lst[1].rsplit(" ")))
                # texts.append(list(int(x) for x in lst[2].rsplit(" ")))

        df = pd.DataFrame({'paths': image_paths, 'bbox': boxes, 'vbox': vboxes})
        df.sample(frac = 1, random_state=42)
        df.reset_index(inplace=True)
        return df

# class YoloLoss(nn.Module):
#     """
#     Loss according to the paper
#     """
#     def __init__(self, laambd_coord, laambd_obj, laambd_noobj):
#         """
#         Intitialize constants like in the paper
#         Args:
#             laambd_coord (int): Constant for penalizing coordinates regression loss
#             laambd_obj (int): Constant for penalizing probabilities when object is present
#             laambd_noobj (int): Constant for penalizing probabilities when object is present

#         """
#         super(YoloLoss, self).__init__()

#         self.laambd_coord = laambd_coord
#         self.laambd_obj = laambd_obj
#         self.laambd_noobj = laambd_noobj

#     def forward(self, x):
#         