from tqdm import tqdm
import os
import numpy as np
import cv2
import random
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def read_dir(root):
    """Function to read the images and annotations
    Args:
        root (str) : Path of the root directory
    """
    print("Reading raw data")
    paths = []
    for parent_dir in tqdm(os.listdir(root)):
        for folder in os.listdir(os.path.join(root,parent_dir)):
            for file in os.listdir(os.path.join(root,parent_dir,folder)):
                file_path = os.path.join(root,parent_dir,folder,file)
                if file_path.endswith(".png"):
                    paths.append(file_path)

    box = []
    plate_text = []

    for path in paths:
        with open(path[:-4]+'.txt', 'r') as t:
            lst = t.readlines()
            box.append(lst[7][:-1].rsplit(" ")[1:])
            plate_text.append(lst[6][:-1].rsplit(" ")[1:])

    box[937] = box[937][:-1]
    bbox= []

    for b in tqdm(box):
        b = list(map(int,b))
        bbox.append(b)
        
    text = set()
    for i in range(len(plate_text)):
        for l in plate_text[i][0]:
            text.add(l)
    ctoi = {}
    for c in sorted(text):
        ctoi[c] = str(len(ctoi))
        
    indices = []
    for text in plate_text:
            indices.append([ctoi[c] for c in text[0]])
            
    ann = np.array(bbox)

    print(f"Found {len(ann)} with annotations")

    return paths, indices, ann

def generate_data(save_dir, paths, indices, ann):
    """Generate images and text files with license plate coordinates and text indices
    Args:
        savedir (str) : Path of directory to save the images and text files
    """
    cann = []
    from tqdm import trange
    for i in trange(len(ann)):
        w = ann[i][2]
        h = ann[i][3]
        xm = int(ann[i][0])
        ym = int(ann[i][1])
        rem_w = 448-w
        rem_h = 448-h
        try:
            dim_x1 = random.randint(30,rem_w-30)
            dim_x2 = rem_w-dim_x1
            dim_y1 = random.randint(30,rem_h-30)
            dim_y2 = rem_h-dim_y1
            img = cv2.imread(paths[i])
            img = img[ym-dim_y1-int(h/2):ym+dim_y2+int(h/2),xm-dim_x1-int(w/2):xm+dim_x2+int(w/2),:]
            cv2.imwrite(save_dir+'\\'+str(i)+'.jpg',img)
            cann.append([dim_x1+int(w),dim_y1+int(h),w,h])
            with open(save_dir+'\\'+str(i)+".txt","w") as txtfile:
                txtfile.write(str(dim_x1+int(w))+" "+str(dim_y1+int(h))+" "+str(w)+" "+str(h)+'\n')
                txtfile.write(" ".join(indices[i]))
        except:
            itr = 0
            while (itr<=200 and (ym-dim_y1-int(h/2)<0 or ym+dim_y2+int(h/2)<0 or xm-dim_x1-int(w/2)<0 or xm+dim_x2+int(w/2)<0)):
                dim_x1 = random.randint(10,rem_w-10)
                dim_x2 = rem_w-dim_x1
                dim_y1 = random.randint(10,rem_h-10)
                dim_y2 = rem_h-dim_y1
                itr += 1
            if(itr<51):
                img = cv2.imread(paths[i])
                img = img[ym-dim_y1-int(h/2):ym+dim_y2+int(h/2),xm-dim_x1-int(w/2):xm+dim_x2+int(w/2),:]
                cv2.imwrite(save_dir+'\\'+str(i)+'.jpg',img)
                cann.append([dim_x1+int(w),dim_y1+int(h),w,h])
                with open(save_dir+'\\'+str(i)+".txt","w") as txtfile:
                    txtfile.write(str(dim_x1+int(w))+" "+str(dim_y1+int(h))+" "+str(w)+" "+str(h)+'\n')
                    txtfile.write(" ".join(indices[i]))

    print(f"{len(cann)} examples generated ")



class LicensePlateDataset(Dataset):
    """Cropped Images with Transformed annotations and Plate text"""

    def __init__(self, root, df, transform = None):
        """
        Args:
            root (str): Path of directory
            df (DataFrame): Pandas DataFrame
        """
        
