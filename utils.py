import pandas as pd
import os
from tqdm import tqdm

def generate_dataframe(root):
        image_paths = []
        boxes = []
        paths =[]
        vboxes = []
        texts = []
        for file in os.listdir(root):
            if(file[-3:]=='jpg'):
                paths.append(root+'\\'+file[:-4])

        for path in tqdm(paths):
            with open(path+'.txt',"r") as txt:
                lst = txt.readlines()
                image_paths.append(path.rsplit('\\')[-1])
                boxes.append(list(int(x) for x in lst[0].rsplit(" ")))
                vboxes.append(list(int(x) for x in lst[1].rsplit(" ")))
                texts.append(list(int(x) for x in lst[2].rsplit(" ")))

        df = pd.DataFrame({'paths': image_paths, 'bbox': boxes, 'vbox': vboxes, 'texts':texts})

        return df