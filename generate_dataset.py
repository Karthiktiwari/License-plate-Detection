from tqdm import tqdm
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

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

def crop_transfrom(ann, img, extra_dim = 50):
    """Crop the original image and transform the bounding box coordinates respectively
    """
    width = ann[2]
    height = ann[3]
    x = int(ann[0])
    y = int(ann[1])
    rem_w = 448-width
    rem_h = 256-height
    dim_x1 = random.randint(extra_dim,rem_w-extra_dim)
    dim_x2 = rem_w-dim_x1
    dim_y1 = random.randint(extra_dim,rem_h-extra_dim)
    dim_y2 = rem_h-dim_y1
    
    cropped_img = img[y-dim_y1:y+dim_y2+int(height),x-dim_x1:x+dim_x2+int(width),:]
    
    transformed_ann = [dim_x1,dim_y1,width,height]
    return cropped_img, transformed_ann





def generate_data(save_dir, paths, indices, ann):
    """Generate images and text files with license plate coordinates and text indices
    Args:
        savedir (str) : Path of directory to save the images and text files
    """
    from tqdm import trange
    ctr = 0
    for i in trange(len(ann)):
        try:
            img = cv2.imread(paths[i]).copy()
            cropped_img, transformed_ann = crop_transfrom(ann[i], img)
            if(cropped_img.shape[0]>=250 and cropped_img.shape[1]>=440):
                cv2.imwrite(save_dir+'\\'+str(i)+'.jpg',cv2.resize(cropped_img, (448,256)))
                with open(save_dir+'\\'+str(i)+".txt","w") as txtfile:
                    txtfile.write(" ".join([str(c) for c in transformed_ann])+'\n')
                    txtfile.write(" ".join(indices[i]))
                    ctr += 1
        except:
            # itr = 0
            # while (itr<=200 and (ym-dim_y1<0 or ym+dim_y2<0 or xm-dim_x1<0 or xm+dim_x2<0)):
            #     dim_x1 = random.randint(20,rem_w-20)
            #     dim_x2 = rem_w-dim_x1
            #     dim_y1 = random.randint(20,rem_h-20)
            #     dim_y2 = rem_h-dim_y1
            #     itr += 1
            # if(itr<51):
            #     img = cv2.imread(paths[i])
            #     img = img[ym-dim_y1-int(h/2):ym+dim_y2+int(h/2),xm-dim_x1-int(w/2):xm+dim_x2+int(w/2),:]
            #     cv2.imwrite(save_dir+'\\'+str(i)+'.jpg',cv2.resize( img, (448,448)))
            #     cann.append([dim_x1,dim_y1,w,h])
            #     with open(save_dir+'\\'+str(i)+".txt","w") as txtfile:
            #         txtfile.write(str(dim_x1+int(w))+" "+str(dim_y1+int(h))+" "+str(w)+" "+str(h)+'\n')
            #         txtfile.write(" ".join(indices[i]))
            continue

    print(f"{ctr} examples generated ")

root = r"C:\Users\win10\Downloads\UFPR-ALPR\UFPR-ALPR dataset\Train data"
savedir = r"C:\Users\win10\Documents\images" 

paths, indices, ann = read_dir(root = root)         #Read raw data
generate_data(save_dir = savedir, paths = paths, indices = indices, ann = ann)       #Generate cropped images, transformed annotations and vectorized plate text
