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
    vehicle_box = []

    for path in paths:
        with open(path[:-4]+'.txt', 'r') as t:
            lst = t.readlines()
            box.append(lst[7][:-1].rsplit(" ")[1:])
            vehicle_box.append(lst[1][:-1].rsplit(" ")[1:])
            plate_text.append(lst[6][:-1].rsplit(" ")[1:])

    box[937] = box[937][:-1]
    bbox= []
    vbox = []

    for b, vb in tqdm(zip(box,vehicle_box)):
        b = list(map(int,b))
        bbox.append(b)
        v = list(map(int, vb))
        vbox.append(v)
        
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
    vehicle_ann = np.array(vbox)

    print(f"Found {len(ann)} with annotations")

    return paths, indices, ann, vbox

def crop_transfrom(carbox, ann, img, extra_dim = 0, dim = 448):
    """Crop the original image and transform the bounding box coordinates respectively
    """
    carx, cary, carw, carh = carbox
    width = ann[2]
    height = ann[3]
    x = int(ann[0])
    y = int(ann[1])
    relx, rely = (x-carx), (y-cary)
    rem_w = dim-carw
    rem_h = dim-carh
    dim_x1 = random.randint(extra_dim,rem_w-extra_dim)
    dim_x2 = rem_w-dim_x1
    dim_y1 = random.randint(extra_dim,rem_h-extra_dim)
    dim_y2 = rem_h-dim_y1
    
    cropped_img = img[cary-dim_y1:cary+dim_y2+int(carh),carx-dim_x1:carx+dim_x2+int(carw),:]
    
    transformed_vb = [dim_x1,dim_y1,carw,carh]
    transformed_ann = [dim_x1+relx, dim_y1+rely, width, height]
    return cropped_img, transformed_ann, transformed_vb





def generate_data(save_dir, paths, indices, ann, vb):
    """Generate images and text files with license plate coordinates and text indices
    Args:
        savedir (str) : Path of directory to save the images and text files
    """
    from tqdm import trange
    ctr = 0
    dim = 768
    image_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'annotations')
    for i in trange(len(ann)):
        try:
            img = cv2.imread(paths[i]).copy()
            cropped_img, transformed_ann, transformed_vb = crop_transfrom(vb[i], ann[i], img, dim=dim)
            if(dim-cropped_img.shape[0]<=10 and dim-cropped_img.shape[1]<=10):
                cv2.imwrite(image_dir+'\\'+str(i)+'.jpg',cv2.resize(cropped_img, (dim,dim)))
                with open(labels_dir+'\\'+str(i)+".txt","w") as txtfile:
                    txtfile.write(" ".join([str(c) for c in transformed_ann])+'\n')
                    txtfile.write(" ".join([str(c) for c in transformed_vb])+'\n')
                    ctr += 1
        except:
            img = cv2.imread(paths[i]).copy()
            tries = 0
            try:
                while(dim-cropped_img.shape[0]>=10 and dim-cropped_img.shape[1]>=10):
                    if(tries>50):
                        break
                    cropped_img, transformed_ann, transformed_vb = crop_transfrom(vb[i], ann[i], img, dim=dim)
                    tries+=1
                if(tries<50):
                    cv2.imwrite(image_dir+'\\'+str(i)+'.jpg',cv2.resize(cropped_img, (dim,dim)))
                    with open(labels_dir+'\\'+str(i)+".txt","w") as txtfile:
                        txtfile.write(" ".join([str(c) for c in transformed_ann])+'\n')
                        txtfile.write(" ".join([str(c) for c in transformed_vb])+'\n')
                        ctr += 1
                else:
                    continue
            except:
                print(dim - cropped_img.shape[0], dim - cropped_img.shape[1])

    print(f"{ctr} examples generated ")

root = r"C:\Users\win10\Downloads\UFPR-ALPR\UFPR-ALPR dataset\Train data"
savedir = r"C:\Users\win10\Documents\UFPR-ALPR" 

paths, indices, ann, vb = read_dir(root = root)         #Read raw data
generate_data(save_dir = savedir, paths = paths, indices = indices, ann = ann, vb = vb)       #Generate cropped images, transformed annotations and vectorized plate text
