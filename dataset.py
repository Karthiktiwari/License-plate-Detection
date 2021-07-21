from tqdm import tqdm

def read_dir(root):
    """Function to read the images and annotations
    Args:
        root (str) : Path of the root directory
    """
    
    paths = []
    for parent_dir in tqdm(os.listdir(root)):
        for folder in os.listdir(os.path.join(root,parent_dir)):
            for file in os.listdir(os.path.join(root,parent_dir,folder)):
                file_path = os.path.join(root,parent_dir,folder,file)
                if file_path.endswith(".png"):
                    paths.append(file_path)

    len(paths)
    root = r"C:\Users\win10\Downloads\UFPR-ALPR\UFPR-ALPR dataset\Train data"
    box = []
    images = []

    for path in paths:
        with open(path[:-4]+'.txt', 'r') as t:
            lst = t.readlines()
            box.append(lst[7][:-1].rsplit(" ")[1:])
            
    save_dir = r"C:\Users\win10\Documents\images" 

    box[937] = box[937][:-1]
    bbox= []

    for b in tqdm(box):
        b = list(map(int,b))
        bbox.append(b)
        
    ann = np.array(bbox)
