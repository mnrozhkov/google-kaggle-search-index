import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def list_all_files(directory, is_relative=True):
    
    dirpath = Path(directory).expanduser()
    assert dirpath.is_dir()
    
    file_list = []
    for x in dirpath.iterdir():
        
        if x.is_file():
            if is_relative is True:
                file_list.append(x.relative_to(dirpath.parent).__str__())
            else:
                file_list.append(x.__str__())
        elif x.is_dir():
            file_list.extend(list_all_files(x, is_relative))
    return file_list


def index_search(index, q_emb, k=5):
    xq = np.expand_dims(q_emb, 0)
    D, I = index.search(xq, k)  # actual search, D - lisdt of distances, I - list of ids 
    return D, I


def process_search_results(q, D, I, q_df):
    """
    Return: [{'id', 'file', 'label', 'label_name', 'distance', 'q_id', 'true_label'}]
    """
    
    # Get result details 
    q_results = [q_df.iloc[i].to_dict() for i in I[0]]
    [r.update({'distance':d}) for r, d in zip(q_results, D[0])]   # Add distance 
    [r.update({'query_id':q.get('id')}) for r in q_results]       # Add query image id 
    [r.update({'true_label':q.get('label')}) for r in q_results]  # Add true_label
    
    return q_results