import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Text

from src.utils.config import load_config
from search_index.src.utils import CustomImageDataset


def get_embeddings(config_path: Text) -> None:
    """Get embeddings fot the dataset with a model selected
    Args:
        config_path {Text}: path to config
        
    Notes: 
        The input is tfrecords and the output is embeddings
    """

    config = load_config(config_path)
    DATASET_NAME = config.data.dataset_name
    DATASET = config.data[DATASET_NAME]
    STORAGE = DATASET.data_dir
    BASEDIR = config.data.data_base_dir
    PATH_DATASET_SOURCE = Path(STORAGE, DATASET_NAME, DATASET.data_subdir).expanduser()
    PATH_EMBEDDINGS_DIR = Path(BASEDIR, DATASET_NAME, "embeddings")
    PATH_EMBEDDINGS = Path(PATH_EMBEDDINGS_DIR, "embeddings.parquet")
    PATH_EMB_MODEL = Path(config.embeddings.models_dir, config.embeddings.model_name)
    BATCH_SIZE = config.embeddings.batch_size
    PATH_DATASET_ANNOTATIONS = Path(BASEDIR, DATASET_NAME, "annotations.csv")
    
    # Prepare data folder structure
    os.makedirs(PATH_EMBEDDINGS_DIR, exist_ok=True)
    
    # Load a model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(PATH_EMB_MODEL, map_location=torch.device(device))
    model.eval()
    
    # Prepare dataset 
    data_transform = T.Compose([
        T.Resize((224,224)),
        T.ToPILImage(),
        T.ToTensor()
    ])

    ds = CustomImageDataset(
        annotations_file = PATH_DATASET_ANNOTATIONS, 
        img_dir = PATH_DATASET_SOURCE, 
        transform=data_transform, 
        target_transform=None
    )
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    annotations = pd.read_csv(PATH_DATASET_ANNOTATIONS)
    
    # Compute embeddings    
    emb_frames = []
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        output = model(inputs)
        output = output.to(device).cpu().data.numpy()
        emb_frames.append(output)
    
    # Save embeddings 
    emb_df = annotations.copy()
    emb_df['embedding'] = np.concatenate(emb_frames, axis=0).tolist()
    print(emb_df.shape)
    emb_df.to_parquet(PATH_EMBEDDINGS)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    get_embeddings(config_path=args.config)
