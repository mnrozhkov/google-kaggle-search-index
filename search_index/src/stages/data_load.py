import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Text
import zipfile

from src.utils.config import load_config
from search_index.src.utils import list_all_files


def data_load(config_path: Text) -> None:
    """Unzip data and prepare annotation.csv file 
    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)
    DATASET_NAME = config.data.dataset_name
    DATASET = config.data[DATASET_NAME]
    STORAGE = DATASET.data_dir
    BASEDIR = config.data.data_base_dir
    PATH_DATASET_ANNOTATIONS = Path(f"{BASEDIR}/{DATASET_NAME}/annotations.csv")
    PATH_DATASET_SOURCE = Path(STORAGE, DATASET_NAME, DATASET.data_subdir).expanduser()
    
    # Prepare data folder structure
    os.makedirs(BASEDIR, exist_ok=True)
    os.makedirs(os.path.join(BASEDIR, DATASET_NAME), exist_ok=True)
    
    # Unzip data if needed
    if DATASET.is_zipped is True:
        with zipfile.ZipFile(DATASET.data_zip_filename,"r") as zip_ref:
            zip_ref.extractall(Path(BASEDIR, DATASET_NAME))

    # Extract annotations (from dirs structure)
    dirs = [f for f in PATH_DATASET_SOURCE.iterdir() if f.is_dir()]
    frames = []
    for i, dir in enumerate(dirs):
        print(dir)
        paths = list_all_files(dir)
        frame = pd.DataFrame({
            'file': paths, 
            'label': i, 
            'label_name': dir.relative_to(PATH_DATASET_SOURCE).name,
            })
        frames.append(frame)

    # Sample and save annotations
    annotations = pd.concat(frames, axis=0)
    
    if config.data_load.num_examples != 'all':
        annotations = annotations.sample(n = config.data_load.num_examples)
        print(annotations.shape)
        
    annotations.to_csv(PATH_DATASET_ANNOTATIONS, index=False)
    

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)