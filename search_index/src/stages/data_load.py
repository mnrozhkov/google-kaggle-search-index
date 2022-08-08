import argparse
import json
import os
from pathlib import Path
import tensorflow_datasets as tfds
from typing import Text


import image_embeddings 
from src.utils.config import load_config


def data_load(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)
    STORAGE = config.data.data_storage_dir
    BASEDIR = config.data.data_base_dir
    DATASET = config.data.dataset_name
    PATH_IMAGES = f"{BASEDIR}/{DATASET}/{config.data.images_dir}"
    PATH_TFRECORDS = f"{BASEDIR}/{DATASET}/{config.data.tfrecords_dir}"
    
    # Prepare data folder structure
    os.makedirs(BASEDIR, exist_ok=True)
    os.makedirs(os.path.join(BASEDIR, DATASET), exist_ok=True)
    os.makedirs(PATH_IMAGES, exist_ok=True)
    os.makedirs(PATH_TFRECORDS, exist_ok=True)

    
    
    # Download a dataset
    ds, ds_info = tfds.load(config.data.dataset_name, 
                        data_dir=STORAGE,
                        split='train', with_info=True)
    
    # Save dataset metadata (dataset_info.json)
    ds_info.write_to_directory(f"{BASEDIR}/{DATASET}")
    
    # Save images    
    image_embeddings.downloader.save_examples(ds_info, ds, 1000, PATH_IMAGES)
    
    # Transform image to tf records
    image_embeddings.inference.write_tfrecord(
        image_folder=PATH_IMAGES, output_folder=PATH_TFRECORDS, num_shards=10
    )
    
    
        

    

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
