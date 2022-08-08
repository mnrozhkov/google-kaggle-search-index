import argparse
import image_embeddings 
import os
import pandas as pd
from typing import Text


from src.utils.config import load_config


def get_embeddings(config_path: Text) -> None:
    """Get embeddings fot the dataset with a model selected
    Args:
        config_path {Text}: path to config
        
    Notes: 
        The input is tfrecords and the output is embeddings
    """

    config = load_config(config_path)
    BASEDIR = config.data.data_base_dir
    DATASET = config.data.dataset_name
    PATH_TFRECORDS = f"{BASEDIR}/{DATASET}/{config.data.tfrecords_dir}"
    PATH_EMBEDDINGS = f"{BASEDIR}/{DATASET}/{config.data.embeddings_dir}"

    os.makedirs(PATH_EMBEDDINGS , exist_ok=True)
    
    # Build embeddings
    image_embeddings.inference.run_inference(
        tfrecords_folder=PATH_TFRECORDS, output_folder=PATH_EMBEDDINGS, batch_size=1000
    )


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    get_embeddings(config_path=args.config)
